import cv2
import torch
import mediapipe as mp
import time
import threading
from morse_dict import get_char
from blink_predictor import SmartPredictor
from audio_utils import setup_audio, start_tts_thread, speak_text_async
from vision_utils import load_vision_model, preprocess_eye, get_eye_roi

MODEL_PATH = "./results/tuning_results/best_hyper_model.pth"
OUTPUT_FILE = "output.txt"  

THRESHOLD_CLOSE, THRESHOLD_OPEN = 0.40, 0.60   
MIN_INTENTIONAL_BLINK = 0.30     
MAX_DOT_DURATION = 1.20          
SUPER_DASH_DURATION = 2.20       
CHAR_PAUSE_THRESHOLD = 3.0
EYE_OPEN_GRACE_FRAMES = 2 

LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]

def save_text_to_file(text):
    with open(OUTPUT_FILE, "a") as f: 
        f.write(text)

class VideoStream:
    def __init__(self, src=0):
        self.stream = cv2.VideoCapture(src)
        self.stream.set(cv2.CAP_PROP_BUFFERSIZE, 1) 
        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.stream.set(cv2.CAP_PROP_FPS, 30)
        
        (self.grabbed, self.frame) = self.stream.read()
        self.stopped = False

    def start(self):
        threading.Thread(target=self.update, daemon=True).start()
        return self

    def update(self):
        while True:
            if self.stopped:
                return
            (self.grabbed, self.frame) = self.stream.read()

    def read(self):
        return self.frame

    def stop(self):
        self.stopped = True
        self.stream.release()

class BackgroundPredictor:
    def __init__(self, predictor):
        self.predictor = predictor
        self.latest_text = ""
        self.current_suggestion = ""
        self.is_running = True
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()

    def _worker(self):
        last_processed = ""
        while self.is_running:
            if self.latest_text != last_processed:
                text_to_process = self.latest_text
                suggestion = self.predictor.get_suggestion(text_to_process)
                
                if self.latest_text == text_to_process:
                    self.current_suggestion = suggestion
                    
                last_processed = text_to_process
            time.sleep(0.05) 

    def update_text(self, text):
        if self.latest_text != text:
            self.latest_text = text
            self.current_suggestion = "" 

    def get_suggestion(self):
        return self.current_suggestion


def draw_ui(frame, current_morse, decoded_message, current_suggestion, live_feedback):
    h, w, _ = frame.shape
    cv2.rectangle(frame, (0, h - 200), (w, h), (20, 20, 20), -1)
    cv2.putText(frame, f"Action: {live_feedback}", (20, h - 160), 1, 1.5, (0, 255, 255), 2)
    cv2.putText(frame, f"Input: {current_morse}", (20, h - 120), 1, 2, (255, 255, 0), 2)
    
    lines = decoded_message.split('\n')[-3:]
    for i, line in enumerate(lines):
        base_text = f"> {line}"
        cv2.putText(frame, base_text, (20, h - 80 + (i * 30)), 1, 1.5, (255, 255, 255), 2)
        
        if i == len(lines) - 1 and current_suggestion:
            text_size = cv2.getTextSize(base_text, 1, 1.5, 2)[0]
            offset_x = 20 + text_size[0]
            cv2.putText(frame, f"{current_suggestion}", (offset_x, h - 80 + (i * 30)), 1, 1.5, (150, 150, 150), 2)

def main():
    model = load_vision_model(MODEL_PATH)
    raw_predictor = SmartPredictor()
    ai_engine = BackgroundPredictor(raw_predictor)
    
    start_tts_thread()
    sound_dot, sound_dash, sound_accept = setup_audio()
    face_mesh = mp.solutions.face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=False)
    
    cap = VideoStream(0).start()
    time.sleep(1.0) 
    
    blink_start_time, last_open_time, is_eye_closed = 0, time.time(), False
    first_open_time = 0 
    current_morse, decoded_message, live_feedback = "", "", ""
    
    beep_dot_played = False
    beep_dash_played = False
    beep_super_played = False
    open_frame_count = 0 
    
    cv2.namedWindow('BlinkSpeak', cv2.WINDOW_NORMAL)
    print("System Ready!")

    while True: 
        frame = cap.read()
        if frame is None: 
            continue
            
        frame = cv2.flip(frame, 1)
        results = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark
            left_t = preprocess_eye(get_eye_roi(frame, landmarks, LEFT_EYE))
            right_t = preprocess_eye(get_eye_roi(frame, landmarks, RIGHT_EYE))
            
            if left_t is not None and right_t is not None:
                with torch.no_grad():
                    eye_batch = torch.cat((left_t, right_t), dim=0)
                    avg_pred = torch.sigmoid(model(eye_batch)).mean().item()
                
                # --- EYE CLOSE LOGIC ---
                if not is_eye_closed and avg_pred < THRESHOLD_CLOSE:
                    is_eye_closed, blink_start_time = True, time.time()
                    beep_dot_played = False
                    beep_dash_played = False
                    beep_super_played = False
                    open_frame_count = 0 
                
                # --- EYE OPEN LOGIC ---
                elif is_eye_closed and avg_pred > THRESHOLD_OPEN:
                    if open_frame_count == 0:
                        first_open_time = time.time() 

                    open_frame_count += 1
                    if open_frame_count >= EYE_OPEN_GRACE_FRAMES:
                        is_eye_closed = False
                        dur = first_open_time - blink_start_time 
                        last_open_time = time.time()
                        open_frame_count = 0 
                        
                        if dur > SUPER_DASH_DURATION: 
                            suggestion = ai_engine.get_suggestion()
                            if suggestion:
                                if suggestion.startswith(' ') and decoded_message.endswith(' '):
                                    decoded_message += suggestion[1:]
                                else:
                                    decoded_message += suggestion
                                if not decoded_message.endswith(' '):
                                    decoded_message += " "
                                ai_engine.update_text(decoded_message)
                                
                            current_morse = "" 
                            live_feedback = "AI WORD ACCEPTED!"
                        elif dur >= MAX_DOT_DURATION: 
                            current_morse += "-"
                        elif dur > MIN_INTENTIONAL_BLINK: 
                            current_morse += "."
                
                elif is_eye_closed and avg_pred <= THRESHOLD_OPEN:
                    open_frame_count = 0
                
                # --- LIVE UI & AUDIO PROGRESS ---
                if is_eye_closed:
                    curr_dur = time.time() - blink_start_time
                    
                    if curr_dur > SUPER_DASH_DURATION:
                        live_feedback = "RELEASE TO ACCEPT" 
                        if not beep_super_played:
                            sound_accept.play() 
                            beep_super_played = True
                    elif curr_dur >= MAX_DOT_DURATION:
                        live_feedback = "DASH (-)"
                        if not beep_dash_played:
                            sound_dash.play()   
                            beep_dash_played = True
                    elif curr_dur > MIN_INTENTIONAL_BLINK:
                        live_feedback = "DOT (.)"
                        if not beep_dot_played:
                            sound_dot.play()    
                            beep_dot_played = True

        if not is_eye_closed and current_morse != "" and (time.time() - last_open_time) > CHAR_PAUSE_THRESHOLD:
            char = get_char(current_morse)
            
            if char == '[BACKSPACE]': 
                decoded_message = decoded_message[:-1]
                ai_engine.update_text(decoded_message)
            elif char == '[CLEAR]': 
                decoded_message = ""
                ai_engine.update_text("")
            elif char == '[SPACE]': 
                decoded_message += " "
                ai_engine.update_text(decoded_message)
            elif char == '[NEWLINE]': 
                save_text_to_file(decoded_message + "\n")
                speak_text_async(decoded_message) 
                decoded_message = ""
                ai_engine.update_text("")
            elif char != '?': 
                decoded_message += char.lower()
                ai_engine.update_text(decoded_message)
                
            current_morse, live_feedback = "", "Char Added"

        current_suggestion = ai_engine.get_suggestion()
        
        draw_ui(frame, current_morse, decoded_message, current_suggestion, live_feedback)
        cv2.imshow('BlinkSpeak', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break
        
    cap.stop() 
    cv2.destroyAllWindows()

if __name__ == "__main__": main()