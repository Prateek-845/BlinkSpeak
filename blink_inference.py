import cv2
import torch
import torch.nn as nn
import numpy as np
import mediapipe as mp
import time
from morse_dict import get_char

MODEL_PATH = "./tuning_results/best_hyper_model.pth"
IMG_SIZE = 128

# HYSTERESIS THRESHOLDS
THRESHOLD_CLOSE = 0.40  
THRESHOLD_OPEN = 0.60   

# TIMING LOGIC 
MIN_INTENTIONAL_BLINK = 0.20 
MAX_DOT_DURATION = 0.70    
CHAR_PAUSE_THRESHOLD = 3.0  

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ConvNet(nn.Module):
    def __init__(self, dropout_rate=0.4, dense_units=128): 
        super(ConvNet, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2, 2))
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2, 2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2, 2)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(), nn.MaxPool2d(2, 2)
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(256, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(), nn.MaxPool2d(2, 2)
        )
        self.conv_final = nn.Sequential(
            nn.Conv2d(512, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU()
        )
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.classifier = nn.Sequential(
            nn.Linear(512, dense_units), nn.ReLU(), nn.BatchNorm1d(dense_units),
            nn.Dropout(dropout_rate), nn.Linear(dense_units, 1)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv_final(x)
        x = self.adaptive_pool(x)
        x = self.flatten(x)
        x = self.classifier(x)
        return x

def load_model():
    model = ConvNet(dropout_rate=0.4, dense_units=128).to(device)
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        print("Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {e}")
        exit()
    model.eval()
    return model

def preprocess_eye(eye_img):
    try:
        gray = cv2.cvtColor(eye_img, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (IMG_SIZE, IMG_SIZE))
        normalized = resized / 255.0
        tensor = torch.tensor(normalized, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        return tensor.to(device)
    except:
        return None

def get_eye_roi(frame, landmarks, eye_indices, padding=5):
    h, w, _ = frame.shape
    x_coords = [int(landmarks[idx].x * w) for idx in eye_indices]
    y_coords = [int(landmarks[idx].y * h) for idx in eye_indices]
    min_x, max_x = max(0, min(x_coords) - padding), min(w, max(x_coords) + padding)
    min_y, max_y = max(0, min(y_coords) - padding), min(h, max(y_coords) + padding)
    return frame[min_y:max_y, min_x:max_x]

def main():
    model = load_model()
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True)
    LEFT_EYE = [362, 385, 387, 263, 373, 380]
    RIGHT_EYE = [33, 160, 158, 133, 153, 144]
    
    cap = cv2.VideoCapture(0)
    blink_start_time = 0
    last_open_time = time.time()
    is_eye_closed = False
    
    current_morse = ""
    decoded_message = ""
    live_feedback = "" 
    eye_state_text = "OPEN"

    print("System Ready - Blink to type")

    # WINDOW CONFIGURATION 
    window_name = 'BlinkSpeak - Configured'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1000, 750) 

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)
        frame_h, frame_w, _ = frame.shape
        
        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark
            left_t = preprocess_eye(get_eye_roi(frame, landmarks, LEFT_EYE))
            right_t = preprocess_eye(get_eye_roi(frame, landmarks, RIGHT_EYE))
            
            if left_t is not None and right_t is not None:
                with torch.no_grad():
                    avg_pred = (torch.sigmoid(model(left_t)).item() + torch.sigmoid(model(right_t)).item()) / 2

                if not is_eye_closed:
                    if avg_pred < THRESHOLD_CLOSE:
                        is_eye_closed = True
                        blink_start_time = time.time()
                
                else: 
                    if avg_pred > THRESHOLD_OPEN:
                        is_eye_closed = False
                        duration = time.time() - blink_start_time
                        last_open_time = time.time()
                        
                        if MIN_INTENTIONAL_BLINK < duration < MAX_DOT_DURATION:
                            current_morse += "."
                        elif duration >= MAX_DOT_DURATION:
                            current_morse += "-"
                        
                        live_feedback = "" 
                
                eye_state_text = "CLOSED" if is_eye_closed else "OPEN"

                if is_eye_closed:
                    current_duration = time.time() - blink_start_time
                    if current_duration < MIN_INTENTIONAL_BLINK:
                        live_feedback = "IGNORING NOISE"
                    elif current_duration < MAX_DOT_DURATION:
                        live_feedback = "DOT (.)"
                    else:
                        live_feedback = "DASH (-)"

            for idx in LEFT_EYE + RIGHT_EYE:
                x, y = int(landmarks[idx].x * frame_w), int(landmarks[idx].y * frame_h)
                cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)

        time_since_open = time.time() - last_open_time
        if not is_eye_closed and current_morse != "" and time_since_open > CHAR_PAUSE_THRESHOLD:
            char = get_char(current_morse)
            
            if char == '[BACKSPACE]':
                decoded_message = decoded_message[:-1]
            elif char == '[CLEAR]':
                decoded_message = ""
            elif char == '[SPACE]':
                decoded_message += " "
            elif char == '[NEWLINE]':
                decoded_message += "\n"
            elif char != '?':
                decoded_message += char
                
            current_morse = ""
            live_feedback = "Char Added"
        
        #  UI DISPLAY 
        cv2.rectangle(frame, (0, frame_h - 200), (frame_w, frame_h), (20, 20, 20), -1)
        
        state_color = (0, 0, 255) if is_eye_closed else (0, 255, 0)
        feedback_color = (128, 128, 128) # Gray for Ignoring
        if "DOT" in live_feedback: feedback_color = (0, 255, 255) # Yellow
        if "DASH" in live_feedback: feedback_color = (0, 165, 255) # Orange
        
        cv2.putText(frame, f"{eye_state_text}", (20, frame_h - 160), cv2.FONT_HERSHEY_SIMPLEX, 0.8, state_color, 2)
        cv2.putText(frame, f"{live_feedback}", (150, frame_h - 160), cv2.FONT_HERSHEY_SIMPLEX, 1, feedback_color, 3)
        cv2.putText(frame, f"Input: {current_morse}", (20, frame_h - 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
        
        # Multi-line Message Rendering
        lines = decoded_message.split('\n')
        start_y = frame_h - 80
        line_height = 30
        
        visible_lines = lines[-3:] 
        
        for i, line in enumerate(visible_lines):
            y_pos = start_y + (i * line_height)
            prefix = "Msg: " if i == 0 and len(visible_lines) == len(lines) else "> "
            cv2.putText(frame, f"{prefix}{line}", (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

        cv2.imshow(window_name, frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
