import numpy as np
import pygame
import pyttsx3
import threading
import queue

speech_queue = queue.Queue()

def tts_worker():
    engine = pyttsx3.init()
    voices = engine.getProperty('voices')
    if len(voices) > 1:
        engine.setProperty('voice', voices[1].id) # Hazel
    engine.setProperty('rate', 150)
    
    while True:
        text = speech_queue.get()
        if text:
            print(f"\n[TTS Engine Speaking]: {text}")
            engine.say(text)
            engine.runAndWait()
        speech_queue.task_done()

def start_tts_thread():
    tts_thread = threading.Thread(target=tts_worker, daemon=True)
    tts_thread.start()

def speak_text_async(text):
    speech_queue.put(text.strip())

def setup_audio():
    pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)
    def generate_beep(frequency, duration_ms):
        sample_rate = 44100
        n_samples = int(round(duration_ms * (sample_rate / 1000.0)))
        t = np.linspace(0, duration_ms / 1000.0, n_samples, False)
        wave = np.sin(frequency * t * 2 * np.pi)
        
        fade = int(sample_rate * 0.01)
        if n_samples > 2 * fade:
            wave[:fade] *= np.linspace(0, 1, fade)
            wave[-fade:] *= np.linspace(1, 0, fade)
            
        audio = np.int16(wave * 32767 * 0.3) 
        stereo_audio = np.column_stack((audio, audio))
        return pygame.sndarray.make_sound(stereo_audio)

    sound_dot = generate_beep(800, 100)      # Short, mid-pitch
    sound_dash = generate_beep(800, 350)     # Long, mid-pitch
    sound_accept = generate_beep(1200, 200)  # Short, high-pitch
    
    return sound_dot, sound_dash, sound_accept