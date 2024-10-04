import pyaudio
import speech_recognition as sr
import numpy as np
import logging
import time

class Al_ears:
    def __init__(self, transcription_queue):
        logging.debug("Initializing Al_ears")
        self.transcription_queue = transcription_queue
        self.recognizer = sr.Recognizer()
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=1024)
        self.is_running = False
        self.is_paused = False
        logging.debug("Al_ears initialized")

    def listen(self):
        self.is_running = True
        logging.debug("Starting warm-up period...")
        time.sleep(5)  # Warm-up period
        logging.debug("Warm-up complete, now listening...")
        
        while self.is_running:
            try:
                with sr.Microphone() as source:
                    audio = self.recognizer.listen(source, timeout=10, phrase_time_limit=15)
                    try:
                        text = self.recognizer.recognize_google(audio)
                        logging.info(f"Transcription: {text}")
                        
                        if self.is_paused:
                            if "al" in text.lower() or "resume" in text.lower():
                                self.is_paused = False
                                self.transcription_queue.put("Resumed")
                                if "al" in text.lower() and len(text.split()) > 1:
                                    self.transcription_queue.put(text)
                        else:
                            if "pause" in text.lower():
                                self.is_paused = True
                                self.transcription_queue.put("Paused")
                            else:
                                self.transcription_queue.put(text)
                        
                    except sr.UnknownValueError:
                        logging.warning("Speech recognition could not understand audio")
                    except sr.RequestError as e:
                        logging.error(f"Could not request results from speech recognition service; {e}")
            except Exception as e:
                logging.error(f"Error in listening: {e}")

    def stop(self):
        self.is_running = False
        self.stream.stop_stream()
        self.stream.close()
        self.p.terminate()

    def pause(self):
        self.is_paused = True

    def resume(self):
        self.is_paused = False
