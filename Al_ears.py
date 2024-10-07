import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import pyaudio
import numpy as np
import logging
import time
import re
from Al_eyes import Al_eyes

class Al_ears:
    def __init__(self, transcription_queue):
        logging.debug("Initializing Al_ears")
        self.transcription_queue = transcription_queue
        self.is_running = False
        self.is_paused = False
        
        # Initialize Whisper model
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        model_id = "openai/whisper-large-v3-turbo"
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
        )
        self.model.to(device)

        processor = AutoProcessor.from_pretrained(model_id)

        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=self.model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            torch_dtype=torch_dtype,
            device=device,
        )

        # Initialize PyAudio
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(format=pyaudio.paFloat32, channels=1, rate=16000, input=True, frames_per_buffer=1024)
        
        logging.debug("Al_ears initialized")

    def listen(self):
        self.is_running = True
        logging.debug("Starting warm-up period...")
        time.sleep(5)  # Warm-up period
        logging.debug("Warm-up complete, now listening...")
        
        while self.is_running:
            try:
                audio_data = self.record_audio(duration=5)  # Record 5 seconds of audio
                text = self.transcribe_audio(audio_data)
                
                if text and self.is_valid_transcription(text):
                    logging.info(f"Transcription: {text}")
                    
                    if self.is_paused:
                        if "al" in text.lower() or "resume" in text.lower():
                            self.is_paused = False
                            self.transcription_queue.put("Resumed")
                            if "al" in text.lower() and len(text.split()) > 1:
                                processed_text = Al_eyes.process_keywords(text)
                                self.transcription_queue.put(processed_text)
                    else:
                        if "pause" in text.lower():
                            self.is_paused = True
                            self.transcription_queue.put("Paused")
                        else:
                            processed_text = Al_eyes.process_keywords(text)
                            self.transcription_queue.put(processed_text)
                    
            except Exception as e:
                logging.error(f"Error in listening: {e}")

    def record_audio(self, duration):
        frames = []
        for _ in range(0, int(16000 / 1024 * duration)):
            data = self.stream.read(1024)
            frames.append(np.frombuffer(data, dtype=np.float32))
        return np.concatenate(frames)

    def transcribe_audio(self, audio_data):
        try:
            result = self.pipe(audio_data)
            return result["text"].strip()
        except Exception as e:
            logging.error(f"Error in transcription: {e}")
            return ""

    def is_valid_transcription(self, text):
        # Strip whitespace and convert to lowercase
        cleaned_text = text.strip().lower()
        
        # Check if the text consists only of periods
        if re.match(r'^\.+$', cleaned_text):
            return False
        
        # Check if the text has any non-whitespace characters
        if len(cleaned_text) == 0:
            return False
        
        return True

    def stop(self):
        self.is_running = False
        self.stream.stop_stream()
        self.stream.close()
        self.p.terminate()

    def pause(self):
        self.is_paused = True

    def resume(self):
        self.is_paused = False
