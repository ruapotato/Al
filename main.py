import logging
import queue
import threading
import time
import re
from Al_brain import Al_brain
from Al_ears import Al_ears
from Al_voice import Al_voice

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

class Al:
    def __init__(self):
        self.transcription_queue = queue.Queue()
        self.response_queue = queue.Queue()
        self.ears = Al_ears(self.transcription_queue)
        self.brain = Al_brain()
        self.voice = Al_voice()
        self.is_running = False
        self.is_ready = False

    def run(self):
        self.is_running = True
        logging.info("Al is initializing...")
        
        conversation_thread = threading.Thread(target=self.conversation_control)
        conversation_thread.start()
        
        listening_thread = threading.Thread(target=self.ears.listen)
        listening_thread.start()
        
        # Wait for initialization
        while not self.is_ready and self.is_running:
            time.sleep(0.1)
        
        if self.is_running:
            logging.info("Al is ready and listening... (Press Ctrl+C to stop)")
        
        try:
            while self.is_running:
                time.sleep(0.1)
        except KeyboardInterrupt:
            logging.info("Keyboard interrupt received. Stopping Al.")
            self.stop()
        
        conversation_thread.join()
        listening_thread.join()

    def stop(self):
        self.is_running = False
        self.ears.stop()
        self.voice.stop_speaking()
        logging.info("Al stopped.")

    def conversation_control(self):
        logging.debug("Conversation control thread started")
        while self.is_running:
            try:
                transcription = self.transcription_queue.get(timeout=0.1)
                
                if transcription is None or transcription == "Speech recognition could not understand audio":
                    continue  # Skip without saying anything

                logging.info(f"\nUser: {transcription}")
                
                self.voice.stop_speaking()
                
                if transcription.lower().startswith("change model to "):
                    model_name = transcription.lower().replace("change model to ", "").strip()
                    response = self.brain.change_model(model_name)
                    self.voice.speak(response)
                elif transcription.lower() == "list models":
                    response = self.brain.list_models()
                    self.voice.speak(response)
                else:
                    response_stream = self.brain.generate_response(transcription)
                    self.stream_response(response_stream)
                
                print("\n")  # New line after response
            except queue.Empty:
                pass
        logging.debug("Conversation control thread ended")

    def stream_response(self, response_stream):
        full_response = ""
        current_segment = ""

        for chunk in response_stream:
            if not self.is_running:
                logging.debug("Stopping response generation because Al is no longer running")
                break

            full_response += chunk
            current_segment += chunk
            logging.debug(f"Received chunk: {chunk}")

            if re.search(r'[.!?,]', chunk):
                logging.debug(f"Speaking segment: {current_segment.strip()}")
                self.voice.speak(current_segment.strip())
                logging.info(f"Al (partial): {current_segment.strip()}")
                current_segment = ""

        if current_segment:
            logging.debug(f"Speaking final segment: {current_segment.strip()}")
            self.voice.speak(current_segment.strip())
            logging.info(f"Al (partial): {current_segment.strip()}")

        logging.info(f"Al (full): {full_response.strip()}")
if __name__ == "__main__":
    al = Al()
    try:
        al.run()
    except KeyboardInterrupt:
        logging.info("Keyboard interrupt received. Stopping Al.")
        al.stop()
