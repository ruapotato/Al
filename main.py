import logging
import queue
import threading
import time
from Al_brain import Al_brain
from Al_ears import Al_ears
from Al_voice import Al_voice

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class Al:
    def __init__(self):
        self.transcription_queue = queue.Queue()
        self.ears = Al_ears(self.transcription_queue)
        self.brain = Al_brain()
        self.voice = Al_voice()
        self.is_running = False
        self.current_response_thread = None
        self.stop_event = threading.Event()

    def run(self):
        self.is_running = True
        logging.info("Al is initializing...")
        
        conversation_thread = threading.Thread(target=self.conversation_control)
        conversation_thread.start()
        
        listening_thread = threading.Thread(target=self.ears.listen)
        listening_thread.start()
        
        logging.info("Al is ready and listening...")
        
        try:
            while self.is_running:
                time.sleep(0.1)
        except KeyboardInterrupt:
            logging.info("Keyboard interrupt received. Stopping Al.")
            self.stop()
        
        conversation_thread.join()
        listening_thread.join()

    def stop(self):
        logging.info("Stopping Al...")
        self.is_running = False
        self.stop_event.set()
        self.ears.stop()
        self.voice.stop_speaking()
        if self.current_response_thread and self.current_response_thread.is_alive():
            self.current_response_thread.join(timeout=1)
        logging.info("Al stopped.")

    def conversation_control(self):
        while self.is_running:
            try:
                transcription = self.transcription_queue.get(timeout=0.1)
                
                if transcription is None or transcription == "Speech recognition could not understand audio":
                    continue

                print(f"\nYou: {transcription}")
                
                # Stop any ongoing response and speech
                self.stop_event.set()
                self.voice.stop_speaking()
                if self.current_response_thread and self.current_response_thread.is_alive():
                    self.current_response_thread.join(timeout=1)
                
                # Reset the stop event and start a new response
                self.stop_event.clear()
                self.current_response_thread = threading.Thread(target=self.generate_and_stream_response, args=(transcription,))
                self.current_response_thread.start()
                
            except queue.Empty:
                pass

    def generate_and_stream_response(self, transcription):
        response_stream = self.brain.generate_response(transcription)
        self.stream_response(response_stream)

    def stream_response(self, response_stream):
        full_response = ""
        current_segment = ""

        print("Al: ", end="", flush=True)

        for chunk in response_stream:
            if self.stop_event.is_set():
                print("\n[Response interrupted]")
                break

            full_response += chunk
            current_segment += chunk
            print(chunk, end="", flush=True)

            if chunk.endswith(('.', '!', '?')) or len(current_segment) > 50:
                if self.stop_event.is_set():
                    print("\n[Response interrupted]")
                    break
                self.voice.speak(current_segment.strip())
                current_segment = ""

        if current_segment and not self.stop_event.is_set():
            self.voice.speak(current_segment.strip())

        print("\n")

if __name__ == "__main__":
    al = Al()
    try:
        al.run()
    except KeyboardInterrupt:
        logging.info("Keyboard interrupt received. Stopping Al.")
        al.stop()
