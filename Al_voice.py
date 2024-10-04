import subprocess
import logging
import threading

class Al_voice:
    def __init__(self):
        self.current_process = None
        self.speech_lock = threading.Lock()

    def speak(self, text):
        with self.speech_lock:
            logging.debug(f"Starting to speak: {text}")
            try:
                subprocess.run(["espeak", text], check=True)
                logging.debug(f"Finished speaking: {text}")
            except subprocess.CalledProcessError as e:
                logging.error(f"Failed to speak text: {e}")
            except Exception as e:
                logging.error(f"Unexpected error in speak method: {e}", exc_info=True)

    def stop_speaking(self):
        if self.current_process and self.current_process.poll() is None:
            logging.debug("Terminating ongoing speech")
            self.current_process.terminate()
            self.current_process.wait()
            self.current_process = None
        else:
            logging.debug("No ongoing speech to terminate")

    def is_speaking(self):
        return self.current_process is not None and self.current_process.poll() is None
