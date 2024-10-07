import subprocess
import logging
import threading
import os
import signal

class Al_voice:
    def __init__(self):
        self.current_process = None
        self.speech_lock = threading.Lock()
        self.stop_event = threading.Event()

    def speak(self, text):
        with self.speech_lock:
            logging.debug(f"Starting to speak: {text}")
            try:
                self.stop_speaking()  # Stop any ongoing speech before starting new one
                self.stop_event.clear()
                self.current_process = subprocess.Popen(["espeak", "-p", "35", text], preexec_fn=os.setsid)
                
                # Wait for the process to complete or be interrupted
                while self.current_process.poll() is None:
                    if self.stop_event.wait(timeout=0.1):
                        self.stop_speaking()
                        break
                
                logging.debug(f"Finished speaking: {text}")
            except subprocess.CalledProcessError as e:
                logging.error(f"Failed to speak text: {e}")
            except Exception as e:
                logging.error(f"Unexpected error in speak method: {e}", exc_info=True)
            finally:
                self.current_process = None

    def stop_speaking(self):
        if self.current_process and self.current_process.poll() is None:
            logging.debug("Terminating ongoing speech")
            try:
                os.killpg(os.getpgid(self.current_process.pid), signal.SIGTERM)
                self.current_process.wait(timeout=0.5)
            except subprocess.TimeoutExpired:
                os.killpg(os.getpgid(self.current_process.pid), signal.SIGKILL)
            except Exception as e:
                logging.error(f"Error stopping speech: {e}")
            finally:
                self.current_process = None
                self.stop_event.set()
        else:
            logging.debug("No ongoing speech to terminate")

    def is_speaking(self):
        return self.current_process is not None and self.current_process.poll() is None
