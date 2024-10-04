import torch
import numpy as np
import pyaudio
import speech_recognition as sr
import time
import os
import logging
import queue
import threading
import bisect
import functools
import warnings
from typing import List, NamedTuple, Optional
import subprocess
from ollama_integration import OllamaIntegration

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Check for onnxruntime at the beginning
try:
    import onnxruntime
except ImportError:
    print("The onnxruntime package is required for VAD functionality.")
    print("Please install it using: pip install onnxruntime")
    print("The script will continue without VAD. Some features may not work as expected.")
    onnxruntime = None

# Constants
RATE = 16000
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
VAD_STRIDE = 0.5

class VadOptions(NamedTuple):
    threshold: float = 0.5
    min_speech_duration_ms: int = 250
    max_speech_duration_s: float = float("inf")
    min_silence_duration_ms: int = 2000
    window_size_samples: int = 1024
    speech_pad_ms: int = 400

def get_speech_timestamps(
    audio: np.ndarray,
    vad_options: Optional[VadOptions] = None,
    **kwargs,
) -> List[dict]:
    if vad_options is None:
        vad_options = VadOptions(**kwargs)

    threshold = vad_options.threshold
    min_speech_duration_ms = vad_options.min_speech_duration_ms
    max_speech_duration_s = vad_options.max_speech_duration_s
    min_silence_duration_ms = vad_options.min_silence_duration_ms
    window_size_samples = vad_options.window_size_samples
    speech_pad_ms = vad_options.speech_pad_ms

    if window_size_samples not in [512, 1024, 1536]:
        warnings.warn(
            "Unusual window_size_samples! Supported window_size_samples:\n"
            " - [512, 1024, 1536] for 16000 sampling_rate"
        )

    sampling_rate = 16000
    min_speech_samples = sampling_rate * min_speech_duration_ms / 1000
    speech_pad_samples = sampling_rate * speech_pad_ms / 1000
    max_speech_samples = (
        sampling_rate * max_speech_duration_s
        - window_size_samples
        - 2 * speech_pad_samples
    )
    min_silence_samples = sampling_rate * min_silence_duration_ms / 1000
    min_silence_samples_at_max_speech = sampling_rate * 98 / 1000

    audio_length_samples = len(audio)

    model = get_vad_model()
    state = model.get_initial_state(batch_size=1)

    speech_probs = []
    for current_start_sample in range(0, audio_length_samples, window_size_samples):
        chunk = audio[current_start_sample : current_start_sample + window_size_samples]
        if len(chunk) < window_size_samples:
            chunk = np.pad(chunk, (0, int(window_size_samples - len(chunk))))
        speech_prob, state = model(chunk, state, sampling_rate)
        speech_probs.append(speech_prob)

    triggered = False
    speeches = []
    current_speech = {}
    neg_threshold = threshold - 0.15

    temp_end = 0
    prev_end = next_start = 0

    for i, speech_prob in enumerate(speech_probs):
        if (speech_prob >= threshold) and temp_end:
            temp_end = 0
            if next_start < prev_end:
                next_start = window_size_samples * i

        if (speech_prob >= threshold) and not triggered:
            triggered = True
            current_speech["start"] = window_size_samples * i
            continue

        if (
            triggered
            and (window_size_samples * i) - current_speech["start"] > max_speech_samples
        ):
            if prev_end:
                current_speech["end"] = prev_end
                speeches.append(current_speech)
                current_speech = {}
                if next_start < prev_end:
                    triggered = False
                else:
                    current_speech["start"] = next_start
                prev_end = next_start = temp_end = 0
            else:
                current_speech["end"] = window_size_samples * i
                speeches.append(current_speech)
                current_speech = {}
                prev_end = next_start = temp_end = 0
                triggered = False
                continue

        if (speech_prob < neg_threshold) and triggered:
            if not temp_end:
                temp_end = window_size_samples * i
            if (window_size_samples * i) - temp_end > min_silence_samples_at_max_speech:
                prev_end = temp_end
            if (window_size_samples * i) - temp_end < min_silence_samples:
                continue
            else:
                current_speech["end"] = temp_end
                if (
                    current_speech["end"] - current_speech["start"]
                ) > min_speech_samples:
                    speeches.append(current_speech)
                current_speech = {}
                prev_end = next_start = temp_end = 0
                triggered = False
                continue

    if (
        current_speech
        and (audio_length_samples - current_speech["start"]) > min_speech_samples
    ):
        current_speech["end"] = audio_length_samples
        speeches.append(current_speech)

    for i, speech in enumerate(speeches):
        if i == 0:
            speech["start"] = int(max(0, speech["start"] - speech_pad_samples))
        if i != len(speeches) - 1:
            silence_duration = speeches[i + 1]["start"] - speech["end"]
            if silence_duration < 2 * speech_pad_samples:
                speech["end"] += int(silence_duration // 2)
                speeches[i + 1]["start"] = int(
                    max(0, speeches[i + 1]["start"] - silence_duration // 2)
                )
            else:
                speech["end"] = int(
                    min(audio_length_samples, speech["end"] + speech_pad_samples)
                )
                speeches[i + 1]["start"] = int(
                    max(0, speeches[i + 1]["start"] - speech_pad_samples)
                )
        else:
            speech["end"] = int(
                min(audio_length_samples, speech["end"] + speech_pad_samples)
            )

    return speeches

def collect_chunks(audio: np.ndarray, chunks: List[dict]) -> np.ndarray:
    if not chunks:
        return np.array([], dtype=np.float32)

    return np.concatenate([audio[chunk["start"] : chunk["end"]] for chunk in chunks])

@functools.lru_cache
def get_vad_model():
    asset_dir = os.path.join(os.path.dirname(__file__), "assets")
    path = os.path.join(asset_dir, "silero_vad.onnx")
    return SileroVADModel(path)

class SileroVADModel:
    def __init__(self, path):
        try:
            import onnxruntime
        except ImportError as e:
            raise RuntimeError(
                "Applying the VAD filter requires the onnxruntime package"
            ) from e

        opts = onnxruntime.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 1
        opts.log_severity_level = 4

        self.session = onnxruntime.InferenceSession(
            path,
            providers=["CPUExecutionProvider"],
            sess_options=opts,
        )

    def get_initial_state(self, batch_size: int):
        h = np.zeros((2, batch_size, 64), dtype=np.float32)
        c = np.zeros((2, batch_size, 64), dtype=np.float32)
        return h, c

    def __call__(self, x, state, sr: int):
        if len(x.shape) == 1:
            x = np.expand_dims(x, 0)
        if len(x.shape) > 2:
            raise ValueError(
                f"Too many dimensions for input audio chunk {len(x.shape)}"
            )
        if sr / x.shape[1] > 31.25:
            raise ValueError("Input audio chunk is too short")

        h, c = state

        ort_inputs = {
            "input": x,
            "h": h,
            "c": c,
            "sr": np.array(sr, dtype="int64"),
        }

        out, h, c = self.session.run(None, ort_inputs)
        state = (h, c)

        return out, state

class Al_ears:
    def __init__(self, transcription_queue):
        logging.debug("Initializing Al_ears")
        self.transcription_queue = transcription_queue
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
        self.recognizer = sr.Recognizer()
        logging.debug("Al_ears initialized")

    def record_audio_with_vad(self):
        logging.debug("Starting audio recording with VAD")
        print("Listening...")
        frames = []
        audio = np.array([], dtype=np.float32)
        
        while True:
            data = self.stream.read(CHUNK)
            frames.append(data)
            
            # Convert data to float32 for VAD
            audio_chunk = np.frombuffer(data, dtype=np.int16)
            audio_chunk = audio_chunk.astype(np.float32) / 32768.0
            
            audio = np.concatenate([audio, audio_chunk])
            
            if len(audio) >= RATE * VAD_STRIDE:
                speech_timestamps = get_speech_timestamps(audio, VadOptions())
                
                if speech_timestamps:
                    print("Speech detected. Recording...")
                    break
                
                audio = audio[int(RATE * VAD_STRIDE):]
        
        # Continue recording until silence is detected
        silence_counter = 0
        while silence_counter < 30:  # Adjust this value to change sensitivity
            data = self.stream.read(CHUNK)
            frames.append(data)
            
            audio_chunk = np.frombuffer(data, dtype=np.int16)
            audio_chunk = audio_chunk.astype(np.float32) / 32768.0
            audio = np.concatenate([audio, audio_chunk])
            
            speech_timestamps = get_speech_timestamps(audio[-RATE:], VadOptions())
            
            if not speech_timestamps:
                silence_counter += 1
            else:
                silence_counter = 0
        
        print("Speech ended.")
        return b''.join(frames)

    def speech_to_text(self, audio_data):
        audio_file = sr.AudioData(audio_data, RATE, 2)  # 2 bytes per sample for int16
        try:
            text = self.recognizer.recognize_google(audio_file)
            return text
        except sr.UnknownValueError:
            return "Speech recognition could not understand audio"
        except sr.RequestError as e:
            return f"Could not request results from speech recognition service; {e}"

    def listen(self):
        while True:
            audio_data = self.record_audio_with_vad()
            transcription = self.speech_to_text(audio_data)
            logging.info(f"Transcription: {transcription}")
            self.transcription_queue.put(transcription)

    def stop(self):
        self.stream.stop_stream()
        self.stream.close()
        self.p.terminate()

class Al_voice:
    def __init__(self):
        # Check if espeak is installed
        try:
            subprocess.run(["espeak", "--version"], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except subprocess.CalledProcessError:
            logging.error("espeak is not installed. Please install it using your package manager.")
            raise SystemExit("espeak is required for text-to-speech functionality.")
        self.current_process = None

    def stop_speaking(self):
        if self.current_process and self.current_process.poll() is None:
            logging.debug("Terminating ongoing speech")
            self.current_process.terminate()
            self.current_process.wait()
            self.current_process = None

    def speak(self, text):
        logging.debug(f"Speaking: {text}")
        self.stop_speaking()  # Stop any ongoing speech
        try:
            self.current_process = subprocess.Popen(["espeak", text])
        except subprocess.CalledProcessError as e:
            logging.error(f"Failed to speak text: {e}")


class Al_ears:
    def __init__(self, transcription_queue):
        logging.debug("Initializing Al_ears")
        self.transcription_queue = transcription_queue
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
        self.recognizer = sr.Recognizer()
        self.vad_model = get_vad_model() if onnxruntime else None
        logging.debug("Al_ears initialized")

    def record_audio_with_vad(self):
        logging.debug("Starting audio recording with VAD")
        print("Listening...")
        frames = []
        audio = np.array([], dtype=np.float32)
        
        while True:
            data = self.stream.read(CHUNK)
            frames.append(data)
            
            audio_chunk = np.frombuffer(data, dtype=np.int16)
            audio_chunk = audio_chunk.astype(np.float32) / 32768.0
            
            audio = np.concatenate([audio, audio_chunk])
            
            if len(audio) >= RATE * VAD_STRIDE:
                if self.vad_model:
                    speech_timestamps = get_speech_timestamps(audio, VadOptions())
                    if speech_timestamps:
                        print("Speech detected. Recording...")
                        break
                else:
                    # Simple energy-based detection if VAD is not available
                    if np.abs(audio).mean() > 0.02:
                        print("Speech detected. Recording...")
                        break
                
                audio = audio[int(RATE * VAD_STRIDE):]
        
        # Continue recording until silence is detected
        silence_counter = 0
        while silence_counter < 30:  # Adjust this value to change sensitivity
            data = self.stream.read(CHUNK)
            frames.append(data)
            
            audio_chunk = np.frombuffer(data, dtype=np.int16)
            audio_chunk = audio_chunk.astype(np.float32) / 32768.0
            audio = np.concatenate([audio, audio_chunk])
            
            if self.vad_model:
                speech_timestamps = get_speech_timestamps(audio[-RATE:], VadOptions())
                if not speech_timestamps:
                    silence_counter += 1
                else:
                    silence_counter = 0
            else:
                # Simple energy-based detection
                if np.abs(audio[-RATE:]).mean() < 0.02:
                    silence_counter += 1
                else:
                    silence_counter = 0
        
        print("Speech ended.")
        return b''.join(frames)

    def speech_to_text(self, audio_data):
        audio_file = sr.AudioData(audio_data, RATE, 2)  # 2 bytes per sample for int16
        try:
            text = self.recognizer.recognize_google(audio_file)
            return text
        except sr.UnknownValueError:
            return "Speech recognition could not understand audio"
        except sr.RequestError as e:
            return f"Could not request results from speech recognition service; {e}"

    def listen(self):
        while True:
            audio_data = self.record_audio_with_vad()
            transcription = self.speech_to_text(audio_data)
            logging.info(f"Transcription: {transcription}")
            self.transcription_queue.put(transcription)

    def stop(self):
        if self.stream.is_active():
            self.stream.stop_stream()
        self.stream.close()
        self.p.terminate()


class Al_brain:
    def __init__(self, max_history=5):
        self.conversation_history = []
        self.ollama = OllamaIntegration()
        self.max_history = max_history
        self.system_prompt = """
You are an AI assistant named Al. Built by David Hamner and based on many Open Source projects. Your knowledge cutoff is December 2023, and you don't have real-time updates.
Keep your responses concise and relevant to the conversation context.
Always maintain consistency in your self-awareness and capabilities.
Personalize your responses based on the user's previous questions and the conversation flow.
"""

    def generate_response(self, user_input):
        logging.debug(f"Generating response for input: {user_input}")
        
        # Add user input to conversation history
        self.conversation_history.append(f"Human: {user_input}")
        
        # Prepare the prompt with context
        context = "\n".join(self.conversation_history[-self.max_history:])
        prompt = f"""
{self.system_prompt}

Context of the conversation:
{context}

Based on this context, provide a concise and relevant response to the last human input.
AI:"""
        
        response = self.ollama.generate_response(prompt)
        if response.startswith("Error"):
            logging.warning(f"Failed to generate response: {response}")
            return "I apologize, but I'm having trouble generating a response right now."
        else:
            logging.debug(f"Generated response: {response}")
            # Add AI response to conversation history
            self.conversation_history.append(f"AI: {response}")
            # Trim conversation history if it exceeds max_history
            if len(self.conversation_history) > self.max_history * 2:
                self.conversation_history = self.conversation_history[-self.max_history * 2:]
            return response

class Al:
    def __init__(self):
        self.transcription_queue = queue.Queue()
        self.response_queue = queue.Queue()
        self.ears = Al_ears(self.transcription_queue)
        self.brain = Al_brain()
        self.voice = Al_voice()
        self.is_running = False

    def run(self):
        self.is_running = True
        logging.info("Al is listening... (Press Ctrl+C to stop)")
        
        conversation_thread = threading.Thread(target=self.conversation_control)
        conversation_thread.start()
        
        listening_thread = threading.Thread(target=self.ears.listen)
        listening_thread.start()
        
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
        self.voice.stop_speaking()  # Ensure any ongoing speech is stopped
        logging.info("Al stopped.")

    def conversation_control(self):
        logging.debug("Conversation control thread started")
        while self.is_running:
            try:
                transcription = self.transcription_queue.get(timeout=0.1)
                logging.info(f"\nUser: {transcription}")
                
                # Stop any ongoing speech when new input is received
                self.voice.stop_speaking()
                
                if transcription.lower().startswith("change model to "):
                    model_name = transcription.lower().replace("change model to ", "").strip()
                    response = self.brain.change_model(model_name)
                elif transcription.lower() == "list models":
                    response = self.brain.list_models()
                else:
                    response = self.brain.generate_response(transcription)
                
                logging.info(f"Al: {response}")
                
                self.voice.speak(response)
                print("\n")  # New line after response
            except queue.Empty:
                pass
        logging.debug("Conversation control thread ended")

if __name__ == "__main__":
    al = Al()
    try:
        al.run()
    except KeyboardInterrupt:
        logging.info("Keyboard interrupt received. Stopping Al.")
        al.stop()
