# Al - Conversational AI Assistant

Al is an advanced conversational AI assistant that uses speech recognition, natural language processing, and text-to-speech capabilities to engage in dynamic, context-aware conversations. This project is based on the [mini-omni project](https://github.com/gpt-omni/mini-omni) and extends its functionality with improved context management and integration with the Ollama API.

## Features

- Speech recognition for user input
- Natural language processing using Ollama API
- Text-to-speech output using espeak
- Context-aware conversations
- Voice Activity Detection (VAD) for improved speech recognition
- Ability to change AI models
- Pause and resume functionality
- Immediate interruption of ongoing responses
- Clipboard content reading
- Screen text extraction

## Prerequisites

- Python 3.8 or higher
- espeak (for text-to-speech)
- Ollama (for language model inference)
- Microphone and speakers/headphones (to prevent feedback loops)

## Setup

1. Clone the repository:
   ```
   git clone https://github.com/ruapotato/Al.git
   cd Al
   ```

2. Create a virtual environment:
   ```
   python -m venv pyenv
   ```

3. Activate the virtual environment:
   - Linux:
     ```
     source pyenv/bin/activate
     ```

4. Install the required packages:
   ```
   pip install torch numpy pyaudio SpeechRecognition requests onnxruntime ollama pyperclip pytesseract pillow
   ```

5. Install espeak and other system dependencies:
   - For Ubuntu/Debian: `sudo apt-get install espeak tesseract-ocr xdotool`

6. Install Ollama following the instructions on their [official website](https://ollama.ai/).

## Usage

1. Ensure Ollama is running and the desired model is available. (Check ollama_integration.py for the default model being used)

2. Run the main script:
   ```
   python main.py
   ```

3. Speak to Al when prompted. You can:
   - Ask questions or give commands
   - Say "pause" to pause Al's listening and response generation
   - Say "Al" or "resume" to resume from a paused state
   - Use "Al [command]" to resume and immediately process a command
   - Say "read my clipboard" to have Al read the contents of your clipboard
   - Say "read my screen" to have Al extract and read text from your active window

4. Press Ctrl+C to stop the program.

## Key Components

- `main.py`: The main script that initializes and runs the Al assistant
- `Al_brain.py`: Handles the conversation logic and integration with Ollama
- `Al_ears.py`: Manages speech recognition and transcription
- `Al_voice.py`: Handles text-to-speech output
- `Al_eyes.py`: Manages clipboard reading and screen text extraction
- `ollama_integration.py`: Integrates with the Ollama API for language model inference

## Customization

- To change the default AI model, modify the `model_name` parameter in the `OllamaIntegration` class initialization in `ollama_integration.py`.
- Adjust the `max_history` parameter in `Al_brain.py` to control the conversation context length.

## Troubleshooting

- If you encounter audio-related errors, ensure your microphone and speakers are properly configured and recognized by your system.
- For Ollama-related issues, check that the Ollama service is running and the desired model is available.
- If clipboard or screen reading fails, ensure you have the necessary permissions and that xclip and xdotool are properly installed.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Acknowledgments

- This project is based on the [mini-omni project](https://github.com/gpt-omni/mini-omni).
- Special thanks to the Ollama team for their API.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Disclaimer

This AI assistant is a simulation and does not have real-world knowledge beyond its training data. It should not be used for critical decision-making or as a substitute for professional advice.
