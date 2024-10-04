# Al - Conversational AI Assistant

Al is an advanced conversational AI assistant that uses speech recognition, natural language processing, and text-to-speech capabilities to engage in dynamic, context-aware conversations. This project is based on the [mini-omni project](https://github.com/gpt-omni/mini-omni) and extends its functionality with improved context management and integration with the Ollama API.

## Features

- Speech recognition for user input
- Natural language processing using Ollama API
- Text-to-speech output using espeak
- Context-aware conversations
- Voice Activity Detection (VAD) for improved speech recognition
- Ability to change AI models

## Prerequisites

- Python 3.8 or higher
- espeak (for text-to-speech)
- Ollama (for language model inference)
- mic/headphones (to keep the AI from talking to itself)

## Setup

1. Clone the repository:
   ```
   git clone https://github.com/ruapotato/Al.git
   cd al-assistant
   ```

2. Create a virtual environment:
   ```
   python -m venv pyenv
   ```

3. Activate the virtual environment:
   - On Windows:
     ```
     pyenv\Scripts\activate
     ```
   - On macOS and Linux:
     ```
     source pyenv/bin/activate
     ```

4. Install the required packages:
   ```
   pip install torch numpy pyaudio SpeechRecognition requests onnxruntime ollama
   ```

5. Install espeak on your system:
   - For Ubuntu/Debian: `sudo apt-get install espeak`
   - For macOS with Homebrew: `brew install espeak`
   - For Windows, download and install from the official website

6. Install Ollama following the instructions on their [official website](https://ollama.ai/).


## Usage

1. Ensure Ollama is running and the desired model is available. (Check ollama_integration.py for what model is being used)

2. Run the main script:
   ```
   python main.py
   ```

3. Speak to Al when prompted.

4. Press Ctrl+C to stop the program.

## Acknowledgments

- This project is based on the [mini-omni project](https://github.com/gpt-omni/mini-omni).
- Special thanks to the Ollama team for their API.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Disclaimer

This AI assistant is a simulation and does not have real-world knowledge beyond its training data. It should not be used for critical decision-making or as a substitute for professional advice.
