import ollama
import logging

class OllamaIntegration:
    def __init__(self, model_name: str = "llama3.2"):
        self.model_name = model_name

    def generate_response(self, messages, stream: bool = False):
        try:
            response = ollama.chat(model=self.model_name, messages=messages, stream=stream)
            
            if stream:
                return self._stream_response(response)
            else:
                return response['message']['content'] if isinstance(response, dict) else str(response)
        except Exception as e:
            error_msg = f"Error in Ollama API call: {str(e)}"
            logging.error(error_msg)
            return error_msg

    def _stream_response(self, response):
        try:
            for chunk in response:
                if isinstance(chunk, dict) and 'message' in chunk:
                    content = chunk['message'].get('content', '')
                    if content:
                        yield content
                elif isinstance(chunk, str):
                    yield chunk
        except Exception as e:
            logging.error(f"Error in stream processing: {str(e)}")
            yield f"Error: {str(e)}"

    def list_models(self) -> list:
        try:
            models = ollama.list()
            return [model['name'] for model in models['models']]
        except Exception as e:
            logging.error(f"Error listing models: {str(e)}")
            return []

    def change_model(self, new_model_name: str) -> bool:
        if new_model_name in self.list_models():
            self.model_name = new_model_name
            logging.info(f"Model changed to {new_model_name}")
            return True
        else:
            logging.warning(f"Model {new_model_name} not found.")
            return False
