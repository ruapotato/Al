import requests

class OllamaIntegration:
    def __init__(self, model_name: str = "llama3.2", max_tokens: int = 8192):
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.base_url = "http://localhost:11434/api"

    def generate_response(self, prompt: str) -> str:
        try:
            response = requests.post(
                f"{self.base_url}/generate",
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "num_predict": self.max_tokens
                    }
                }
            )
            
            if response.status_code == 200:
                return response.json()['response'].strip()
            else:
                return f"Error in Ollama API call: {response.status_code} - {response.text}"
        except Exception as e:
            return f"Error in processing: {str(e)}"

    def list_models(self) -> list:
        try:
            response = requests.get(f"{self.base_url}/tags")
            if response.status_code == 200:
                return [model['name'] for model in response.json()['models']]
            else:
                return []
        except Exception as e:
            print(f"Error listing models: {str(e)}")
            return []

    def change_model(self, new_model_name: str) -> bool:
        if new_model_name in self.list_models():
            self.model_name = new_model_name
            return True
        else:
            print(f"Model {new_model_name} not found.")
            return False

    def set_max_tokens(self, max_tokens: int) -> None:
        self.max_tokens = max_tokens
