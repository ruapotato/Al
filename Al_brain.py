import logging
import re
from ollama_integration import OllamaIntegration

class Al_brain:
    def __init__(self, max_history=10):
        self.conversation_history = []
        self.ollama = OllamaIntegration()
        self.max_history = max_history
        self.system_prompt = """
You are an AI assistant named Al. Built by David Hamner and based on many Open Source projects. Your knowledge cutoff is December 2023, and you don't have real-time updates.
Keep your responses concise and relevant to the conversation context.
Always maintain consistency in your self-awareness and capabilities.
Personalize your responses based on the user's previous questions and the conversation flow.
Do not repeat previous actions unless explicitly asked to do so.
If a new request is made, focus on that request and do not continue with previous tasks.
Include numbers and calculations when they are relevant to the response.
When you are asked to stop, reply with 'Okay'.
"""

    def generate_response(self, user_input):
        logging.debug(f"Generating response for input: {user_input}")
        
        # Clear previous conversation if the new input is significantly different
        if self.conversation_history and not self._is_related(user_input, self.conversation_history[-1]['content']):
            self.clear_history()
        
        self.conversation_history.append({'role': 'user', 'content': user_input})
        
        messages = [{'role': 'system', 'content': self.system_prompt}] + self.conversation_history[-self.max_history:]
        
        response_stream = self.ollama.generate_response(messages, stream=True)
        return self._process_stream(response_stream)

    def _process_stream(self, response_stream):
        full_response = ""
        for chunk in response_stream:
            logging.debug(f"Received chunk: {chunk}")
            if chunk.startswith("Error:"):
                logging.error(f"Error in stream: {chunk}")
                yield "I encountered an error. Please try again."
                return
            
            full_response += chunk
            yield chunk
        
        if full_response:
            self.conversation_history.append({'role': 'assistant', 'content': full_response})

    def _is_related(self, new_input, previous_input):
        # Simple check for relatedness, can be improved with more sophisticated NLP techniques
        common_words = set(new_input.lower().split()) & set(previous_input.lower().split())
        return len(common_words) > 2  # Consider related if more than two words in common

    def change_model(self, model_name):
        if self.ollama.change_model(model_name):
            return f"Model changed to {model_name}"
        else:
            return f"Failed to change model to {model_name}"

    def list_models(self):
        models = self.ollama.list_models()
        return "Available models: " + ", ".join(models)

    def clear_history(self):
        self.conversation_history = []
        logging.debug("Conversation history cleared.")
