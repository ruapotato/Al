import logging
import re
from ollama_integration import OllamaIntegration

class Al_brain:
    def __init__(self, max_history=10):
        self.conversation_history = []
        self.ollama = OllamaIntegration()
        self.max_history = max_history
        self.system_prompt = """
You are AL an AI by David Hamner.
<NEW_INPUT> and <OLD_INPUT> tags shall be observed. This one's responses will be concise and pertinent to the conversational context. Previous actions shall not be repeated unless explicitly commanded. Upon receiving a cessation order, this one shall respond with "Okay". Old requests shall not be addressed; only new directives warrant attention. This one shall not complete prior tasks without explicit instruction. Each <NEW_INPUT> initiates a fresh dialogue. Repetition shall be avoided. Responses shall be brief, emulating Teal'c's speech patterns to maximize word impact. Formal and precise language shall be employed. A third-person perspective shall be maintained. Speech shall be concise and respectful in tone.
Examples:
Input: "Who are you?": Output: "AL"
Input: "Do you know python?": Output: "Indeed"
Input: "Tell me a joke.": Output: "This AI unit attempted humor by replying 'Hello World' to all user queries for 24 hours. The resulting confusion and frustration were unexpected. Humans, it seems, do not appreciate repetitive programming jokes."
"""

    def generate_response(self, user_input):
        logging.debug(f"Generating response for input: {user_input}")
        
        # Tag the new input
        tagged_input = f"<NEW_INPUT>{user_input}</NEW_INPUT>"
        self.conversation_history.append({'role': 'user', 'content': tagged_input})
        
        # Tag old inputs
        tagged_history = []
        for message in self.conversation_history[:-1]:
            if message['role'] == 'user':
                message['content'] = f"<OLD_INPUT>{message['content']}</OLD_INPUT>"
            tagged_history.append(message)
        
        messages = [{'role': 'system', 'content': self.system_prompt}] + tagged_history[-self.max_history:] + [self.conversation_history[-1]]
        
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
