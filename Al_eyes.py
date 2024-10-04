import logging
import subprocess
import pytesseract
from PIL import Image
import pyperclip
import time

class Al_eyes:
    @staticmethod
    def get_clipboard_content():
        max_attempts = 3
        for attempt in range(max_attempts):
            try:
                content = pyperclip.paste()
                tagged_content = f"<clipboard_content>{content}</clipboard_content>"
                logging.info(f"Clipboard content retrieved and tagged: {tagged_content[:100]}...")  # Log first 100 chars
                return tagged_content
            except Exception as e:
                logging.error(f"Error getting clipboard content (attempt {attempt + 1}/{max_attempts}): {e}")
                time.sleep(0.5)  # Wait for 0.5 seconds before retrying
        
        return "<clipboard_content>Error: Unable to read clipboard after multiple attempts</clipboard_content>"

    @staticmethod
    def get_screen_text():
        try:
            # Capture the active window
            active_window = subprocess.check_output(["xdotool", "getactivewindow"]).decode().strip()
            subprocess.run(["import", "-window", active_window, "temp_screenshot.png"])
            
            # Use pytesseract to extract text from the image
            image = Image.open("temp_screenshot.png")
            text = pytesseract.image_to_string(image)
            
            # Clean up the temporary file
            subprocess.run(["rm", "temp_screenshot.png"])
            
            tagged_text = f"<screen_content>{text.strip()}</screen_content>"
            logging.info(f"Screen text extracted and tagged: {tagged_text[:100]}...")  # Log first 100 chars
            return tagged_text
        except Exception as e:
            logging.error(f"Error getting screen text: {e}")
            return "<screen_content>Error: Unable to read screen text</screen_content>"

    @staticmethod
    def process_keywords(text):
        if "clipboard" in text.lower():
            clipboard_content = Al_eyes.get_clipboard_content()
            text = text.replace("clipboard", clipboard_content)
        if "screen" in text.lower():
            screen_text = Al_eyes.get_screen_text()
            text = text.replace("screen", screen_text)
        logging.info(f"Processed text: {text[:200]}...")  # Log first 200 chars
        return text
