# --- main.py ---

import requests
import logging

# Configuration
LANGUAGE_SERVICE_URL = "http://127.0.0.1:8000/detect-language" # URL where the service is running
PDF_FILE_TO_CHECK = "test_oc.pdf"

def get_language_from_service(file_path: str):
    """
    Sends a file to the language detection service and gets the result.
    """
    try:
        with open(file_path, "rb") as f:
            files = {'file': (file_path, f, 'application/pdf')}
            response = requests.post(LANGUAGE_SERVICE_URL, files=files, timeout=60) # Set a timeout!

            # Check if the request was successful
            response.raise_for_status() # Raises an HTTPError for bad responses (4xx or 5xx)

            return response.json()

    except requests.exceptions.RequestException as e:
        logging.error(f"Could not connect to language service: {e}")
        return {"status": "error", "error_message": "Service unavailable"}
    except FileNotFoundError:
        logging.error(f"Local file not found: {file_path}")
        return {"status": "error", "error_message": "Local file not found"}

# --- Usage ---
result = get_language_from_service(PDF_FILE_TO_CHECK)
print(f"Service response: {result}")