# --- main.py ---

import requests
import logging
import json # For pretty-printing the result
import os

# --- Configuration ---
# We now have two different services running on different ports
LANGUAGE_SERVICE_URL = "http://127.0.0.1:8000/detect-language"
STRUCTURING_SERVICE_URL = "http://127.0.0.1:8001/structure-pdf/" # New service on a new port
PDF_FILE_TO_CHECK = "test_oc.pdf" # Make sure this file exists

# --- Setup basic logging ---
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')


def get_language_from_service(file_path: str):
    """Sends a file to the language detection service and gets the result."""
    try:
        with open(file_path, "rb") as f:
            files = {'file': (os.path.basename(file_path), f, 'application/pdf')}
            response = requests.post(LANGUAGE_SERVICE_URL, files=files, timeout=60)
            response.raise_for_status()
            return response.json()
    except requests.exceptions.RequestException as e:
        logging.error(f"Could not connect to language service: {e}")
        return {"status": "error", "error_message": "Service unavailable"}
    except FileNotFoundError:
        logging.error(f"Local file not found: {file_path}")
        return {"status": "error", "error_message": "Local file not found"}


def get_structured_data_from_service(file_path: str):
    """
    Sends a file to the new structuring service and gets the result.
    """
    logging.info(f"Sending '{file_path}' to the structuring service at {STRUCTURING_SERVICE_URL}")
    try:
        with open(file_path, "rb") as f:
            # The 'file' key must match the parameter name in the FastAPI endpoint
            files = {'file': (os.path.basename(file_path), f, 'application/pdf')}
            # This request might take longer for complex documents
            response = requests.post(STRUCTURING_SERVICE_URL, files=files, timeout=300) # Increased timeout
            response.raise_for_status()
            return response.json() # The result is a list of dictionaries
    except requests.exceptions.RequestException as e:
        logging.error(f"Could not connect to structuring service: {e}")
        return [{"error": "Structuring service unavailable"}]
    except FileNotFoundError:
        logging.error(f"Local file not found: {file_path}")
        return [{"error": "Local file not found"}]


# --- Usage ---
if __name__ == '__main__':
    # Call the new structuring service
    structured_result = get_structured_data_from_service(PDF_FILE_TO_CHECK)
    print("--- Structuring Service Response ---")
    # Pretty-print the JSON response to make it readable
    print(json.dumps(structured_result, indent=2, ensure_ascii=False))