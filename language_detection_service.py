# --- language_detection_service.py ---
# This file contains the complete, self-contained language detection service.
# uvicorn language_detection_service:app --reload --port 8000

# --- Imports ---
import logging
import os
import tempfile
from typing import Dict, Any, Optional

import fitz  # PyMuPDF
from fastapi import FastAPI, UploadFile, File, HTTPException
from langdetect import detect, LangDetectException


# ####################################################################
# ##### Detector Class for PDF Language Detection #####
# ####################################################################

class PdfLanguageDetector:
    """
    A robust class to detect the language of a given PDF file.
    Designed for production use with proper logging and error handling.
    """
    def _extract_text(self, file_path: str) -> Optional[str]:
        """
        Extracts all text from a PDF file.

        Args:
            file_path (str): The path to the PDF file.

        Returns:
            Optional[str]: The extracted text as a single string, or None if an error occurs.
        """
        try:
            with fitz.open(file_path) as doc:
                # Check if the PDF is password-protected
                if doc.is_encrypted:
                    logging.warning(f"File '{file_path}' is encrypted and cannot be opened.")
                    return None

                # Efficiently join text from all pages into one string
                full_text = "".join(page.get_text() for page in doc)
                return full_text

        except Exception as e:
            # Catch any other unexpected errors during file processing
            logging.error(f"An unexpected error occurred while extracting text from '{file_path}': {e}")
            return None

    def detect_language(self, file_path: str) -> Dict[str, Any]:
        """
        Analyzes a PDF file and returns a structured result dictionary.
        """
        logging.info(f"Starting language detection for file: {file_path}")
        
        result = {
            "file_path": file_path,
            "language": None,
            "status": "failed",
            "error_message": None
        }

        try:
            # Step 1: Extract text from the file
            text = self._extract_text(file_path)

            # Step 2: Check if text was found
            if not text or not text.strip():
                raise ValueError("No text content found in the document to analyze.")

            # Step 3: Detect the language
            try:
                language = detect(text)
                result["language"] = language
                result["status"] = "success"
                logging.info(f"Successfully detected language '{language}' for file '{file_path}'.")

            except LangDetectException:
                # This error is thrown by langdetect for ambiguous or short texts
                logging.warning(f"Language could not be reliably detected for '{file_path}'.")
                result["status"] = "completed_with_warning"
                result["error_message"] = "Language detection failed (text too short or ambiguous)."

        except Exception as e:
            # Catch any other errors (e.g., from text extraction or ValueError)
            logging.critical(f"A critical error occurred for '{file_path}': {e}", exc_info=True)
            result["error_message"] = str(e)

        return result


# ##############################################################
# ##### FastAPI Application Setup #####
# ##############################################################

# Initialize the FastAPI app
app = FastAPI(title="Language Detection Service")

# Initialize the detector once at startup to be reused for all requests
detector = PdfLanguageDetector()

# Setup basic logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# ##############################################################
# ##### API Endpoint Definition #####
# ##############################################################

@app.post("/detect-language")
async def detect_language_endpoint(file: UploadFile = File(...)):
    """
    API endpoint that accepts a file upload, saves it to a temporary file,
    detects its language, and then cleans up the temporary file.
    """
    # Use tempfile for a cross-platform, secure way to handle temporary files.
    # 'delete=False' is important on Windows to allow reopening the file.
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as temp_file:
        temp_file_path = temp_file.name
        
    try:
        # Write the uploaded content to the temporary file
        with open(temp_file_path, "wb") as buffer:
            buffer.write(await file.read())

        logging.info(f"Received file '{file.filename}', saved temporarily to '{temp_file_path}'.")

        # Use the robust detector class to process the file
        result = detector.detect_language(temp_file_path)

        # If the detection process returned a failure status, raise a server error
        if result["status"] == "failed":
            raise HTTPException(status_code=500, detail=result)
        
        # Otherwise, return the successful or warning result
        return result

    except Exception as e:
        # Catch any unexpected exceptions in the endpoint logic
        logging.critical(f"A critical error occurred in the endpoint for file '{file.filename}': {e}", exc_info=True)
        raise HTTPException(status_code=500, detail={"error": "An internal server error occurred."})
    
    finally:
        # Crucial cleanup step: ensure the temporary file is always deleted
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
            logging.info(f"Cleaned up temporary file: {temp_file_path}")