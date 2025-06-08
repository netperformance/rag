# --- structuring_service.py ---
# This file contains the complete, self-contained document structuring service.
# FINAL VERSION WITH HARDCODED PATHS TO BYPASS ENVIRONMENT ISSUES.

# --- Imports ---
import logging
import os
import tempfile
from typing import List, Dict

# import fitz # fitz is used by unstructured internally, no need to import here unless used directly
from fastapi import FastAPI, UploadFile, File, HTTPException
from unstructured.partition.pdf import partition_pdf
import pytesseract # Import pytesseract to set its path

# ####################################################################################
# ##### DEFINITIVE SOLUTION: Set all Tesseract paths directly in the code #####
# ####################################################################################

# 1. Define the correct installation path that you found
TESSERACT_INSTALL_PATH = r'C:\Users\info\AppData\Local\Programs\Tesseract-OCR'

# 2. Set the command path for pytesseract
pytesseract.pytesseract.tesseract_cmd = os.path.join(TESSERACT_INSTALL_PATH, 'tesseract.exe')

# 3. Set the TESSDATA_PREFIX environment variable for the current process
# This tells Tesseract where to find its language data files (.traineddata)
os.environ['TESSDATA_PREFIX'] = os.path.join(TESSERACT_INSTALL_PATH, 'tessdata')

logging.info(f"Forcing Tesseract command to: {pytesseract.pytesseract.tesseract_cmd}")
logging.info(f"Forcing TESSDATA_PREFIX to: {os.environ['TESSDATA_PREFIX']}")

# ####################################################################################


# ##### FastAPI Application Setup #####
app = FastAPI(title="Structuring Service")

# Setup basic logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# ##### API Endpoint Definition #####
@app.post("/structure-pdf/")
async def structure_pdf_endpoint(file: UploadFile = File(...)) -> List[Dict]:
    """
    API endpoint that accepts a PDF file, processes it with unstructured.io,
    and returns a list of structured element dictionaries.
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file_path = temp_file.name

    try:
        with open(temp_file_path, "wb") as buffer:
            buffer.write(await file.read())

        logging.info(f"Received file '{file.filename}', saved to '{temp_file_path}' for structuring.")

        elements = partition_pdf(
            filename=temp_file_path,
            strategy="hi_res",
            infer_table_structure=True
        )

        result = [el.to_dict() for el in elements]
        logging.info(f"Successfully structured document into {len(result)} elements.")
        
        return result

    except Exception as e:
        logging.critical(f"A critical error occurred while structuring file '{file.filename}': {e}", exc_info=True)
        raise HTTPException(status_code=500, detail={"error": "An internal server error occurred during structuring."})

    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
            logging.info(f"Cleaned up temporary file: {temp_file_path}")