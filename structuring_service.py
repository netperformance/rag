# --- structuring_service.py ---
# This service accepts a PDF and returns its content as a structured list of elements.

# --- Imports ---
import logging
import os
import tempfile
from typing import List, Dict

from fastapi import FastAPI, UploadFile, File, HTTPException
from unstructured.partition.pdf import partition_pdf

# ##############################################################
# ##### FastAPI Application Setup #####
# ##############################################################

# Initialize the FastAPI app
app = FastAPI(title="Structuring Service")

# Setup basic logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# ##############################################################
# ##### API Endpoint Definition #####
# ##############################################################

@app.post("/structure-pdf/")
async def structure_pdf_endpoint(file: UploadFile = File(...)) -> List[Dict]:
    """
    API endpoint that accepts a PDF file, processes it with unstructured.io,
    and returns a list of structured element dictionaries.
    """
    # Use tempfile for a cross-platform, secure way to handle temporary files.
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file_path = temp_file.name

    try:
        # Save the uploaded content to the temporary file
        with open(temp_file_path, "wb") as buffer:
            buffer.write(await file.read())

        logging.info(f"Received file '{file.filename}', saved to '{temp_file_path}' for structuring.")

        # Use unstructured.io to partition the PDF.
        # This function reads the PDF and splits it into semantic elements
        # like Title, NarrativeText, ListItem, Table, etc.
        # The 'strategy="hi_res"' is recommended for complex documents like tenders.
        elements = partition_pdf(
            filename=temp_file_path,
            strategy="hi_res", # Use "hi_res" for complex docs, "fast" for quicker processing
            infer_table_structure=True # Tries to extract table structure into HTML
        )

        # Convert the list of Element objects to a JSON-serializable list of dictionaries
        result = [el.to_dict() for el in elements]
        logging.info(f"Successfully structured document into {len(result)} elements.")
        
        return result

    except Exception as e:
        # Catch any unexpected exceptions from the partitioning logic
        logging.critical(f"A critical error occurred while structuring file '{file.filename}': {e}", exc_info=True)
        raise HTTPException(status_code=500, detail={"error": "An internal server error occurred during structuring."})

    finally:
        # Crucial cleanup step: ensure the temporary file is always deleted
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
            logging.info(f"Cleaned up temporary file: {temp_file_path}")