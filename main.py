# --- main.py (Orchestrator) ---

import requests
import logging
import json
import os

# --- Configuration ---
STRUCTURING_SERVICE_URL = "http://127.0.0.1:8001/structure-pdf/"
NLP_SERVICE_URL = "http://127.0.0.1:8002/process/" # New NLP Service URL
PDF_FILE_TO_CHECK = "test_oc.pdf"

# ... (Setup logging, get_structured_data_from_service function remains the same) ...
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def get_structured_data_from_service(file_path: str):
    # This function remains unchanged
    # ...
    logging.info(f"Sending '{file_path}' to the structuring service at {STRUCTURING_SERVICE_URL}")
    try:
        with open(file_path, "rb") as f:
            files = {'file': (os.path.basename(file_path), f, 'application/pdf')}
            response = requests.post(STRUCTURING_SERVICE_URL, files=files, timeout=300)
            response.raise_for_status()
            return response.json()
    except Exception as e:
        logging.error(f"Could not connect to structuring service: {e}")
        return [{"error": "Structuring service unavailable"}]

def process_text_with_nlp_service(text: str):
    """Sends a block of text to the NLP service."""
    logging.info("Sending text to NLP service for processing...")
    try:
        response = requests.post(NLP_SERVICE_URL, json={"text": text}, timeout=120)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        logging.error(f"Could not connect to NLP service: {e}")
        return {"error": "NLP service unavailable"}

# --- Main Execution Logic ---
if __name__ == '__main__':
    # 1. Get structured data from the PDF
    structured_elements = get_structured_data_from_service(PDF_FILE_TO_CHECK)

    if structured_elements and "error" not in structured_elements[0]:
        # 2. Filter and combine relevant text blocks for NLP analysis
        text_to_process = ""
        for element in structured_elements:
            if element.get("type") in ["NarrativeText", "UncategorizedText", "ListItem", "Title"]:
                text_to_process += element.get("text", "") + "\n\n"
        
        if text_to_process.strip():
            # 3. Send the combined text to the new NLP service
            nlp_results = process_text_with_nlp_service(text_to_process)
            
            # --- NEW AND IMPROVED OUTPUT HANDLING ---
            if nlp_results and "error" not in nlp_results:
                print("\n" + "="*20 + " NLP RESULTS " + "="*20)

                # --- Print Entities ---
                print("\n--- 1. Named Entities (NER) ---")
                entities = nlp_results.get("entities", [])
                if entities:
                    # Print each entity on its own line for clarity
                    for ent in entities:
                        print(f"- Text: '{ent['text']}',  Label: {ent['label']}")
                else:
                    print("No named entities were found in the processed text.")

                # --- Print Lemmas (shortened) ---
                print("\n--- 2. Lemmatization (First 20 Tokens) ---")
                lemmas = nlp_results.get("lemmas", [])
                if lemmas:
                    # To avoid flooding the screen, we only show a preview
                    print(json.dumps(lemmas[:20], indent=2, ensure_ascii=False))
                    if len(lemmas) > 20:
                        print(f"  ... and {len(lemmas) - 20} more lemmas.")
                else:
                    print("No lemmas were generated.")

                print("\n" + "="*53)

            else:
                print("--- Error from NLP Service ---")
                print(json.dumps(nlp_results, indent=2, ensure_ascii=False))
        else:
            print("No processable text found in the document after structuring.")
    else:
        print("--- Error from Structuring Service ---")
        print(json.dumps(structured_elements, indent=2, ensure_ascii=False))
