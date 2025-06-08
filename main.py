# --- main.py (Orchestrator) ---

import requests
import logging
import json
import os
import sys # Importieren Sie sys

# --- Configuration ---
STRUCTURING_SERVICE_URL = "http://127.0.0.1:8001/structure-pdf/"
NLP_SERVICE_URL = "http://127.0.0.1:8002/process/" # New NLP Service URL
PDF_FILE_TO_CHECK = "test_oc.pdf"
LOG_FILE_PATH = "logging.txt" # Pfad zur Log-Datei

# --- Setup logging ---
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def get_structured_data_from_service(file_path: str):
    """Sends a PDF file to the structuring service and returns the structured data."""
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
    # Öffnen der Log-Datei im Schreibmodus ('w') mit UTF-8 Kodierung
    # existierende Datei wird gelöscht und neu erstellt
    with open(LOG_FILE_PATH, 'w', encoding='utf-8') as log_file:
        # Standard-stdout sichern und auf die Datei umleiten
        original_stdout = sys.stdout
        sys.stdout = log_file

        try:
            print("\n" + "="*10 + " PIPELINE START " + "="*10)

            # 1. Get structured data from the PDF
            print("\n--- STEP 1: PDF Structuring ---")
            structured_elements = get_structured_data_from_service(PDF_FILE_TO_CHECK)

            if structured_elements and "error" not in structured_elements[0]:
                print("\n--- Output of Step 1 (Full Structured Elements) ---")
                # **ÄNDERUNG**: Kein Slicing mehr, gibt alle Elemente aus
                print(json.dumps(structured_elements, indent=2, ensure_ascii=False))
                print(f"Total structured elements: {len(structured_elements)}") # Zusätzliche Info
                print("-" * 40)

                # 2. Filter and combine relevant text blocks for NLP analysis
                print("\n--- STEP 2: Text Extraction and Combination for NLP ---")
                text_to_process = ""
                for element in structured_elements:
                    if element.get("type") in ["NarrativeText", "UncategorizedText", "ListItem", "Title"]:
                        text_to_process += element.get("text", "") + "\n\n"
                
                if text_to_process.strip():
                    print("\n--- Output of Step 2 (Full Combined Text) ---")
                    print(f"Combined text length: {len(text_to_process.strip())} characters")
                    print("Full Text:\n```")
                    # **ÄNDERUNG**: Kein Slicing mehr, gibt den gesamten Text aus
                    print(text_to_process.strip())
                    print("```")
                    print("-" * 40)

                    # 3. Send the combined text to the new NLP service
                    print("\n--- STEP 3: NLP Processing ---")
                    nlp_results = process_text_with_nlp_service(text_to_process)
                    
                    if nlp_results and "error" not in nlp_results:
                        print("\n--- Output of Step 3 (NLP Results) ---")
                        print("\n--- 3.1: Named Entities (NER) ---")
                        entities = nlp_results.get("entities", [])
                        if entities:
                            for ent in entities:
                                print(f"- Text: '{ent.get('text', '')}',  Label: {ent.get('label', '')}")
                        else:
                            print("No named entities were found in the processed text.")

                        print("\n--- 3.2: Lemmatization (Full List of Tokens) ---")
                        lemmas = nlp_results.get("lemmas", [])
                        if lemmas:
                            # **ÄNDERUNG**: Kein Slicing mehr, gibt alle Lemmata aus
                            print(json.dumps(lemmas, indent=2, ensure_ascii=False))
                            print(f"Total lemmas generated: {len(lemmas)}") # Zusätzliche Info
                        else:
                            print("No lemmas were generated.")
                        print("-" * 40)

                    else:
                        print("\n--- Error from NLP Service in Step 3 ---")
                        print(json.dumps(nlp_results, indent=2, ensure_ascii=False))
                        print("-" * 40)
                else:
                    print("\n--- No processable text found in the document after structuring (Step 2 failed) ---")
                    print("-" * 40)
            else:
                print("\n--- Error from Structuring Service in Step 1 ---")
                print(json.dumps(structured_elements, indent=2, ensure_ascii=False))
                print("-" * 40)
            
            print("\n" + "="*10 + " PIPELINE END " + "="*10)

        finally:
            sys.stdout = original_stdout
            logging.info(f"Full pipeline output written to '{LOG_FILE_PATH}'")