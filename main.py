# --- main.py (Orchestrator) ---

import requests
import logging
import json
import os
import sys
from langdetect import detect, LangDetectException
from typing import Optional # Dies ist wichtig und sollte am Anfang der Datei sein

# Helper function to deep merge dictionaries (required for nested configs)
def deep_update(base_dict, update_dict):
    for key, value in update_dict.items():
        if isinstance(value, dict) and key in base_dict and isinstance(base_dict[key], dict):
            deep_update(base_dict[key], value)
        else:
            base_dict[key] = value

# --- Configuration Loading ---
CONFIG_FILE = "config.json"
DEFAULT_CONFIG = {
    "nlp_model_config": {
        "ner_model_name_de": "domischwimmbeck/bert-base-german-cased-fine-tuned-ner",
        "lemmatization_model_name_de": "de_core_news_sm",
        "ner_model_name_en": "dslim/bert-base-NER",
        "lemmatization_model_name_en": "en_core_web_sm",
        "lemmatization_enabled": True
    },
    "service_urls": {
        "structuring_service": "http://127.0.0.1:8001/structure-pdf/",
        "nlp_service": "http://127.0.0.1:8002/process/",
        "language_detection_service": "http://127.0.0.1:8000/detect-language",
        "deepseek_enrichment_service": "http://127.0.0.1:8003/enrich-text/" # Neue URL
    },
    "orchestrator_config": {
        "pdf_file_to_check": "test_oc.pdf",
        "log_file_path": "logging.txt"
    },
    "logging_config": { # Standard-Logging-Konfiguration
        "enabled": True, # Korrigiert: 'true' zu 'True'
        "level": "INFO"
    },
    "ollama_config": { # Hinzugefügt, da config von hier gelesen wird
        "ollama_base_url": "http://localhost:11434",
        "deepseek_model_name": "deepseek-coder-v2:latest"
    }
}

config = DEFAULT_CONFIG

try:
    with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
        loaded_config = json.load(f)
        deep_update(config, loaded_config)
    logging.info(f"Konfiguration aus '{CONFIG_FILE}' erfolgreich geladen.")
except FileNotFoundError:
    logging.warning(f"Konfigurationsdatei '{CONFIG_FILE}' nicht gefunden. Verwende Standardwerte.")
except json.JSONDecodeError:
    logging.error(f"Fehler beim Parsen der Konfigurationsdatei '{CONFIG_FILE}'. Überprüfen Sie das JSON-Format. Verwende Standardwerte.")
except Exception as e:
    logging.error(f"Unerwarteter Fehler beim Laden der Konfiguration: {e}. Verwende Standardwerte.")

# Konfigurationswerte extrahieren
STRUCTURING_SERVICE_URL = config["service_urls"]["structuring_service"]
NLP_SERVICE_URL = config["service_urls"]["nlp_service"]
DEEPSEEK_ENRICHMENT_SERVICE_URL = config["service_urls"]["deepseek_enrichment_service"] # NEU
PDF_FILE_TO_CHECK = config["orchestrator_config"]["pdf_file_to_check"]
LOG_FILE_PATH = config["orchestrator_config"]["log_file_path"]

# Logging-Konfiguration extrahieren
LOGGING_ENABLED = config["logging_config"]["enabled"]
LOGGING_LEVEL_STR = config["logging_config"]["level"].upper() # Stellen Sie sicher, dass es Großbuchstaben sind

# Das Logging-Level-Mapping
LOGGING_LEVELS = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL
}

# Bestimmen Sie das tatsächliche Logging-Level
if LOGGING_ENABLED:
    log_level = LOGGING_LEVELS.get(LOGGING_LEVEL_STR, logging.INFO) # Standardmäßig INFO, falls ungültig
else:
    log_level = logging.CRITICAL # Setzt das Level auf CRITICAL, um andere Log-Nachrichten zu unterdrücken

# --- Setup Logging ---
# Löschen Sie alle vorhandenen Handler, um doppelte Ausgaben zu vermeiden, falls logging.basicConfig mehrfach aufgerufen wird
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)
logging.basicConfig(level=log_level, format='%(levelname)s: %(message)s')


def get_structured_data_from_service(file_path: str):
    """Sends a PDF file to the structuring service and returns the structured data."""
    logging.info(f"Sende '{file_path}' an den Strukturierungsdienst unter {STRUCTURING_SERVICE_URL}")
    try:
        with open(file_path, "rb") as f:
            files = {'file': (os.path.basename(file_path), f, 'application/pdf')}
            response = requests.post(STRUCTURING_SERVICE_URL, files=files, timeout=300)
            response.raise_for_status()
            return response.json()
    except Exception as e:
        logging.error(f"Verbindung zum Strukturierungsdienst fehlgeschlagen: {e}")
        return [{"error": "Structuring service unavailable"}]

def detect_text_language(text: str) -> str:
    """Erkennt die Sprache des gegebenen Textes."""
    try:
        language = detect(text)
        logging.info(f"Sprache des Textes erkannt: {language}")
        return language
    except LangDetectException:
        logging.warning("Sprache konnte nicht zuverlässig erkannt werden, standardmäßig auf Deutsch gesetzt.")
        return "de" # Standard auf Deutsch
    except Exception as e:
        logging.error(f"Unerwarteter Fehler bei der Spracherkennung: {e}")
        return "de" # Standard auf Deutsch im Fehlerfall

def process_text_with_nlp_service(text: str, language: Optional[str] = None):
    """Sendet einen Textblock an den NLP-Dienst zur Verarbeitung mit optionaler Sprachangabe."""
    logging.info(f"Sende Text an NLP-Dienst zur Verarbeitung (Sprache: {language if language else 'Autoerkennung'})...")
    try:
        payload = {"text": text}
        if language:
            payload["language"] = language
        
        response = requests.post(NLP_SERVICE_URL, json=payload, timeout=120)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        logging.error(f"Verbindung zum NLP-Dienst fehlgeschlagen: {e}")
        return {"error": "NLP service unavailable"}

# NEUE FUNKTION: DeepSeek Anreicherungsdienst aufrufen
def call_deepseek_enrichment_service(text: str, task: str = "chunk_and_summarize"):
    """Ruft den DeepSeek Anreicherungsdienst auf."""
    logging.info(f"Sende Text zur DeepSeek-Anreicherung (Task: {task})...")
    payload = {"text": text, "task": task}
    try:
        response = requests.post(DEEPSEEK_ENRICHMENT_SERVICE_URL, json=payload, timeout=600) # Längeres Timeout
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        logging.error(f"Fehler bei der Verbindung zum DeepSeek Anreicherungsdienst: {e}")
        return {"error": "DeepSeek enrichment service unavailable"}


# --- Main Execution Logic ---
if __name__ == '__main__':
    with open(LOG_FILE_PATH, 'w', encoding='utf-8') as log_file:
        original_stdout = sys.stdout
        sys.stdout = log_file

        try:
            print("\n" + "="*10 + " PIPELINE START " + "="*10)

            print("\n--- SCHRITT 1: PDF Strukturierung ---")
            structured_elements = get_structured_data_from_service(PDF_FILE_TO_CHECK)

            if structured_elements and not ("error" in structured_elements[0] if structured_elements else False): # Robusterer Check
                print("\n--- Ausgabe von SCHRITT 1 (Vollständige strukturierte Elemente) ---")
                print(json.dumps(structured_elements, indent=2, ensure_ascii=False))
                print(f"Gesamtzahl strukturierter Elemente: {len(structured_elements)}")
                print("-" * 40)

                print("\n--- SCHRITT 2: Textextraktion und Kombination für NLP-Analyse ---")
                text_to_process = ""
                for element in structured_elements:
                    if element.get("type") in ["NarrativeText", "UncategorizedText", "ListItem", "Title"]:
                        text_to_process += element.get("text", "") + "\n\n"
                
                if text_to_process.strip():
                    detected_language = detect_text_language(text_to_process)
                    print(f"Erkannte Sprache für NLP-Verarbeitung: {detected_language}")

                    print("\n--- SCHRITT 3: Basis-NLP-Verarbeitung (spaCy/HuggingFace NER) ---")
                    nlp_results = process_text_with_nlp_service(text_to_process, detected_language)
                    
                    if nlp_results and "error" not in nlp_results:
                        print("\n--- Ausgabe von SCHRITT 3 (NLP-Ergebnisse) ---")
                        print(f"Verarbeitete Sprache vom NLP-Dienst: {nlp_results.get('processed_language', 'Unbekannt')}")
                        print("\n--- 3.1: Benannte Entitäten (NER) ---")
                        entities = nlp_results.get("entities", [])
                        if entities:
                            for ent in entities:
                                print(f"- Text: '{ent.get('text', '')}',  Label: {ent.get('label', '')}")
                        else:
                            print("Keine benannten Entitäten im verarbeiteten Text gefunden.")

                        print("\n--- 3.2: Lemmatisierung (Vollständige Liste der Token) ---")
                        lemmas = nlp_results.get("lemmas", [])
                        if lemmas:
                            print(json.dumps(lemmas, indent=2, ensure_ascii=False))
                            print(f"Gesamtzahl generierter Lemmata: {len(lemmas)}")
                        else:
                            print("Keine Lemmata generiert.")
                        print("-" * 40)

                        # NEU: SCHRITT 4: Intelligentes Chunking & Anreicherung mit DeepSeek
                        print("\n--- SCHRITT 4: Intelligentes Chunking & Anreicherung mit DeepSeek ---")
                        deepseek_enrichment_results = call_deepseek_enrichment_service(text_to_process, task="chunk_and_summarize")
                        
                        if deepseek_enrichment_results and deepseek_enrichment_results.get("status") == "success":
                            print("\n--- Ausgabe von SCHRITT 4 (DeepSeek Anreicherungsergebnisse) ---")
                            # Zeige die ersten 2 angereicherten Chunks zur Überprüfung an
                            print(json.dumps(deepseek_enrichment_results["results"][:2] if deepseek_enrichment_results["results"] else [], indent=2, ensure_ascii=False))
                            print(f"Gesamtzahl der von DeepSeek generierten Chunks: {len(deepseek_enrichment_results['results'])}")
                            print("-" * 40)

                            # HIER WÜRDE DER NÄCHSTE SCHRITT KOMMEN:
                            # 5. Embedding-Generierung (mit einem spezialisierten Embedding-Modell)
                            #    Loop durch deepseek_enrichment_results["results"]
                            #    Für jeden 'original_chunk' (oder 'summary') das Embedding generieren
                            # 6. Speicherung in der Vektordatenbank
                            #    Speichere 'original_chunk', 'summary', 'keywords' und das Embedding
                            #    (und alle Metadaten aus früheren Schritten wie NER-Entitäten, Sprache)

                        else:
                            print("\n--- Fehler/Warnung vom DeepSeek Anreicherungsdienst in SCHRITT 4 ---")
                            print(json.dumps(deepseek_enrichment_results, indent=2, ensure_ascii=False))
                            print("-" * 40)

                    else:
                        print("\n--- Fehler vom NLP-Dienst in SCHRITT 3 ---")
                        print(json.dumps(nlp_results, indent=2, ensure_ascii=False))
                        print("-" * 40)
                else:
                    print("\n--- Kein verarbeitbarer Text im Dokument nach der Strukturierung gefunden (SCHRITT 2 fehlgeschlagen) ---")
                    print("-" * 40)
            else:
                print("\n--- Fehler vom Strukturierungsdienst in SCHRITT 1 ---")
                print(json.dumps(structured_elements, indent=2, ensure_ascii=False))
                print("-" * 40)
            
            print("\n" + "="*10 + " PIPELINE ENDE " + "="*10)

        finally:
            sys.stdout = original_stdout
            logging.info(f"Vollständige Pipeline-Ausgabe in '{LOG_FILE_PATH}' geschrieben")
