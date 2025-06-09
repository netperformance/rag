# --- main.py (Orchestrator) ---
# FINALE VERSION v4: Implementiert erweitertes Logging zur Anzeige von strukturierten Elementen und Chunks.

import requests
import logging
import json
import os
import sys
from langdetect import detect, LangDetectException
from typing import Optional, List, Dict, Any
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Hilfsfunktion zum tiefen Zusammenführen von Dictionaries
def deep_update(base_dict, update_dict):
    for key, value in update_dict.items():
        if isinstance(value, dict) and key in base_dict and isinstance(base_dict[key], dict):
            deep_update(base_dict[key], value)
        else:
            base_dict[key] = value

# --- Konfigurationsladen ---
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
        "deepseek_enrichment_service": "http://127.0.0.1:8003/enrich-text/",
        "embedding_service": "http://127.0.0.1:8004/generate-embeddings/"
    },
    "orchestrator_config": {
        "pdf_file_to_check": "test_oc.pdf",
        "log_file_path": "logging.txt",
        "chunk_size": 450,
        "chunk_overlap": 100
    },
    "logging_config": {
        "enabled": True,
        "level": "INFO"
    },
    "ollama_config": {
        "ollama_base_url": "http://localhost:11434",
        "deepseek_model_name": "deepseek-coder-v2:latest"
    },
    "embedding_config": {
        "model_name": "intfloat/multilingual-e5-large",
        "chromadb_path": "./chroma_db",
        "collection_name": "rag_documents"
    },
    "deepseek_prompts": {
        "chunk_summary_keywords_prompt": "Fasse den folgenden Textabschnitt prägnant zusammen (max. 3 Sätze, konzentriere dich auf die Kernaussage) und extrahiere 3-5 Schlüsselbegriffe als JSON-Array. Das Ergebnis muss ein JSON-Objekt sein mit den Schlüsseln 'summary' (STRING) und 'keywords' (JSON-Array von Strings). Gib nur das JSON-Objekt zurück.",
        "chunk_questions_prompt": "Generiere 2-3 prägnante Fragen, die durch den folgenden Textabschnitt beantwortet werden können. Gib die Fragen als JSON-Array von Strings zurück. Gib nur das JSON-Array zurück."
    }
}

config = DEFAULT_CONFIG

try:
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
            loaded_config = json.load(f)
            deep_update(config, loaded_config)
except Exception as e:
    print(f"Fehler beim Laden der Konfiguration: {e}. Verwende Standardwerte.")

# Konfigurationswerte extrahieren
STRUCTURING_SERVICE_URL = config["service_urls"]["structuring_service"]
NLP_SERVICE_URL = config["service_urls"]["nlp_service"]
DEEPSEEK_ENRICHMENT_SERVICE_URL = config["service_urls"]["deepseek_enrichment_service"]
EMBEDDING_SERVICE_URL = config["service_urls"]["embedding_service"]
PDF_FILE_TO_CHECK = config["orchestrator_config"]["pdf_file_to_check"]
LOG_FILE_PATH = config["orchestrator_config"]["log_file_path"]
CHUNK_SIZE = config["orchestrator_config"]["chunk_size"]
CHUNK_OVERLAP = config["orchestrator_config"]["chunk_overlap"]

PROMPT_SUMMARY_KEYWORDS = config["deepseek_prompts"]["chunk_summary_keywords_prompt"]
PROMPT_QUESTIONS = config["deepseek_prompts"]["chunk_questions_prompt"]

# Saubere Logging-Konfiguration
log_level_str = config["logging_config"]["level"].upper()
log_level = getattr(logging, log_level_str, logging.INFO)

for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

logging.basicConfig(
    level=log_level,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE_PATH, mode='w', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)


def get_structured_data_from_service(file_path: str):
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
    try:
        language = detect(text)
        logging.info(f"Sprache des Textes erkannt: {language}")
        return language
    except LangDetectException:
        logging.warning("Sprache konnte nicht zuverlässig erkannt werden, standardmäßig auf Deutsch gesetzt.")
        return "de"
    except Exception as e:
        logging.error(f"Unerwarteter Fehler bei der Spracherkennung: {e}")
        return "de"

def process_text_with_nlp_service(text: str, language: Optional[str] = None):
    logging.info(f"Sende Text an NLP-Dienst zur Verarbeitung (Sprache: {language if language else 'Autoerkennung'})...")
    try:
        payload = {"text": text, "language": language} if language else {"text": text}
        response = requests.post(NLP_SERVICE_URL, json=payload, timeout=120)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        logging.error(f"Verbindung zum NLP-Dienst fehlgeschlagen: {e}")
        return {"error": "NLP service unavailable"}

def call_deepseek_enrichment_service(text_to_process: str, prompt_content: str):
    logging.info(f"Sende Text (Länge: {len(text_to_process)} Zeichen) zur DeepSeek-Anreicherung...")
    payload = {"text": text_to_process, "prompt_content": prompt_content}
    try:
        response = requests.post(DEEPSEEK_ENRICHMENT_SERVICE_URL, json=payload, timeout=900)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        logging.error(f"Fehler bei der Verbindung zum DeepSeek Anreicherungsdienst: {e}")
        return {"error": "DeepSeek enrichment service unavailable"}

def call_embedding_service(enriched_chunks: List[Dict[str, Any]], nlp_entities: List[Dict[str, str]], nlp_lemmas: List[Dict[str, str]]):
    logging.info(f"Sende {len(enriched_chunks)} angereicherte Chunks an den Embedding-Dienst.")
    
    embedding_request_items = []
    for i, chunk_data in enumerate(enriched_chunks):
        chunk_id = f"doc_{os.path.basename(PDF_FILE_TO_CHECK)}_chunk_{i}" 
        text_for_embedding = chunk_data.get("original_chunk", "")
        
        metadata = chunk_data.copy()
        metadata.update({
            "source_document": os.path.basename(PDF_FILE_TO_CHECK),
            "chunk_index": i,
            "nlp_entities": nlp_entities,
            "nlp_lemmas": nlp_lemmas
        })

        embedding_request_items.append({
            "id": chunk_id,
            "text": text_for_embedding,
            "metadata": metadata
        })

    try:
        response = requests.post(EMBEDDING_SERVICE_URL, json=embedding_request_items, timeout=600)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        logging.error(f"Fehler bei der Verbindung zum Embedding-Dienst: {e}")
        return {"error": "Embedding service unavailable"}


# --- Hauptausführungslogik ---
if __name__ == '__main__':
    try:
        logging.info("\n" + "="*10 + " PIPELINE START " + "="*10)

        # SCHRITT 1 & 2: Strukturierung und Textextraktion
        logging.info("\n--- SCHRITT 1: PDF Strukturierung ---")
        structured_elements = get_structured_data_from_service(PDF_FILE_TO_CHECK)
        if not (structured_elements and "error" not in structured_elements[0]):
            logging.error("\n--- FEHLER: Pipeline gestoppt wegen Fehler im Strukturierungsdienst. ---")
            logging.error(json.dumps(structured_elements, indent=2, ensure_ascii=False))
            sys.exit(1)
        
        # NEU: Detailliertes Logging der strukturierten Elemente
        logging.info("\n--- Ausgabe von SCHRITT 1 (Vollständige strukturierte Elemente) ---")
        logging.info(json.dumps(structured_elements, indent=2, ensure_ascii=False))
        logging.info(f"Gesamtzahl strukturierter Elemente: {len(structured_elements)}")
        logging.info("-" * 40)
        
        logging.info("\n--- SCHRITT 2: Textextraktion ---")
        text_to_process = "".join(
            element.get("text", "") + "\n\n" 
            for element in structured_elements 
            if element.get("type") in ["NarrativeText", "UncategorizedText", "ListItem", "Title"]
        )
        if not text_to_process.strip():
            logging.error("\n--- FEHLER: Kein verarbeitbarer Text im Dokument gefunden. ---")
            sys.exit(1)
        logging.info("Textextraktion erfolgreich abgeschlossen.")

        # SCHRITT 3: Basis-NLP-Verarbeitung (einmal für das ganze Dokument)
        logging.info("\n--- SCHRITT 3: NLP-Verarbeitung ---")
        detected_language = detect_text_language(text_to_process)
        nlp_results = process_text_with_nlp_service(text_to_process, detected_language)
        if not nlp_results or "error" in nlp_results:
            logging.error("\n--- FEHLER: Pipeline gestoppt wegen Fehler im NLP-Dienst. ---")
            logging.error(json.dumps(nlp_results, indent=2, ensure_ascii=False))
            sys.exit(1)
        
        base_nlp_entities = nlp_results.get("entities", [])
        base_nlp_lemmas = nlp_results.get("lemmas", [])
        
        # NEU: Detailliertes Logging der NLP-Ergebnisse
        logging.info("\n--- Ausgabe von SCHRITT 3 (NLP-Ergebnisse) ---")
        logging.info(f"Verarbeitete Sprache vom NLP-Dienst: {nlp_results.get('processed_language', 'Unbekannt')}")
        logging.info("\n--- 3.1: Benannte Entitäten (NER) ---")
        for ent in base_nlp_entities:
            logging.info(f"- Text: '{ent.get('text', '')}',  Label: {ent.get('label', '')}")
        logging.info("\n--- 3.2: Lemmatisierung (Vollständige Liste der Token) ---")
        logging.info(json.dumps(base_nlp_lemmas, indent=2, ensure_ascii=False))
        logging.info(f"Gesamtzahl generierter Lemmata: {len(base_nlp_lemmas)}")
        logging.info("-" * 40)
        
        # SCHRITT 4: Robustes Vor-Chunking mit RecursiveCharacterTextSplitter
        logging.info(f"\n--- SCHRITT 4: Robustes Vor-Chunking (Größe: {CHUNK_SIZE}, Überlappung: {CHUNK_OVERLAP}) ---")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP, length_function=len)
        preliminary_chunks = text_splitter.split_text(text_to_process)
        logging.info(f"Dokument wurde in {len(preliminary_chunks)} vorläufige Chunks aufgeteilt.")

        # NEU: Detailliertes Logging der erstellten Chunks
        logging.info("\n--- Ausgabe von SCHRITT 4 (Erstellte Chunks) ---")
        for i, chunk in enumerate(preliminary_chunks):
            logging.info(f"\n----- Chunk {i+1} / {len(preliminary_chunks)} -----")
            logging.info(chunk)
            logging.info("--------------------")
        
        # SCHRITT 5: Semantische Anreicherung für jeden einzelnen Chunk
        logging.info(f"\n--- SCHRITT 5: Starte Anreicherung für {len(preliminary_chunks)} Chunks mit DeepSeek ---")
        all_enriched_chunks = []
        for i, chunk_text in enumerate(preliminary_chunks):
            logging.info(f"Verarbeite Chunk {i + 1}/{len(preliminary_chunks)}...")
            current_chunk_data = {"original_chunk": chunk_text, "summary": "", "keywords": [], "questions": []}
            
            summary_prompt = f"{PROMPT_SUMMARY_KEYWORDS}\n\nText: {chunk_text}"
            summary_res = call_deepseek_enrichment_service(chunk_text, summary_prompt)
            if summary_res and summary_res.get("status") == "success" and isinstance(summary_res.get("results"), dict):
                current_chunk_data["summary"] = summary_res["results"].get("summary", "")
                current_chunk_data["keywords"] = summary_res["results"].get("keywords", [])
            
            questions_prompt = f"{PROMPT_QUESTIONS}\n\nText: {chunk_text}"
            questions_res = call_deepseek_enrichment_service(chunk_text, questions_prompt)
            if questions_res and questions_res.get("status") == "success" and isinstance(questions_res.get("results"), list):
                current_chunk_data["questions"] = questions_res["results"]
            
            all_enriched_chunks.append(current_chunk_data)

        logging.info(f"Anreicherung abgeschlossen. {len(all_enriched_chunks)} Chunks bereit für Embedding.")
        if all_enriched_chunks:
            logging.info("\n--- Ausgabe von SCHRITT 5 (Erster angereicherter Chunk) ---")
            logging.info(json.dumps(all_enriched_chunks[0], indent=2, ensure_ascii=False))

        # SCHRITT 6: Embedding-Generierung für die angereicherten Chunks
        if all_enriched_chunks:
            logging.info(f"\n--- SCHRITT 6: Sende {len(all_enriched_chunks)} angereicherte Chunks an den Embedding-Dienst ---")
            embedding_response = call_embedding_service(all_enriched_chunks, base_nlp_entities, base_nlp_lemmas)
            
            if embedding_response and embedding_response.get("status") == "success":
                logging.info("\n--- Ausgabe von SCHRITT 6 (Embedding-Ergebnisse) ---")
                logging.info(json.dumps(embedding_response, indent=2, ensure_ascii=False))
            else:
                logging.error("\n--- Fehler/Warnung vom Embedding-Dienst in SCHRITT 6 ---")
                logging.error(json.dumps(embedding_response, indent=2, ensure_ascii=False))
        else:
            logging.warning("\n--- SCHRITT 6 übersprungen: Keine Chunks nach Anreicherung vorhanden. ---")

        logging.info("\n" + "="*10 + " PIPELINE ENDE " + "="*10)

    except Exception as e:
        logging.critical("Ein unerwarteter, kritischer Fehler ist in der Haupt-Pipeline aufgetreten.", exc_info=True)

