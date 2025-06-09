# --- embedding_service.py ---
# Dieser Dienst ist verantwortlich für die Generierung von Embeddings
# und deren Speicherung in einer ChromaDB-Vektordatenbank.
# Starten Sie diesen Dienst: uvicorn embedding_service:app --host 127.0.0.1 --port 8004 --reload

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.utils import embedding_functions
import logging
import json
import os
from typing import List, Dict, Any, Optional

app = FastAPI(title="Embedding and ChromaDB Service")

# --- Logging-Konfiguration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Konfiguration laden (für Embedding-Modell und ChromaDB-Pfad) ---
CONFIG_FILE = "config.json"
DEFAULT_CONFIG = {
    "embedding_config": {
        "model_name": "intfloat/multilingual-e5-large",
        "chromadb_path": "./chroma_db",
        "collection_name": "rag_documents"
    }
}

config = DEFAULT_CONFIG.copy()

try:
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
            loaded_config = json.load(f)
            def deep_update(base_dict, update_dict):
                for key, value in update_dict.items():
                    if isinstance(value, dict) and key in base_dict and isinstance(base_dict[key], dict):
                        deep_update(base_dict[key], value)
                    else:
                        base_dict[key] = value
            deep_update(config, loaded_config)
        logging.info(f"Konfiguration aus '{CONFIG_FILE}' erfolgreich geladen.")
    else:
        logging.warning(f"Konfigurationsdatei '{CONFIG_FILE}' nicht gefunden. Verwende Standardwerte.")
except json.JSONDecodeError:
    logging.error(f"Fehler beim Parsen der Konfigurationsdatei '{CONFIG_FILE}'. Überprüfen Sie das JSON-Format. Verwende Standardwerte.")
except Exception as e:
    logging.error(f"Unerwarteter Fehler beim Laden der Konfiguration: {e}. Verwende Standardwerte.")

# Konfigurationswerte extrahieren
EMBEDDING_MODEL_NAME = config["embedding_config"]["model_name"]
CHROMADB_PATH = config["embedding_config"]["chromadb_path"]
COLLECTION_NAME = config["embedding_config"]["collection_name"]

# Globale Variablen für das Embedding-Modell und ChromaDB Client/Collection
embedding_model: Optional[SentenceTransformer] = None
chroma_client: Optional[chromadb.PersistentClient] = None
chroma_collection: Optional[chromadb.api.models.Collection.Collection] = None

# --- Modell und ChromaDB beim Start laden ---
@app.on_event("startup")
async def load_embedding_model_and_chromadb():
    global embedding_model, chroma_client, chroma_collection

    logging.info(f"Starte das Laden des Embedding-Modells: {EMBEDDING_MODEL_NAME}...")
    try:
        # Embedding-Modell laden
        embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        logging.info(f"Embedding-Modell '{EMBEDDING_MODEL_NAME}' erfolgreich geladen.")
    except Exception as e:
        logging.critical(f"Kritischer Fehler beim Laden des Embedding-Modells: {e}", exc_info=True)
        embedding_model = None
        raise RuntimeError("Embedding-Modell konnte nicht geladen werden. Der Dienst kann nicht starten.")

    logging.info(f"Initialisiere ChromaDB unter: {CHROMADB_PATH}...")
    try:
        # ChromaDB Client initialisieren
        chroma_client = chromadb.PersistentClient(path=CHROMADB_PATH)
        # Collection erstellen oder abrufen
        chroma_collection = chroma_client.get_or_create_collection(
            name=COLLECTION_NAME,
            # Eine Embedding-Funktion ist hier nicht notwendig, da wir die Embeddings extern generieren
            # und direkt übergeben. Wenn Chroma die Embeddings generieren soll, wäre dies hier anzugeben.
        )
        logging.info(f"ChromaDB Collection '{COLLECTION_NAME}' erfolgreich initialisiert.")
    except Exception as e:
        logging.critical(f"Kritischer Fehler beim Initialisieren von ChromaDB: {e}", exc_info=True)
        chroma_client = None
        chroma_collection = None
        raise RuntimeError("ChromaDB konnte nicht initialisiert werden. Der Dienst kann nicht starten.")

# --- Pydantic Modelle für Request und Response ---
class EmbeddingRequestItem(BaseModel):
    """Definiert das Format für einen einzelnen Text-Chunk zur Einbettung."""
    id: str # Eine eindeutige ID für den Chunk
    text: str # Der Text, der eingebettet werden soll (original_chunk oder summary)
    metadata: Dict[str, Any] = {} # Zusätzliche Metadaten (z.B. keywords, language, source_page)

class EmbeddingResponse(BaseModel):
    """Definiert das Antwortformat des Embedding-Dienstes."""
    status: str
    message: str
    num_embeddings_generated: int
    num_docs_added_to_db: int
    failed_ids: List[str] = []

@app.post("/generate-embeddings/", response_model=EmbeddingResponse)
async def generate_and_store_embeddings(items: List[EmbeddingRequestItem]):
    """
    Generiert Embeddings für eine Liste von Text-Chunks und speichert diese in ChromaDB.
    """
    if embedding_model is None or chroma_collection is None:
        logging.error("Embedding-Modell oder ChromaDB nicht initialisiert.")
        raise HTTPException(status_code=503, detail="Embedding-Service ist nicht bereit.")

    if not items:
        return EmbeddingResponse(status="success", message="Keine Elemente zur Verarbeitung erhalten.", num_embeddings_generated=0, num_docs_added_to_db=0)

    texts_to_embed = []
    ids = []
    metadatas = []
    
    for item in items:
        texts_to_embed.append(item.text)
        ids.append(item.id)
        
        # KORRIGIERT: Konvertiere komplexe Metadaten (Listen, Dicts) in JSON-Strings, um sie in ChromaDB zu speichern.
        processed_metadata = item.metadata.copy()
        # HIER WURDEN 'nlp_entities' UND 'nlp_lemmas' HINZUGEFÜGT
        for key in ['keywords', 'questions', 'key_sentences', 'named_entities_deepseek', 'nlp_entities', 'nlp_lemmas']:
            if key in processed_metadata and isinstance(processed_metadata[key], (list, dict)):
                try:
                    processed_metadata[key] = json.dumps(processed_metadata[key], ensure_ascii=False)
                except Exception as e:
                    logging.warning(f"Konnte Metadaten-Liste '{key}' nicht in JSON-String konvertieren für ID {item.id}: {e}. Speichere als leeren String.")
                    processed_metadata[key] = ""
        
        metadatas.append(processed_metadata)

    logging.info(f"Generiere Embeddings für {len(texts_to_embed)} Texte...")
    try:
        # Batch-Embedding-Generierung
        embeddings = embedding_model.encode(texts_to_embed, convert_to_list=True)
        logging.info(f"Erfolgreich {len(embeddings)} Embeddings generiert.")
    except Exception as e:
        logging.error(f"Fehler bei der Generierung der Embeddings: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Fehler bei der Embedding-Generierung.")

    logging.info(f"Speichere {len(embeddings)} Embeddings in ChromaDB Collection '{COLLECTION_NAME}'...")
    try:
        # Sicherstellen, dass alle Listen die gleiche Länge haben
        if not (len(ids) == len(embeddings) == len(texts_to_embed) == len(metadatas)) and len(embeddings) > 0:
            logging.error("Längen der Listen für IDs, Embeddings, Texte und Metadaten stimmen nicht überein.")
            raise HTTPException(status_code=500, detail="Interne Dateninkonsistenz vor dem Speichern in DB.")

        chroma_collection.add(
            embeddings=embeddings,
            documents=texts_to_embed, # Speichern Sie den Originaltext als Dokument
            metadatas=metadatas,
            ids=ids
        )
        logging.info(f"Erfolgreich {len(embeddings)} Embeddings in ChromaDB gespeichert.")
        return EmbeddingResponse(
            status="success",
            message="Embeddings erfolgreich generiert und in ChromaDB gespeichert.",
            num_embeddings_generated=len(embeddings),
            num_docs_added_to_db=len(embeddings)
        )
    except Exception as e:
        logging.error(f"Fehler beim Speichern der Embeddings in ChromaDB: {e}", exc_info=True)
        # Erstelle eine detailliertere Fehlermeldung für die Antwort
        error_message = f"Fehler beim Speichern in ChromaDB: {str(e)}"
        return EmbeddingResponse(
            status="error",
            message=error_message,
            num_embeddings_generated=len(embeddings),
            num_docs_added_to_db=0,
            failed_ids=ids # Alle IDs werden als fehlgeschlagen markiert
        )
