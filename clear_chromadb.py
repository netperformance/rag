# --- clear_chromadb.py ---
# Dieses Skript verbindet sich mit Ihrer ChromaDB und löscht die angegebene Collection.
# Führen Sie es aus mit: python clear_chromadb.py

import chromadb
import os
import json
import logging

# --- Logging-Konfiguration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Konfiguration laden ---
CONFIG_FILE = "config.json"
DEFAULT_CONFIG = {
    "embedding_config": {
        "chromadb_path": "./chroma_db",
        "collection_name": "rag_documents"
    }
}

config = DEFAULT_CONFIG.copy()

try:
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
            loaded_config = json.load(f)
            # Eine einfache Merge-Funktion
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

CHROMADB_PATH = config["embedding_config"]["chromadb_path"]
COLLECTION_NAME = config["embedding_config"]["collection_name"]

def clear_chroma_collection():
    """
    Verbindet sich mit ChromaDB und löscht die angegebene Collection.
    Nach dem Löschen wird die Collection neu erstellt, um eine leere Instanz zu haben.
    """
    logging.info(f"Verbinde mich mit ChromaDB unter: {CHROMADB_PATH}")
    try:
        client = chromadb.PersistentClient(path=CHROMADB_PATH)
        
        logging.info(f"Versuche, Collection '{COLLECTION_NAME}' zu löschen...")
        try:
            client.delete_collection(name=COLLECTION_NAME)
            logging.info(f"Collection '{COLLECTION_NAME}' erfolgreich gelöscht.")
        except Exception as e:
            logging.warning(f"Collection '{COLLECTION_NAME}' konnte nicht gelöscht werden (möglicherweise nicht vorhanden): {e}")

        # Erstelle die Collection neu, um eine leere Datenbank für neue Einträge zu haben
        collection = client.get_or_create_collection(name=COLLECTION_NAME)
        logging.info(f"Collection '{COLLECTION_NAME}' erfolgreich neu erstellt (ist jetzt leer).")
        print(f"ChromaDB Collection '{COLLECTION_NAME}' wurde geleert und neu initialisiert.")

    except Exception as e:
        logging.critical(f"Kritischer Fehler beim Verbinden oder Leeren von ChromaDB: {e}", exc_info=True)
        print("Fehler beim Leeren der ChromaDB. Stellen Sie sicher, dass keine anderen Prozesse auf die Datenbank zugreifen.")

if __name__ == "__main__":
    # Eine Sicherheitsabfrage vor dem Leeren der Datenbank
    confirmation = input(f"Möchten Sie die ChromaDB Collection '{COLLECTION_NAME}' wirklich leeren? (ja/nein): ")
    if confirmation.lower() == 'ja':
        clear_chroma_collection()
    else:
        print("Vorgang abgebrochen.")
        

