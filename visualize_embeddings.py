# --- visualize_embeddings.py ---
# Dieses Skript verbindet sich mit Ihrer ChromaDB und visualisiert die Embeddings.
# Es verwendet UMAP zur Dimensionalitätsreduktion und Matplotlib zum Plotten.
# Führen Sie es aus mit: python visualize_embeddings.py

import chromadb
import numpy as np
import matplotlib.pyplot as plt
import os
import json
import logging
from umap import UMAP
from typing import List, Dict, Any

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

def visualize_chroma_embeddings():
    """
    Stellt eine Verbindung zu ChromaDB her, ruft Embeddings ab,
    reduziert ihre Dimensionalität mit UMAP und plottet sie.
    """
    logging.info(f"Verbinde mich mit ChromaDB unter: {CHROMADB_PATH}")
    try:
        client = chromadb.PersistentClient(path=CHROMADB_PATH)
        collection = client.get_or_create_collection(name=COLLECTION_NAME)
        logging.info(f"Verbunden mit Collection '{COLLECTION_NAME}'.")
    except Exception as e:
        logging.critical(f"Fehler beim Verbinden mit ChromaDB oder Collection: {e}", exc_info=True)
        print("Stellen Sie sicher, dass Ihre ChromaDB unter dem angegebenen Pfad existiert und zugänglich ist.")
        return

    # Abrufen aller Embeddings, Dokumente und Metadaten
    results = collection.get(
        ids=None,
        where=None,
        limit=None,
        offset=None,
        where_document=None,
        include=['embeddings', 'documents', 'metadatas']
    )

    embeddings = results.get('embeddings')
    documents = results.get('documents')
    metadatas = results.get('metadatas')
    ids = results.get('ids')

    # KORRIGIERT: Robuster Check für leere Embeddings-Liste oder leeres NumPy-Array
    if embeddings is None or \
       (isinstance(embeddings, list) and len(embeddings) == 0) or \
       (isinstance(embeddings, np.ndarray) and embeddings.size == 0):
        logging.warning("Keine Embeddings in der Collection gefunden. Bitte verarbeiten Sie zuerst Dokumente.")
        print("Keine Embeddings in der Collection gefunden. Bitte verarbeiten Sie zuerst Dokumente.")
        return

    logging.info(f"Abgerufen: {len(embeddings)} Embeddings.")

    embeddings_np = np.array(embeddings)

    # Dimensionalitätsreduktion mit UMAP
    # n_components: 2 für 2D-Plot
    # random_state: für reproduzierbare Ergebnisse
    logging.info("Starte Dimensionalitätsreduktion mit UMAP (auf 2 Dimensionen)...")
    try:
        reducer = UMAP(n_components=2, random_state=42)
        reduced_embeddings = reducer.fit_transform(embeddings_np)
        logging.info("Dimensionalitätsreduktion abgeschlossen.")
    except Exception as e:
        logging.critical(f"Fehler bei der UMAP-Reduktion: {e}", exc_info=True)
        print("Fehler bei der UMAP-Reduktion. Überprüfen Sie Ihre Daten oder UMAP-Installation.")
        return

    # Plotten der Ergebnisse
    plt.figure(figsize=(12, 10))
    
    # Versuchen, nach 'source_document' zu gruppieren, falls vorhanden
    source_documents = [m.get('source_document', 'Unbekannt') for m in metadatas]
    unique_sources = sorted(list(set(source_documents)))
    
    # Farben für verschiedene Dokumente
    colors = plt.cm.get_cmap('tab10', len(unique_sources)) if len(unique_sources) <= 10 else plt.cm.get_cmap('viridis', len(unique_sources))

    for i, source in enumerate(unique_sources):
        idx = [j for j, s in enumerate(source_documents) if s == source]
        plt.scatter(
            reduced_embeddings[idx, 0],
            reduced_embeddings[idx, 1],
            c=[colors(i)],
            label=source,
            s=50, # Punktgröße
            alpha=0.7 # Transparenz
        )
        
        # Optional: Beschriftung von einigen Punkten mit ihrer ID oder einem Teil des Textes
        # Dies kann den Plot sehr überladen, daher nur bei wenigen Punkten sinnvoll.
        # for j in idx:
        #     if np.random.rand() < 0.1: # Beschrifte nur 10% der Punkte
        #         plt.annotate(ids[j], (reduced_embeddings[j, 0], reduced_embeddings[j, 1]), textcoords="offset points", xytext=(5,5), ha='center', fontsize=8)


    plt.title(f"2D UMAP Projektion der Embeddings aus ChromaDB Collection '{COLLECTION_NAME}'")
    plt.xlabel("UMAP Dimension 1")
    plt.ylabel("UMAP Dimension 2")
    plt.legend(title="Quell-Dokumente", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    visualize_chroma_embeddings()
