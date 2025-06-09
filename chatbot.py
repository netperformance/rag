# --- chatbot.py ---
# Dieses Skript implementiert den interaktiven RAG-Chatbot.
# Es verwendet die bestehende Vektordatenbank, um Fragen zu einem Dokument zu beantworten.
# Starten Sie es mit: python chatbot.py

import json
import logging
import os
import requests
from typing import List, Dict, Any, Optional

import chromadb
from sentence_transformers import SentenceTransformer

# --- Konfiguration laden ---
# Stellt sicher, dass dieselben Modelle und Pfade wie in der Pipeline verwendet werden.
CONFIG_FILE = "config.json"
DEFAULT_CONFIG = {
    "ollama_config": {
        "ollama_base_url": "http://localhost:11434",
        "deepseek_model_name": "deepseek-coder-v2:latest"
    },
    "embedding_config": {
        "model_name": "intfloat/multilingual-e5-large",
        "chromadb_path": "./chroma_db",
        "collection_name": "rag_documents"
    },
    "rag_config": {
        "num_relevant_chunks": 5,
        "prompt_template": """
Beantworte die folgende Frage ausschließlich auf Basis des untenstehenden Kontexts.
Sei präzise und zitiere, wenn möglich, direkt aus dem Kontext.
Wenn die Antwort im Kontext nicht eindeutig zu finden ist, antworte mit: 'Diese Information ist im bereitgestellten Dokument nicht enthalten.'

Kontext:
{context}

Frage: {question}

Antwort:
"""
    }
}

def deep_update(base_dict, update_dict):
    """Führt verschachtelte Dictionaries zusammen."""
    for key, value in update_dict.items():
        if isinstance(value, dict) and key in base_dict and isinstance(base_dict[key], dict):
            deep_update(base_dict[key], value)
        else:
            base_dict[key] = value

config = DEFAULT_CONFIG.copy()
try:
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
            loaded_config = json.load(f)
            deep_update(config, loaded_config)
except Exception as e:
    print(f"WARNUNG: Fehler beim Laden der Konfiguration: {e}. Verwende Standardwerte.")

# Logging-Konfiguration
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')


class RAGChatbot:
    """
    Ein Chatbot, der das Retrieval-Augmented Generation (RAG) Muster implementiert.
    """

    def __init__(self):
        """
        Initialisiert den Chatbot durch Laden der Modelle und Herstellen der DB-Verbindung.
        """
        print("Initialisiere den RAG-Chatbot...")
        self.config = config
        self.embedding_model = self._load_embedding_model()
        self.collection = self._connect_to_chromadb()
        self.llm_url = self.config["ollama_config"]["ollama_base_url"] + "/api/generate"
        self.llm_model = self.config["ollama_config"]["deepseek_model_name"]
        self.prompt_template = self.config["rag_config"]["prompt_template"]
        self.num_relevant_chunks = self.config["rag_config"]["num_relevant_chunks"]
        print("Initialisierung abgeschlossen. Der Chatbot ist bereit.")

    def _load_embedding_model(self) -> SentenceTransformer:
        """Lädt das SentenceTransformer-Modell."""
        model_name = self.config["embedding_config"]["model_name"]
        print(f"Lade Embedding-Modell: {model_name}...")
        try:
            return SentenceTransformer(model_name)
        except Exception as e:
            logging.critical(f"Kritisches Problem beim Laden des Embedding-Modells: {e}", exc_info=True)
            raise RuntimeError("Das Embedding-Modell konnte nicht geladen werden.")

    def _connect_to_chromadb(self) -> chromadb.Collection:
        """Stellt eine Verbindung zur ChromaDB-Collection her."""
        db_path = self.config["embedding_config"]["chromadb_path"]
        collection_name = self.config["embedding_config"]["collection_name"]
        print(f"Verbinde mit ChromaDB unter '{db_path}' und Collection '{collection_name}'...")
        try:
            client = chromadb.PersistentClient(path=db_path)
            return client.get_collection(name=collection_name)
        except Exception as e:
            logging.critical(f"Konnte keine Verbindung zur ChromaDB-Collection herstellen: {e}", exc_info=True)
            raise RuntimeError(f"Stellen Sie sicher, dass die Collection '{collection_name}' existiert und die Pipeline erfolgreich durchlaufen ist.")

    def _query_llm(self, prompt: str) -> str:
        """Sendet einen Prompt an das LLM (DeepSeek via Ollama) und gibt die Antwort zurück."""
        print("Sende Anfrage an das Sprachmodell...")
        payload = {
            "model": self.llm_model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.1} # Niedrige Temperatur für faktenbasierte Antworten
        }
        try:
            response = requests.post(self.llm_url, json=payload, timeout=120)
            response.raise_for_status()
            response_json = response.json()
            return response_json.get("response", "Fehler: Keine Antwort vom Modell erhalten.").strip()
        except requests.RequestException as e:
            logging.error(f"Fehler bei der Verbindung zu Ollama: {e}")
            return "Fehler: Konnte keine Verbindung zum Sprachmodell herstellen. Ist Ollama gestartet?"

    def ask(self, question: str) -> str:
        """
        Führt den vollständigen RAG-Prozess für eine gegebene Frage aus.
        """
        # Phase 1 & 2: Vektorisierung der Frage und Retrieval der Chunks
        print("1. Vektorisiere Frage und suche nach relevanten Informationen...")
        question_embedding = self.embedding_model.encode(question).tolist()
        
        results = self.collection.query(
            query_embeddings=[question_embedding],
            n_results=self.num_relevant_chunks,
            include=["documents"]
        )
        
        relevant_docs = results.get('documents', [[]])[0]
        if not relevant_docs:
            return "Es konnten keine relevanten Informationen im Dokument gefunden werden."
        
        # Phase 3: Anreicherung (Prompt-Konstruktion)
        print("2. Erstelle Kontext für das Sprachmodell...")
        context = "\n\n---\n\n".join(relevant_docs)
        
        prompt = self.prompt_template.format(context=context, question=question)
        
        # Phase 4: Generierung der Antwort
        print("3. Generiere die finale Antwort...")
        answer = self._query_llm(prompt)
        
        return answer

    def start_chat(self):
        """Startet die interaktive Chat-Schleife."""
        print("\n--- RAG Chatbot gestartet ---")
        print("Stellen Sie Ihre Fragen zum Dokument. Geben Sie 'exit' oder 'quit' ein, um den Chat zu beenden.")
        while True:
            try:
                question = input("\nIhre Frage: ")
                if question.lower() in ["exit", "quit", "beenden"]:
                    print("Chatbot wird beendet. Auf Wiedersehen!")
                    break
                
                answer = self.ask(question)
                print("\nAntwort:")
                print(answer)

            except KeyboardInterrupt:
                print("\nChatbot wird beendet. Auf Wiedersehen!")
                break
            except Exception as e:
                logging.error("Ein unerwarteter Fehler ist im Chat aufgetreten.", exc_info=True)
                print(f"Ein Fehler ist aufgetreten: {e}")

if __name__ == "__main__":
    # Stellt sicher, dass Ollama und die anderen Dienste laufen, bevor Sie dies ausführen.
    chatbot = RAGChatbot()
    chatbot.start_chat()
