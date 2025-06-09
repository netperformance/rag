# --- deepseek_enrichment_service.py ---
# Dieser Dienst nutzt ein lokal laufendes DeepSeek-Modell via Ollama (http://localhost:11434)
# für intelligentes Chunking und Anreicherung von Texten.
# Starten Sie diesen Dienst: uvicorn deepseek_enrichment_service:app --host 127.0.0.1 --port 8003 --reload

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests
import logging
import json
from typing import List, Dict, Any, Optional
import os # Benötigt, um den Dateipfad der Konfiguration zu handhaben
from json_repair import repair_json # NEU: Für robustes JSON-Parsing

app = FastAPI(title="DeepSeek Ollama Text Enrichment Service")

# --- Logging-Konfiguration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Konfiguration laden (für Ollama URL und Modellname) ---
# Diese Konfiguration wird nun aus der config.json Datei geladen.
CONFIG_FILE = "config.json"
DEFAULT_CONFIG = {
    "ollama_config": {
        "ollama_base_url": "http://localhost:11434",
        "deepseek_model_name": "deepseek-coder-v2:latest"
    }
}

config = DEFAULT_CONFIG.copy() # Start mit Standardwerten

try:
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
            loaded_config = json.load(f)
            # Eine einfache Merge-Funktion, um Standardwerte mit geladenen zu aktualisieren
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
OLLAMA_BASE_URL = config["ollama_config"]["ollama_base_url"]
DEEPSEEK_MODEL_NAME = config["ollama_config"]["deepseek_model_name"]

logging.info(f"DeepSeek Ollama Base URL: {OLLAMA_BASE_URL}")
logging.info(f"DeepSeek Model Name: {DEEPSEEK_MODEL_NAME}")


# --- Pydantic Modelle für Request und Response ---
class DeepSeekEnrichmentRequest(BaseModel):
    """
    Definiert das Anforderungsformat für den DeepSeek-Anreicherungsdienst.
    """
    text: str # Der Text, der verarbeitet werden soll
    prompt_content: str # Der vollständige Prompt-Inhalt für DeepSeek


class DeepSeekEnrichmentResponse(BaseModel):
    """
    Definiert das Antwortformat des DeepSeek-Anreicherungsdienstes.
    """
    status: str
    message: str
    results: Any # Ergebnis kann nun beliebige JSON-Struktur sein


# --- Hilfsfunktion zum Generieren von Text mit DeepSeek via Ollama ---
async def generate_deepseek_response_ollama(prompt_content: str, model_name: str, max_tokens: int = 4096) -> str:
    """
    Interagiert mit dem Ollama-DeepSeek-Modell, um eine Antwort zu generieren.
    """
    url = f"{OLLAMA_BASE_URL}/api/generate"
    
    data = {
        "model": model_name,
        "prompt": prompt_content,
        "stream": False,
        "options": {
            "num_predict": max_tokens,
            "temperature": 0.7,
            "top_p": 0.9
        }
    }

    try:
        logging.info(f"Sende Anfrage an Ollama für Modell '{model_name}'.")
        response = requests.post(url, json=data, timeout=300)
        response.raise_for_status()

        response_json = response.json()
        if "response" in response_json:
            return response_json["response"]
        elif "error" in response_json:
            raise HTTPException(status_code=500, detail={"error": f"Ollama Fehler: {response_json['error']}"})
        else:
            raise HTTPException(status_code=500, detail={"error": "Unerwartete Ollama-Antwortstruktur"})
    except requests.exceptions.ConnectionError as e:
        logging.error(f"Verbindungsfehler zu Ollama unter {OLLAMA_BASE_URL}: {e}")
        raise HTTPException(status_code=503, detail=f"Verbindung zu Ollama unter {OLLAMA_BASE_URL} fehlgeschlagen. Ist der Dienst gestartet und das Modell geladen?")
    except requests.exceptions.Timeout:
        logging.error("Timeout bei der Anfrage an Ollama.")
        raise HTTPException(status_code=504, detail="Timeout bei der Anfrage an Ollama.")
    except Exception as e:
        logging.critical(f"Ein unerwarteter Fehler ist im Anreicherungsdienst aufgetreten: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail={"error": "Interner Fehler bei der Ollama-Interaktion."})

# Robuste JSON-Extraktionsfunktion (Jetzt mit json_repair)
def _extract_and_repair_json(text: str) -> Optional[Any]:
    """
    Versucht, einen JSON-String aus einem größeren Text zu extrahieren und zu reparieren.
    Gibt das geparste JSON-Objekt/Array zurück, oder None bei Fehler.
    """
    text = text.strip()
    
    # Entferne Markdown-Codeblock-Wrapper, falls vorhanden
    if text.startswith("```json"):
        text = text[len("```json"):].strip()
    if text.endswith("```"):
        text = text[:-len("```")].strip()
    
    try:
        # Versuche, das JSON zu reparieren und direkt zu laden
        # repair_json gibt bei Erfolg ein Dict oder eine List zurück, bei Misserfolg eine Ausnahme
        parsed_json = repair_json(text)
        return parsed_json
    except Exception as e:
        logging.warning(f"Konnte JSON nicht mit json_repair extrahieren/reparieren: {e}. Roher Text (Anfang): {text[:500]}...")
        return None


# --- Generischer DeepSeek Anreicherungs-Endpunkt ---
@app.post("/enrich-text/", response_model=DeepSeekEnrichmentResponse)
async def enrich_text_with_deepseek(request: DeepSeekEnrichmentRequest):
    """
    Verarbeitet Text mit DeepSeek (via Ollama) basierend auf dem bereitgestellten Prompt.
    Erwartet eine JSON-Ausgabe von DeepSeek, die als 'results' zurückgegeben wird.
    """
    text_to_process = request.text
    full_prompt_content = request.prompt_content

    logging.info(f"Empfange Chunk zur DeepSeek-Anreicherung via Ollama (Länge Text: {len(text_to_process)} Zeichen, Länge Prompt: {len(full_prompt_content)} Zeichen).")

    try:
        raw_deepseek_output = await generate_deepseek_response_ollama(full_prompt_content, DEEPSEEK_MODEL_NAME)
        logging.info(f"DeepSeek (Ollama) hat geantwortet (erster Teil): {raw_deepseek_output[:200]}...")

        # Nutze die neue, robustere JSON-Extraktions- und Reparaturfunktion
        parsed_results = _extract_and_repair_json(raw_deepseek_output)
        
        if parsed_results is not None: # Prüfe auf None, da die Funktion jetzt direkt das geparste Ergebnis zurückgibt
            return DeepSeekEnrichmentResponse(
                status="success",
                message="Text erfolgreich mit DeepSeek (Ollama) basierend auf Prompt angereichert.",
                results=parsed_results
            )
        else:
            logging.error(f"Kein gültiges JSON aus DeepSeek (Ollama) Ausgabe extrahiert oder repariert. Rohe Ausgabe: {raw_deepseek_output}")
            return DeepSeekEnrichmentResponse(
                status="error", # Ändere Status auf "error", da das Parsen fehlgeschlagen ist
                message=f"DeepSeek (Ollama) Antwort konnte nicht geparst/repariert werden. Rohe Ausgabe: {raw_deepseek_output}",
                results=None
            )

    except HTTPException as e:
        raise e
    except Exception as e:
        logging.critical(f"Ein unerwarteter Fehler ist im Anreicherungsdienst aufgetreten: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail={"error": "Interner Serverfehler im Anreicherungsdienst."})
