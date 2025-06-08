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
    # Optionale Felder, um den Anreicherungstyp zu steuern
    task: str = "chunk_and_summarize" # Mögliche Werte: "chunk_and_summarize", "extract_keywords", "custom"
    # Diese Chunk-Parameter sind für das interne Prompting gedacht, nicht für die externe Steuerung des LLM.
    # Das LLM erhält den gesamten Text und soll die Chunks selbst identifizieren.
    chunk_size: Optional[int] = 500
    overlap: Optional[int] = 50

class DeepSeekEnrichmentResponse(BaseModel):
    """
    Definiert das Antwortformat des DeepSeek-Anreicherungsdienstes.
    """
    status: str
    message: str
    results: List[Dict[str, Any]] # Liste von angereicherten Chunks oder anderen Ergebnissen

# --- Hilfsfunktion zum Generieren von Text mit DeepSeek via Ollama ---
async def generate_deepseek_response_ollama(prompt_content: str, model_name: str, max_tokens: int = 4096) -> str:
    """
    Interagiert mit dem Ollama-DeepSeek-Modell, um eine Antwort zu generieren.
    """
    url = f"{OLLAMA_BASE_URL}/api/generate"
    
    # Ollama erwartet den Prompt direkt in der "prompt" Sektion
    data = {
        "model": model_name,
        "prompt": prompt_content,
        "stream": False, # Wir wollen die vollständige Antwort auf einmal
        "options": {
            "num_predict": max_tokens, # Maximale Anzahl der zu generierenden Tokens
            "temperature": 0.7,
            "top_p": 0.9
        }
    }

    try:
        logging.info(f"Sende Anfrage an Ollama für Modell '{model_name}'.")
        response = requests.post(url, json=data, timeout=300) # Timeout anpassen
        response.raise_for_status() # Löst einen HTTPError für schlechte Antworten (4xx oder 5xx) aus

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
        logging.error(f"Fehler bei der Ollama-Anfrage: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail={"error": "Interner Fehler bei der Ollama-Interaktion"})


# --- Intelligentes Chunking und Anreicherung mit DeepSeek ---
@app.post("/enrich-text/", response_model=DeepSeekEnrichmentResponse)
async def enrich_text_with_deepseek(request: DeepSeekEnrichmentRequest):
    """
    Verarbeitet Text mit DeepSeek (via Ollama) für intelligentes Chunking und/oder Anreicherung.
    Die Ausgabe von DeepSeek wird als JSON-Struktur erwartet.
    """
    text_to_process = request.text
    task_type = request.task

    logging.info(f"Empfange Text zur DeepSeek-Anreicherung via Ollama (Task: {task_type}, Länge: {len(text_to_process)} Zeichen).")

    # Prompt-Erstellung basierend auf dem Task-Typ
    prompt_template = """
    Als hochintelligente Textanalyse-KI ist Ihre Aufgabe, den folgenden Text zu verarbeiten.
    Bitte extrahieren und strukturieren Sie die Informationen gemäß dem angegebenen Task.
    Geben Sie Ihre Antwort ausschließlich als gültiges JSON zurück.

    Task: {task_description}

    Text:
    {text}
    """

    task_description = ""
    if task_type == "chunk_and_summarize":
        task_description = (
            f"Teile den Text in semantisch kohärente Abschnitte. "
            f"Fasse jeden Abschnitt prägnant zusammen (max. 3 Sätze) und extrahiere 3-5 Schlüsselbegriffe "
            f"als JSON-Array. Das Ergebnis sollte eine JSON-Liste von Objekten sein, "
            f"wobei jedes Objekt 'original_chunk', 'summary' und 'keywords' enthält."
        )
    elif task_type == "extract_keywords":
        task_description = (
            f"Extrahiere die 10 wichtigsten Schlüsselbegriffe und Phrasen aus dem Text. "
            f"Das Ergebnis sollte ein JSON-Objekt mit dem Schlüssel 'keywords' sein, "
            f"dessen Wert ein JSON-Array von Strings ist."
        )
    elif task_type == "custom":
        task_description = (
            f"Fasse den Text zusammen und extrahiere alle Personen- und Organisationsnamen. "
            f"Das Ergebnis sollte ein JSON-Objekt mit den Schlüsseln 'summary' und 'entities' sein, "
            f"wobei 'entities' ein JSON-Array von Objekten mit 'name' und 'type' ist."
        )
    else:
        raise HTTPException(status_code=400, detail="Ungültiger 'task'-Typ angegeben. Unterstützt: 'chunk_and_summarize', 'extract_keywords', 'custom'.")

    # Finaler Prompt
    full_prompt = prompt_template.format(task_description=task_description, text=text_to_process)

    try:
        raw_deepseek_output = await generate_deepseek_response_ollama(full_prompt, DEEPSEEK_MODEL_NAME)
        logging.info(f"DeepSeek (Ollama) hat geantwortet (erster Teil): {raw_deepseek_output[:200]}...")

        # KORRIGIERTE JSON-EXTRAKTION: Entferne Markdown-Codeblock-Wrapper
        json_string = raw_deepseek_output.strip()
        if json_string.startswith("```json"):
            json_string = json_string[len("```json"):].strip()
        if json_string.endswith("```"):
            json_string = json_string[:-len("```")].strip()
        
        # Sicherstellen, dass die JSON-Zeichenkette mit '{' oder '[' beginnt und endet
        json_start = json_string.find('{')
        json_array_start = json_string.find('[')

        if json_start == -1 and json_array_start == -1:
            raise json.JSONDecodeError("Kein JSON-Startzeichen gefunden", json_string, 0)
        
        # Bestimme den tatsächlichen Startpunkt des JSON
        actual_json_start_idx = -1
        if json_start != -1 and (json_array_start == -1 or json_start < json_array_start):
            actual_json_start_idx = json_start
            expected_end_char = '}'
        elif json_array_start != -1:
            actual_json_start_idx = json_array_start
            expected_end_char = ']'
        else:
            raise json.JSONDecodeError("Kein gültiges JSON-Startzeichen gefunden", json_string, 0)

        json_string = json_string[actual_json_start_idx:]

        # Robustere Methode zum Auffinden des JSON-Endes (für verschachtelte Strukturen)
        # Dies ist eine vereinfachte Methode und kann bei sehr komplexen/fehlformatierten JSONs fehlschlagen.
        # Eine externe Bibliothek wie `json_repair` könnte hier robuster sein.
        brace_balance = 0
        bracket_balance = 0
        json_end_idx = -1
        for i, char in enumerate(json_string):
            if char == '{':
                brace_balance += 1
            elif char == '}':
                brace_balance -= 1
            elif char == '[':
                bracket_balance += 1
            elif char == ']':
                bracket_balance -= 1
            
            if brace_balance == 0 and bracket_balance == 0 and (char == expected_end_char):
                json_end_idx = i
                break
        
        if json_end_idx == -1:
             raise json.JSONDecodeError("Kein gültiges JSON-Endzeichen gefunden oder unvollständige Struktur", json_string, 0)

        json_string = json_string[:json_end_idx + 1]


        try:
            parsed_results = json.loads(json_string)
            
            # Stellen Sie sicher, dass parsed_results eine Liste ist, wenn "chunk_and_summarize" erwartet wird
            if task_type == "chunk_and_summarize" and not isinstance(parsed_results, list):
                logging.warning("DeepSeek-Ausgabe für 'chunk_and_summarize' war kein JSON-Array. Versuch, es als Liste in ein Objekt zu packen.")
                parsed_results = [parsed_results]

            # Stellen Sie sicher, dass parsed_results eine Liste ist, wenn "extract_keywords" erwartet wird
            # (Diese Logik ist möglicherweise nur sinnvoll, wenn die erwartete Antwort eine Liste von Keywords ist,
            # aber das LLM ein { "keywords": [...] } Objekt zurückgibt, das dann in eine Liste von Objekten
            # umgewandelt wird, wie es das Response-Modell erwartet.)
            if task_type == "extract_keywords" and isinstance(parsed_results, dict) and "keywords" in parsed_results:
                parsed_results = [{"keywords": parsed_results["keywords"]}]
            elif task_type == "extract_keywords" and not isinstance(parsed_results, list):
                logging.warning("DeepSeek-Ausgabe für 'extract_keywords' war kein JSON-Array. Versuch, es als Liste in ein Objekt zu packen.")
                parsed_results = [parsed_results]


            logging.info(f"DeepSeek (Ollama) Ausgabe erfolgreich als JSON geparst.")
            return DeepSeekEnrichmentResponse(
                status="success",
                message="Text erfolgreich mit DeepSeek (Ollama) angereichert.",
                results=parsed_results
            )
        except json.JSONDecodeError as e:
            logging.error(f"Fehler beim Parsen der JSON-Antwort von DeepSeek (Ollama): {e}. Rohe Ausgabe nach Bereinigung: {json_string}", exc_info=True)
            return DeepSeekEnrichmentResponse(
                status="error",
                message=f"DeepSeek (Ollama) Antwort konnte nicht geparst werden: {e}. Rohe Ausgabe: {raw_deepseek_output}",
                results=[]
            )
    except HTTPException as e:
        raise e # Re-raise HTTPExceptions from generate_deepseek_response_ollama
    except Exception as e:
        logging.critical(f"Ein unerwarteter Fehler ist im Anreicherungsdienst aufgetreten: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail={"error": "Interner Serverfehler im Anreicherungsdienst."})
