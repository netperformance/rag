from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification
import spacy
import logging
import json
import os # Wird nicht direkt für hartkodierte Pfade verwendet, ist aber oft nützlich

app = FastAPI()

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# --- Konfiguration laden ---
CONFIG_FILE = "config.json"
DEFAULT_CONFIG = {
    "nlp_model_config": {
        "ner_model_name": "domischwimmbeck/bert-base-german-cased-fine-tuned-ner",
        "lemmatization_enabled": True,
        "lemmatization_model_name": "de_core_news_sm"
    },
    "service_urls": {
        "structuring_service": "http://127.0.0.1:8001/structure-pdf/",
        "nlp_service": "http://127.0.0.1:8002/process/"
    }
}

config = DEFAULT_CONFIG # Beginne mit Standardwerten

try:
    with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
        loaded_config = json.load(f)
        # Standardwerte mit geladenen Werten aktualisieren
        # Dies erlaubt, dass nur Teile der Konfiguration in der Datei stehen müssen
        config.update(loaded_config)
    logging.info(f"Konfiguration aus '{CONFIG_FILE}' erfolgreich geladen.")
except FileNotFoundError:
    logging.warning(f"Konfigurationsdatei '{CONFIG_FILE}' nicht gefunden. Verwende Standardwerte.")
except json.JSONDecodeError:
    logging.error(f"Fehler beim Parsen der Konfigurationsdatei '{CONFIG_FILE}'. Überprüfen Sie das JSON-Format. Verwende Standardwerte.")
except Exception as e:
    logging.error(f"Unerwarteter Fehler beim Laden der Konfiguration: {e}. Verwende Standardwerte.")

# Konfigurationswerte aus dem geladenen/Standard-Konfig-Objekt extrahieren
# Vermeidet hartkodierte Strings durch Verwendung der Keys aus DEFAULT_CONFIG
NER_MODEL_NAME = config.get("nlp_model_config", {}).get("ner_model_name", DEFAULT_CONFIG["nlp_model_config"]["ner_model_name"])
LEMMATIZATION_ENABLED = config.get("nlp_model_config", {}).get("lemmatization_enabled", DEFAULT_CONFIG["nlp_model_config"]["lemmatization_enabled"])
LEMMATIZATION_MODEL_NAME = config.get("nlp_model_config", {}).get("lemmatization_model_name", DEFAULT_CONFIG["nlp_model_config"]["lemmatization_model_name"])

# --- NER-Modell laden ---
ner_pipeline = None
try:
    logging.info(f"Lade NER-Modell: {NER_MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(NER_MODEL_NAME)
    model = AutoModelForTokenClassification.from_pretrained(NER_MODEL_NAME)
    ner_pipeline = pipeline(
        "ner",
        model=model,
        tokenizer=tokenizer,
        aggregation_strategy="simple" # Aggregiere Token, die zu einer Entität gehören
    )
    logging.info("NER-Modell erfolgreich geladen.")
except Exception as e:
    logging.error(f"Fehler beim Laden des NER-Modells '{NER_MODEL_NAME}': {e}")
    # Der Dienst wird gestartet, aber /process wird fehlschlagen, wenn ner_pipeline None ist.

# --- Lemmatisierungsmodell laden (falls aktiviert) ---
nlp_lemmatizer = None
if LEMMATIZATION_ENABLED:
    try:
        logging.info(f"Lade Lemmatisierungsmodell: {LEMMATIZATION_MODEL_NAME} (spaCy)...")
        # Versuche, das spaCy-Modell zu laden. Wenn es nicht gefunden wird, muss es heruntergeladen werden
        try:
            nlp_lemmatizer = spacy.load(LEMMATIZATION_MODEL_NAME)
        except OSError:
            logging.warning(f"spaCy Modell '{LEMMATIZATION_MODEL_NAME}' nicht gefunden. Versuche es herunterzuladen...")
            spacy.cli.download(LEMMATIZATION_MODEL_NAME) # Hier kann es zu einem Problem kommen, wenn Berechtigungen fehlen oder spacy.cli nicht funktioniert
            nlp_lemmatizer = spacy.load(LEMMATIZATION_MODEL_NAME)
        logging.info("Lemmatisierungsmodell erfolgreich geladen.")
    except Exception as e:
        logging.error(f"Fehler beim Laden oder Herunterladen des Lemmatisierungsmodells '{LEMMATIZATION_MODEL_NAME}': {e}")
        # Wenn Lemmatisierung fehlschlägt, setzen Sie nlp_lemmatizer auf None, damit es später nicht verwendet wird
        nlp_lemmatizer = None

class TextPayload(BaseModel):
    text: str

@app.post("/process/")
async def process_text(payload: TextPayload):
    if ner_pipeline is None:
        # Falls das NER-Modell beim Start nicht geladen werden konnte
        raise HTTPException(status_code=503, detail="NER-Modell konnte nicht geladen werden, Dienst nicht verfügbar.")

    text = payload.text
    logging.info(f"Empfange Text zur Verarbeitung (Länge: {len(text)} Zeichen).")

    # --- NER-Verarbeitung ---
    ner_results = []
    try:
        entities = ner_pipeline(text)
        # Sicherstellen, dass die Ausgabe für Entities korrekt ist, auch wenn 'word' oder 'entity_group' fehlen
        ner_results = [{"text": ent.get('word', ''), "label": ent.get('entity_group', '')} for ent in entities]
        logging.info(f"NER: {len(ner_results)} Entitäten gefunden.")
    except Exception as e:
        logging.error(f"Fehler bei der NER-Verarbeitung: {e}")
        ner_results = [] # Gib leere Liste bei Fehler zurück

    # --- Lemmatisierung ---
    lemmas_results = []
    if LEMMATIZATION_ENABLED and nlp_lemmatizer:
        try:
            doc = nlp_lemmatizer(text)
            # Sicherstellen, dass die Ausgabe für Lemmata korrekt ist
            lemmas_results = [{"text": token.text, "lemma": token.lemma_} for token in doc]
            logging.info(f"Lemmatisierung: {len(lemmas_results)} Lemmata generiert.")
        except Exception as e:
            logging.error(f"Fehler bei der Lemmatisierung: {e}")
            lemmas_results = []
    elif LEMMATIZATION_ENABLED:
        logging.warning("Lemmatisierung ist in der Konfiguration aktiviert, aber das spaCy-Modell konnte nicht geladen werden. Keine Lemmata generiert.")

    return {
        "entities": ner_results,
        "lemmas": lemmas_results
    }

# Um diesen Dienst zu starten:
# uvicorn nlp_service:app --host 127.0.0.1 --port 8002 --reload