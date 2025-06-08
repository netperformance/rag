from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification, Pipeline
import spacy
import spacy.cli # Zum Herunterladen von spaCy-Modellen
import logging
import json
import os
from typing import Dict, Any, Optional

app = FastAPI()

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# --- Konfiguration laden ---
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
        "nlp_service": "http://127.0.0.1:8002/process/"
    }
}

config = DEFAULT_CONFIG

try:
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
except FileNotFoundError:
    logging.warning(f"Konfigurationsdatei '{CONFIG_FILE}' nicht gefunden. Verwende Standardwerte.")
except json.JSONDecodeError:
    logging.error(f"Fehler beim Parsen der Konfigurationsdatei '{CONFIG_FILE}'. Überprüfen Sie das JSON-Format. Verwende Standardwerte.")
except Exception as e:
    logging.error(f"Unerwarteter Fehler beim Laden der Konfiguration: {e}. Verwende Standardwerte.")

# Konfigurationswerte aus dem geladenen/Standard-Konfig-Objekt extrahieren
NER_MODEL_NAME_DE = config["nlp_model_config"]["ner_model_name_de"]
LEMMATIZATION_MODEL_NAME_DE = config["nlp_model_config"]["lemmatization_model_name_de"]
NER_MODEL_NAME_EN = config["nlp_model_config"]["ner_model_name_en"]
LEMMATIZATION_MODEL_NAME_EN = config["nlp_model_config"]["lemmatization_model_name_en"]
LEMMATIZATION_ENABLED = config["nlp_model_config"]["lemmatization_enabled"]

# Globale Variablen für NER-Pipelines und spaCy-Modelle
ner_pipeline_de: Optional[Pipeline] = None
nlp_lemmatizer_de: Optional[spacy.language.Language] = None
ner_pipeline_en: Optional[Pipeline] = None
nlp_lemmatizer_en: Optional[spacy.language.Language] = None

def load_models_for_language(lang: str):
    """Lädt die entsprechenden NLP-Modelle für die angegebene Sprache."""
    global ner_pipeline_de, nlp_lemmatizer_de, ner_pipeline_en, nlp_lemmatizer_en

    if lang == "de":
        if ner_pipeline_de is None:
            try:
                logging.info(f"Lade deutsches NER-Modell: {NER_MODEL_NAME_DE}...")
                tokenizer_de = AutoTokenizer.from_pretrained(NER_MODEL_NAME_DE)
                model_de = AutoModelForTokenClassification.from_pretrained(NER_MODEL_NAME_DE)
                ner_pipeline_de = pipeline(
                    "ner",
                    model=model_de,
                    tokenizer=tokenizer_de,
                    aggregation_strategy="simple"
                )
                logging.info("Deutsches NER-Modell erfolgreich geladen.")
            except Exception as e:
                logging.error(f"Fehler beim Laden des deutschen NER-Modells '{NER_MODEL_NAME_DE}': {e}")

        if LEMMATIZATION_ENABLED and nlp_lemmatizer_de is None:
            try:
                logging.info(f"Lade deutsches Lemmatisierungsmodell: {LEMMATIZATION_MODEL_NAME_DE} (spaCy)...")
                try:
                    nlp_lemmatizer_de = spacy.load(LEMMATIZATION_MODEL_NAME_DE)
                except OSError:
                    logging.warning(f"spaCy Modell '{LEMMATIZATION_MODEL_NAME_DE}' nicht gefunden. Versuche es herunterzuladen...")
                    spacy.cli.download(LEMMATIZATION_MODEL_NAME_DE)
                    nlp_lemmatizer_de = spacy.load(LEMMATIZATION_MODEL_NAME_DE)
                logging.info("Deutsches Lemmatisierungsmodell erfolgreich geladen.")
            except Exception as e:
                logging.error(f"Fehler beim Laden/Herunterladen des deutschen Lemmatisierungsmodells '{LEMMATIZATION_MODEL_NAME_DE}': {e}")

    elif lang == "en":
        if ner_pipeline_en is None:
            try:
                logging.info(f"Lade englisches NER-Modell: {NER_MODEL_NAME_EN}...")
                tokenizer_en = AutoTokenizer.from_pretrained(NER_MODEL_NAME_EN)
                model_en = AutoModelForTokenClassification.from_pretrained(NER_MODEL_NAME_EN)
                ner_pipeline_en = pipeline(
                    "ner",
                    model=model_en,
                    tokenizer=tokenizer_en,
                    aggregation_strategy="simple"
                )
                logging.info("Englisches NER-Modell erfolgreich geladen.")
            except Exception as e:
                logging.error(f"Fehler beim Laden des englischen NER-Modells '{NER_MODEL_NAME_EN}': {e}")

        if LEMMATIZATION_ENABLED and nlp_lemmatizer_en is None:
            try:
                logging.info(f"Lade englisches Lemmatisierungsmodell: {LEMMATIZATION_MODEL_NAME_EN} (spaCy)...")
                try:
                    nlp_lemmatizer_en = spacy.load(LEMMATIZATION_MODEL_NAME_EN)
                except OSError:
                    logging.warning(f"spaCy Modell '{LEMMATIZATION_MODEL_NAME_EN}' nicht gefunden. Versuche es herunterzuladen...")
                    spacy.cli.download(LEMMATIZATION_MODEL_NAME_EN)
                    nlp_lemmatizer_en = spacy.load(LEMMATIZATION_MODEL_NAME_EN)
                logging.info("Englisches Lemmatisierungsmodell erfolgreich geladen.")
            except Exception as e:
                logging.error(f"Fehler beim Laden/Herunterladen des englischen Lemmatisierungsmodells '{LEMMATIZATION_MODEL_NAME_EN}': {e}")

# Lade Modelle für beide Sprachen beim Start des Dienstes vorab
load_models_for_language("de")
load_models_for_language("en")


class TextProcessPayload(BaseModel):
    text: str
    language: Optional[str] = None # Neues Feld für die erkannte Sprache

@app.post("/process/")
async def process_text(payload: TextProcessPayload):
    text = payload.text
    language = payload.language if payload.language else "de" # Standard auf Deutsch, falls nicht angegeben

    current_ner_pipeline = None
    current_lemmatizer = None

    if language == "en":
        current_ner_pipeline = ner_pipeline_en
        current_lemmatizer = nlp_lemmatizer_en
        logging.info(f"Verwende englische Modelle für die Verarbeitung (Länge: {len(text)} Zeichen).")
    else: # Default or "de"
        current_ner_pipeline = ner_pipeline_de
        current_lemmatizer = nlp_lemmatizer_de
        logging.info(f"Verwende deutsche Modelle für die Verarbeitung (Länge: {len(text)} Zeichen).")

    if current_ner_pipeline is None:
        raise HTTPException(status_code=503, detail=f"NER-Modell für Sprache '{language}' konnte nicht geladen werden, Dienst nicht verfügbar.")

    # --- NER-Verarbeitung ---
    ner_results = []
    try:
        entities = current_ner_pipeline(text)
        ner_results = [{"text": ent.get('word', ''), "label": ent.get('entity_group', '')} for ent in entities]
        logging.info(f"NER: {len(ner_results)} Entitäten gefunden für Sprache '{language}'.")
    except Exception as e:
        logging.error(f"Fehler bei der NER-Verarbeitung für Sprache '{language}': {e}")
        ner_results = []

    # --- Lemmatisierung ---
    lemmas_results = []
    if LEMMATIZATION_ENABLED and current_lemmatizer:
        try:
            doc = current_lemmatizer(text)
            lemmas_results = [{"text": token.text, "lemma": token.lemma_} for token in doc]
            logging.info(f"Lemmatisierung: {len(lemmas_results)} Lemmata generiert für Sprache '{language}'.")
        except Exception as e:
            logging.error(f"Fehler bei der Lemmatisierung für Sprache '{language}': {e}")
            lemmas_results = []
    elif LEMMATIZATION_ENABLED:
        logging.warning(f"Lemmatisierung ist aktiviert, aber das spaCy-Modell für Sprache '{language}' konnte nicht geladen werden. Keine Lemmata generiert.")

    return {
        "entities": ner_results,
        "lemmas": lemmas_results,
        "processed_language": language # Fügen Sie die tatsächlich verwendete Sprache hinzu
    }

# Um diesen Dienst zu starten:
# uvicorn nlp_service:app --host 127.0.0.1 --port 8002 --reload
