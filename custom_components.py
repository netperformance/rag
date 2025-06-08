# --- custom_components.py ---
# FINAL VERSION: Using a different, reliable German NER model to resolve all issues.

from spacy.language import Language
from spacy.tokens import Doc, Span
from transformers import pipeline, Pipeline
import logging
import json
import os

# --- Global variable to hold the NER pipeline ---
# This ensures the model is loaded only once when the module is imported.
ner_pipeline: Pipeline = None

# --- Configuration for this module ---
CONFIG_FILE = "config.json"
# Default model name if not found in config.json
DEFAULT_NER_MODEL_NAME = "oliverguhr/german-bert-ner"

# Load configuration at module import time
# This ensures config is available when initialize_ner_pipeline is called implicitly
_ner_model_name_from_config = None
try:
    with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
        loaded_config = json.load(f)
        # Safely get the NER model name from the loaded config
        _ner_model_name_from_config = loaded_config.get("nlp_model_config", {}).get("ner_model_name")
    if _ner_model_name_from_config:
        logging.info(f"NER model name '{_ner_model_name_from_config}' geladen aus '{CONFIG_FILE}'.")
    else:
        logging.warning(f"NER model name nicht in '{CONFIG_FILE}' gefunden unter 'nlp_model_config.ner_model_name'. Verwende Standardwert.")
        _ner_model_name_from_config = DEFAULT_NER_MODEL_NAME
except FileNotFoundError:
    logging.warning(f"Konfigurationsdatei '{CONFIG_FILE}' nicht gefunden für custom_components. Verwende Standard NER-Modell '{DEFAULT_NER_MODEL_NAME}'.")
    _ner_model_name_from_config = DEFAULT_NER_MODEL_NAME
except json.JSONDecodeError:
    logging.error(f"Fehler beim Parsen der Konfigurationsdatei '{CONFIG_FILE}' in custom_components. Überprüfen Sie das JSON-Format. Verwende Standard NER-Modell '{DEFAULT_NER_MODEL_NAME}'.")
    _ner_model_name_from_config = DEFAULT_NER_MODEL_NAME
except Exception as e:
    logging.error(f"Unerwarteter Fehler beim Laden der Konfiguration für custom_components: {e}. Verwende Standard NER-Modell '{DEFAULT_NER_MODEL_NAME}'.")
    _ner_model_name_from_config = DEFAULT_NER_MODEL_NAME

# Final model name to use
MODEL_NAME_TO_USE = _ner_model_name_from_config


def initialize_ner_pipeline():
    """Initializes the Hugging Face NER pipeline."""
    global ner_pipeline
    if ner_pipeline is None:
        logging.info("Initializing NEW Hugging Face NER pipeline for German...")
        
        # ####################################################################
        # ##### USING MODEL NAME FROM CONFIGURATION FILE #####
        # ####################################################################
        model_name = MODEL_NAME_TO_USE # <-- Using the configurable model name
        
        try:
            # Initialize the pipeline for Named Entity Recognition
            ner_pipeline = pipeline(
                "ner",
                model=model_name,
                tokenizer=model_name,
                grouped_entities=True # Important for getting full entities like "New York"
            )
            logging.info(f"Hugging Face NER pipeline initialized successfully with model: {model_name}")
        except Exception as e:
            logging.error(f"Failed to initialize Hugging Face pipeline with model '{model_name}'. Error: {e}", exc_info=True)
            # Ensure pipeline is not considered partially initialized
            ner_pipeline = "failed"


# This decorator registers the function as a spaCy component.
@Language.component("huggingface_ner_replacer")
def custom_ner_component(doc: Doc) -> Doc:
    """
    This component replaces spaCy's entity recognition with a Hugging Face model.
    """
    # Ensure the pipeline is initialized
    # We check for the string "failed" in case initialization failed before.
    if ner_pipeline is None:
        initialize_ner_pipeline()

    if ner_pipeline == "failed" or ner_pipeline is None:
        logging.warning("Skipping custom NER component because the Hugging Face pipeline failed to initialize.")
        return doc

    # Get entities from the Hugging Face pipeline
    hf_entities = ner_pipeline(doc.text)
    
    # Convert Hugging Face entities to spaCy Spans
    spacy_ents = []
    for ent in hf_entities:
        # The start and end values are character indices
        start = ent['start']
        end = ent['end']
        # Create a spaCy Span from the character indices
        # Ensure the label is correctly mapped to spaCy's expectations if necessary
        # The 'entity_group' from the HF pipeline is typically the entity type (e.g., 'PER', 'ORG')
        span = doc.char_span(start, end, label=ent['entity_group'])
        if span is not None:
            spacy_ents.append(span)
            
    # Overwrite the doc.ents with our new entities
    doc.ents = spacy_ents
    logging.info(f"--- Custom NER component finished. Found {len(spacy_ents)} entities. ---")
    return doc