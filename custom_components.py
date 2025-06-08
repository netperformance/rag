# --- custom_components.py ---
# FINAL VERSION: Using a different, reliable German NER model to resolve all issues.

from spacy.language import Language
from spacy.tokens import Doc, Span
from transformers import pipeline, Pipeline
import logging

# --- Global variable to hold the NER pipeline ---
# This ensures the model is loaded only once when the module is imported.
ner_pipeline: Pipeline = None

def initialize_ner_pipeline():
    """Initializes the Hugging Face NER pipeline."""
    global ner_pipeline
    if ner_pipeline is None:
        logging.info("Initializing NEW Hugging Face NER pipeline for German...")
        
        # ####################################################################
        # ##### WE ARE SWITCHING TO A NEW, ROBUST GERMAN NER MODEL #####
        # ####################################################################
        model_name = "oliverguhr/german-bert-ner"
        
        try:
            # Initialize the pipeline for Named Entity Recognition
            ner_pipeline = pipeline(
                "ner",
                model=model_name,
                tokenizer=model_name,
                grouped_entities=True
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
        span = doc.char_span(start, end, label=ent['entity_group'])
        if span is not None:
            spacy_ents.append(span)
            
    # Overwrite the doc.ents with our new entities
    doc.ents = spacy_ents
    logging.info(f"--- Custom NER component finished. Found {len(spacy_ents)} entities. ---")
    return doc