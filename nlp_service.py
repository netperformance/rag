# --- nlp_service.py ---
# A dedicated FastAPI service for NLP tasks.
# uvicorn nlp_service:app --reload --port 8002

from fastapi import FastAPI
from pydantic import BaseModel
import logging

from nlp_processor import NLPProcessor
from custom_components import custom_ner_component # Import the component function

# --- Data Models for API ---
class ProcessRequest(BaseModel):
    text: str

# --- FastAPI Application Setup ---
app = FastAPI(title="NLP Service")
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Initialize the NLP Processor ---
# This happens once when the server starts.
processor = NLPProcessor(model_name="de_dep_news_trf")

# --- Add your custom component to the pipeline ---
# This demonstrates the modular architecture. We are adding our placeholder component.
# We place it after spaCy's default "ner" component.
processor.add_pipe("huggingface_ner_replacer", last=True)


# --- API Endpoint ---
@app.post("/process/")
def process_text_endpoint(request: ProcessRequest):
    """
    Receives text, processes it with the NLP pipeline,
    and returns extracted entities and lemmas.
    """
    if not processor.nlp:
        return {"error": "NLP model not loaded."}

    logging.info(f"Processing text starting with: '{request.text[:50]}...'")
    
    # Process the text to get a spaCy Doc object
    doc = processor.process_text(request.text)
    
    # Use the processor's methods to extract information
    entities = processor.extract_entities(doc)
    lemmas = processor.extract_lemmas(doc)
    
    return {
        "entities": entities,
        "lemmas": lemmas
    }