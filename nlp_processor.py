# --- nlp_processor.py ---
# This file contains the core NLP processing logic using spaCy.

import spacy
from spacy.tokens import Doc
from typing import List, Dict, Any

class NLPProcessor:
    """
    A class to handle NLP processing using a spaCy model.
    It's designed to be modular and allow for custom pipeline components.
    """
    def __init__(self, model_name: str = "de_dep_news_trf"):
        """
        Loads the spaCy model upon initialization.
        
        Args:
            model_name (str): The name of the spaCy model to load.
        """
        try:
            self.nlp = spacy.load(model_name)
            print(f"Successfully loaded spaCy model '{model_name}'.")
        except OSError:
            print(f"Error: Model '{model_name}' not found. Please run 'python -m spacy download {model_name}'")
            self.nlp = None

    def add_pipe(self, component_name: str, **kwargs):
        """Adds a new component to the spaCy pipeline."""
        if self.nlp:
            self.nlp.add_pipe(component_name, **kwargs)
            print(f"Added '{component_name}' to the pipeline. New pipeline: {self.nlp.pipe_names}")

    def process_text(self, text: str) -> Doc:
        """
        Processes a given text with the spaCy pipeline.

        Args:
            text (str): The text to process.

        Returns:
            Doc: The processed spaCy Doc object.
        """
        if not self.nlp:
            raise ValueError("spaCy model is not loaded.")
        return self.nlp(text)

    def extract_entities(self, doc: Doc) -> List[Dict[str, Any]]:
        """
        Extracts named entities from a processed Doc object.

        Args:
            doc (Doc): A spaCy Doc object.

        Returns:
            List[Dict[str, Any]]: A list of entities, each as a dictionary.
        """
        entities = []
        for ent in doc.ents:
            entities.append({
                "text": ent.text,
                "start_char": ent.start_char,
                "end_char": ent.end_char,
                "label": ent.label_
            })
        return entities

    def extract_lemmas(self, doc: Doc) -> List[Dict[str, str]]:
        """
        Extracts tokens and their lemmas from a processed Doc object.

        Args:
            doc (Doc): A spaCy Doc object.

        Returns:
            List[Dict[str, str]]: A list of tokens with their text and lemma.
        """
        return [{"text": token.text, "lemma": token.lemma_} for token in doc if not token.is_punct and not token.is_space]