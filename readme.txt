python -m spacy download de_core_news_sm
python -m spacy download en_core_web_sm

uvicorn language_detection_service:app --reload --port 8000
uvicorn structuring_service:app --reload --port 8001
uvicorn nlp_service:app --reload --port 8002
uvicorn deepseek_enrichment_service:app --reload --port 8003
uvicorn embedding_service:app --reload --port 8004