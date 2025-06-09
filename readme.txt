RAG Pipeline - Setup und Ausführung

Dieses Dokument beschreibt die Einrichtung und den Betrieb einer Retrieval-Augmented Generation (RAG) Pipeline als Proof-of-Concept.

Die Anwendung basiert auf einer Microservice-Architektur, um Modularität und Skalierbarkeit zu gewährleisten. Ein zentrales Orchestrator-Skript (start_embedding.py) steuert eine Reihe von spezialisierten Diensten, die jeweils über eine FastAPI-Schnittstelle angesprochen werden.

Schritt 0: Voraussetzungen
Bevor die Python-Umgebung eingerichtet wird, müssen mehrere externe Programme auf dem System installiert und konfiguriert werden.

Tesseract OCR
Tesseract wird von der unstructured-Bibliothek für die Texterkennung (OCR) aus Bildern innerhalb der PDF-Dateien benötigt.
- Ladene das Installationsprogramm für Windows von der offiziellen Tesseract-Seite bei der UB Mannheim herunter.
- Führe das Installationsprogramm aus und installiere die benötigten Sprachpakete z.B. Deutsch und Englisch.
- Setze die Umgebungsvariablen: a) PATH = C:\folder\Tesseract-OCR & b) TESSDATA_PREFIX = C:\folder\Tesseract-OCR

Poppler
Poppler wird von unstructured für bestimmte PDF-Verarbeitungsaufgaben benötigt, insbesondere um PDF-Seiten in Bilder für die Layout-Analyse umzuwandeln.
- Installiere Poppler und setze die Umgebungsvariablen

Ollama und KI-Modell
- Installiere Ollama mit z.B. DeepSeek als LLM. Anleitung: https://aaron.de/index.php/2025/02/01/ollama-inkl-modell-mit-nvidia-gpu-unterstuetzung-unter-docker-und-win-11-ausfuehren-openwebui/


Schritt 1: Python-Umgebung und Abhängigkeiten

Virtuelle Umgebung erstellen:
- python -m venv venv

Virtuelle Umgebung aktivieren:
- .\venv\Scripts\activate

Python-Bibliotheken installieren:
- pip install -r requirements.txt

SpaCy-Modelle herunterladen:
- python -m spacy download de_core_news_sm
- python -m spacy download en_core_web_sm


Schritt 2: Starten der Microservices

Für jeden der folgenden Befehle muss ein eigenes, neues Terminal-Fenster geöffnet werden. In jedem dieser Fenster muss die virtuelle Umgebung (.\venv\Scripts\activate) aktiv sein. Diese fünf Dienste müssen während des Betriebs der gesamten Anwendung im Hintergrund laufen.

Terminal 1 - Language Service:
- uvicorn language_detection_service:app --reload --port 8000

Terminal 2 - Structuring Service:
- uvicorn structuring_service:app --reload --port 8001

Terminal 3 - NLP Service:
- uvicorn nlp_service:app --reload --port 8002

Terminal 4 - DeepSeek Service:
- uvicorn deepseek_enrichment_service:app --reload --port 8003

Terminal 5 - Embedding Service:
- uvicorn embedding_service:app --reload --port 8004


Schritt 3: Datenpipeline ausführen (Vektoren erstellen)
Dieser Schritt liest die PDF-Datei (test_oc.pdf), verarbeitet sie durch alle laufenden Dienste und füllt die Vektor-Datenbank (ChromaDB).
Öffne ein weiteres, sechstes Terminal, aktiviere die virtuelle Umgebung und führen Sie das Orchestrator-Skript aus:
- python start_embedding.py


Schritt 4: Chatbot starten und Fragen stellen
Nachdem die Datenbank gefüllt ist, kann die interaktive Chat-Anwendung gestartet werden.
- python chatbot.py
