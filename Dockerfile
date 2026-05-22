FROM python:3.11-slim

# System libs needed by PyMuPDF, Pillow, OpenCV (used by easyocr)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download spaCy NER model
RUN python -m spacy download en_core_web_sm

# Pre-download EasyOCR English model so first request isn't slow
RUN python -c "import easyocr; easyocr.Reader(['en'], gpu=False, verbose=False)"

COPY . .

RUN mkdir -p data/uploads data/faiss_index data/chunks data/bm25 data/graph
