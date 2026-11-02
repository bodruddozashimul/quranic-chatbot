FROM python:3.11-slim

WORKDIR /app

# Keep ONNX Runtime single-purpose and predictable on a small CPU.
ENV OMP_NUM_THREADS=4 \
    HF_HOME=/opt/models \
    PYTHONUNBUFFERED=1

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY build_dataset.py ingest.py ./

# Bake the embedding model into the image so demo day needs no internet.
RUN python -c "from fastembed import TextEmbedding; \
    TextEmbedding(model_name='sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')"

CMD ["sh", "-c", "python build_dataset.py && python ingest.py"]
