FROM python:3.11-slim

WORKDIR /app

# System deps needed by faiss-cpu / sentence-transformers wheels at runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy only what the app needs to run
COPY backend/ backend/
COPY data/ data/
COPY demo/ demo/
COPY indexes/ indexes/

EXPOSE 8000

CMD ["uvicorn", "backend.app.api.server:app", "--host", "0.0.0.0", "--port", "8000"]