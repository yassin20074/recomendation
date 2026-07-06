FROM python:3.10-slim AS builder

WORKDIR /app

 
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

 
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

 
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')"

 
FROM python:3.10-slim AS runner

WORKDIR /app

 
COPY --from=builder /root/.local /root/.local
COPY --from=builder /root/.cache/huggingface /root/.cache/huggingface

 
ENV PATH=/root/.local/bin:$PATH
ENV PYTHONUNBUFFERED=1

 
COPY main.py app.py ./
COPY pleace.csv ./C:/Users/matrg/Desktop/recomendation/pleace.csv
 
COPY pleace.csv . 

 
EXPOSE 8000

 
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]