# syntax=docker/dockerfile:1
FROM python:3.11-slim

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl && \
    rm -rf /var/lib/apt/lists/*

# Workdir
WORKDIR /app

# Copy minimal first (better caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy your app
COPY . .

# Expose DO App Platform default port
ENV PORT=8080
ENV HOST=0.0.0.0

# Uvicorn entrypoint; App Platform expects listening on $PORT
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]