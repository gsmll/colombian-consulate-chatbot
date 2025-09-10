# Multi-stage Dockerfile for consulate chatbot
# Stage 1: base runtime
FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=5000

# System deps (minimal)
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install dependencies first to leverage layer caching
COPY requirements.txt ./
RUN pip install --upgrade pip setuptools wheel \
    && pip install -r requirements.txt

# Add application source
COPY consulate_chatbot.py ./
COPY consulate_information.pdf ./
COPY wsgi.py ./

# Create non-root user
RUN useradd -m appuser \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 5000

# Gunicorn configuration: 2 workers adequate for lightweight I/O bound app; tweak via env if needed
ENV GUNICORN_CMD_ARGS="--workers=2 --threads=4 --timeout=60 --bind 0.0.0.0:${PORT:-5000}"

CMD ["sh","-c","gunicorn wsgi:app $GUNICORN_CMD_ARGS"]