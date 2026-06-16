# FunStay Concierge — webhook service image.
# Serves the FastAPI app (Dialogflow /webhook + browser /voice demo + /health).
# The Chroma knowledge base is built offline (scripts/ingest.py); mount it at
# runtime as a volume on /app/data/chroma. Without it, RAG answers gracefully
# decline (the anti-hallucination guardrail) — /health and routing still work.
FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Dependencies first for layer caching.
COPY requirements.txt ./
RUN pip install --upgrade pip && pip install -r requirements.txt

# Application code only (data/secrets/artefacts excluded via .dockerignore).
COPY app/ ./app/

# Run as an unprivileged user.
RUN useradd --create-home --uid 10001 appuser \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health', timeout=4).status==200 else 1)"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
