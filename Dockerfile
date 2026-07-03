# ==============================================================
# Dockerfile — Multimodal Hierarchical RAG (FastAPI)
# Configured for Hugging Face Spaces (Docker SDK):
#   - listens on port 7860 (HF's expected default)
#   - runs as a non-root user with a writable /app (HF requirement)
# ==============================================================

FROM python:3.11-slim

# ── System dependencies ────────────────────────────────────────
# poppler-utils   → pdf2image / unstructured PDF page rendering
# tesseract-ocr   → pytesseract OCR fallback for scanned pages
# libgl1 / glib   → OpenCV, used by unstructured's hi_res layout model
# build-essential → some pip packages compile native extensions
RUN apt-get update && apt-get install -y --no-install-recommends \
    poppler-utils \
    tesseract-ocr \
    libgl1 \
    libglib2.0-0 \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── Python dependencies (cached separately from app code) ──────
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# ── Application code ─────────────────────────────────────────
COPY app/ ./app/
COPY frontend.html .

# ── Runtime storage + non-root user (HF Spaces runs containers as
#    a non-root user; /app must be writable by it) ─────────────
RUN mkdir -p storage/uploads storage/images storage/json storage/chroma \
    && useradd -m -u 1000 appuser \
    && chown -R appuser:appuser /app

USER appuser

ENV PYTHONUNBUFFERED=1

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:7860/health')" || exit 1

# GOOGLE_API_KEY is injected at runtime as a Space secret — never baked
# into the image.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]

