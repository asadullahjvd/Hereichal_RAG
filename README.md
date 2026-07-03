---
title: AsadNav
emoji: 📄
colorFrom: green
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
---

# AsadNav — Multimodal Hierarchical RAG

Ask questions over a PDF (text, tables, and charts) and get grounded answers, backed by a
parent/child chunking pipeline, Chroma vector search, and Gemini for captioning + answering.

## Project structure

```
.
├── app/
│   ├── __init__.py
│   ├── main.py         # FastAPI routes
│   ├── models.py        # Pydantic request/response schemas
│   ├── storage.py       # SQLite models (documents, parents, children)
│   ├── job_store.py     # Ingestion job status tracking
│   ├── ingest.py        # Parse → caption → chunk → embed pipeline
│   └── retrieval.py     # Search → prompt build → answer generation
├── frontend.html         # Chat UI, served at /
├── requirements.txt
├── Dockerfile
├── .dockerignore
└── storage/              # Created automatically — uploads, images, SQLite, Chroma index
```

## Environment variables

| Variable | Required | Description |
|---|---|---|
| `GOOGLE_API_KEY` | Yes | Gemini API key, used for image captioning and answer generation |

Never commit a real key. Pass it in at runtime (see below).

---

## Run locally (no Docker)

```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt

export GOOGLE_API_KEY="your-real-key-here"   # Windows: set GOOGLE_API_KEY=...

uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Then open:
- **App UI** — http://127.0.0.1:8000/
- **Swagger / try-it-out** — http://127.0.0.1:8000/docs

---

## Run with Docker

### Build the image
```bash
docker build -t asadnav .
```

### Run the container
```bash
docker run -d \
  --name asadnav \
  -p 7860:7860 \
  -e GOOGLE_API_KEY="your-real-key-here" \
  -v asadnav_storage:/app/storage \
  asadnav
```

- `-p 7860:7860` — exposes the API/UI on your host at `http://localhost:7860` (7860 is used
  instead of 8000 so the image matches Hugging Face Spaces' expected port — see below)
- `-v asadnav_storage:/app/storage` — persists uploaded PDFs, extracted images, SQLite DB,
  and the Chroma index across container restarts. Without this, everything is wiped when
  the container is removed.

### Check it's healthy
```bash
docker ps                       # STATUS should show "healthy" after ~30s
curl http://localhost:7860/health
```

### View logs (ingestion progress prints here)
```bash
docker logs -f asadnav
```

### Stop / remove
```bash
docker stop asadnav && docker rm asadnav
```

---

## Docker Compose (optional)

```yaml
services:
  asadnav:
    build: .
    ports:
      - "8000:8000"
    environment:
      - GOOGLE_API_KEY=${GOOGLE_API_KEY}
    volumes:
      - asadnav_storage:/app/storage
    restart: unless-stopped

volumes:
  asadnav_storage:
```

Run with:
```bash
export GOOGLE_API_KEY="your-real-key-here"
docker compose up -d --build
```

---

## API endpoints

| Method | Path | Purpose |
|---|---|---|
| GET | `/` | Serves the chat UI (`frontend.html`) |
| GET | `/health` | Health check |
| POST | `/ingest` | Upload a PDF (multipart, field `file`) → `{job_id, doc_id}` |
| GET | `/status/{job_id}` | Poll ingestion progress |
| POST | `/query` | `{question, doc_id}` → `{answer, sources}` |
| GET | `/documents` | List all indexed documents |
| GET | `/documents/{doc_id}` | Get one document's metadata |
| DELETE | `/documents/{doc_id}` | Remove a document (e.g. a stuck/failed row) |
| GET | `/images/{doc_id}/{filename}` | Serve an extracted table/chart image |

## Notes

- PDF parsing uses `unstructured`'s `hi_res` strategy, which needs `poppler-utils` and
  `tesseract-ocr` on the system — already installed in the Docker image. If running
  locally without Docker, install them yourself (`apt install poppler-utils tesseract-ocr`
  on Debian/Ubuntu, `brew install poppler tesseract` on macOS).
- Max upload size is 50 MB per PDF.
- Re-uploading a PDF that's already indexed (matched by content hash) returns the
  existing document instead of reprocessing it.
