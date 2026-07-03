# ==============================================================
# app/models.py
# Pydantic request / response schemas
# ==============================================================

from pydantic import BaseModel
from typing import Optional

# ── /ingest ───────────────────────────────────────────────────
class IngestResponse(BaseModel):
    job_id  : str
    doc_id  : str
    status  : str
    message : str

# ── /status ───────────────────────────────────────────────────
class StatusResponse(BaseModel):
    job_id   : str
    doc_id   : Optional[str] = None
    status   : str                    # started | processing | done | failed
    progress : Optional[str] = None
    error    : Optional[str] = None

# ── /query request ────────────────────────────────────────────
class QueryRequest(BaseModel):
    question      : str
    doc_id        : str
    base_image_url: Optional[str] = "http://localhost:8000/images"

# ── /query response ───────────────────────────────────────────
class Source(BaseModel):
    parent_id  : str
    page_num   : Optional[int]
    score      : float
    snippet    : str
    image_urls : list[str]   # full URLs to /images/... endpoint

class QueryResponse(BaseModel):
    answer  : str
    sources : list[Source]
