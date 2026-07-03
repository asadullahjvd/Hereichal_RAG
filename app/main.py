# ==============================================================
# app/main.py
# FastAPI application — Improved Version (corrected)
# ==============================================================

import os
import uuid
import asyncio
from concurrent.futures import ThreadPoolExecutor
from fastapi.responses import FileResponse

from fastapi import (
    FastAPI,
    UploadFile,
    File,
    HTTPException,
    BackgroundTasks,
)
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from app.models import (
    QueryRequest,
    QueryResponse,
    IngestResponse,
    StatusResponse,
)
from app.job_store import create_job, get_job
from app.storage import init_db, get_session, DocModel
from app.ingest import run_ingestion_pipeline
from app.retrieval import retrieve


# ==============================================================
# Create required directories BEFORE mounting static files
# ==============================================================

REQUIRED_DIRS = [
    "storage",
    "storage/uploads",
    "storage/images",
    "storage/json",
    "storage/chroma",
]

for directory in REQUIRED_DIRS:
    os.makedirs(directory, exist_ok=True)


# ==============================================================
# FastAPI App
# ==============================================================

app = FastAPI(
    title="Multimodal Document Navigator",
    description="Hierarchical RAG over enterprise documents",
    version="1.0.0",
)


# ==============================================================
# CORS
# ==============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",   # React
        "http://localhost:8501",   # Streamlit
        "http://127.0.0.1:5500",
        "http://localhost:5500",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==============================================================
# Static Images
# ==============================================================

app.mount(
    "/images",
    StaticFiles(directory="storage/images"),
    name="images",
)


# ==============================================================
# Thread Pool
# ==============================================================

executor = ThreadPoolExecutor(max_workers=2)


# ==============================================================
# Startup Event
# ==============================================================

@app.on_event("startup")
def startup():
    """
    Runs once when FastAPI starts.
    """

    init_db()

    print("=" * 60)
    print("Server Started Successfully")
    print("=" * 60)


# ==============================================================
# POST /ingest
# ==============================================================

MAX_FILE_SIZE_MB = 50


@app.post("/ingest", response_model=IngestResponse)
async def ingest(
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = BackgroundTasks(),
):

    contents = await file.read()

    if len(contents) > MAX_FILE_SIZE_MB * 1024 * 1024:
        raise HTTPException(
            status_code=400,
            detail=f"File exceeds {MAX_FILE_SIZE_MB} MB."
        )

    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported."
        )

    import hashlib

    pdf_hash = hashlib.md5(contents).hexdigest()

    session = get_session()

    existing = session.query(DocModel).filter_by(hash=pdf_hash).first()

    session.close()

    if existing:
        # NOTE: this returns "already_indexed" regardless of the existing
        # row's status (processing / done / failed). The frontend is
        # responsible for checking existing.status before treating this
        # as ready-to-query — see frontend.html's already_indexed handler.
        return IngestResponse(
            job_id=existing.doc_id,
            doc_id=existing.doc_id,
            status="already_indexed",
            message="This PDF has already been indexed.",
        )

    job_id = str(uuid.uuid4())
    doc_id = str(uuid.uuid4())

    pdf_path = f"storage/uploads/{doc_id}.pdf"

    with open(pdf_path, "wb") as f:
        f.write(contents)

    create_job(job_id, doc_id)

    # FIX: wrap in a proper async function so run_in_executor is awaited
    # on the loop that owns it (BackgroundTasks runs sync callables in a
    # worker thread, which is NOT safe for scheduling loop.run_in_executor
    # directly). This also ensures any exception is actually logged
    # instead of vanishing silently.
    def run_pipeline_wrapper():
     try:
        run_ingestion_pipeline(
            job_id,
            pdf_path,
            doc_id,
            pdf_hash,
            original_filename=file.filename,
        )
     except Exception as e:
        print("=" * 60)
        print("INGESTION FAILED")
        print(e)
        print("=" * 60)

    background_tasks.add_task(run_pipeline_wrapper)

    return IngestResponse(
        job_id=job_id,
        doc_id=doc_id,
        status="started",
        message="Ingestion started.",
    )


# ==============================================================
# GET /status/{job_id}
# ==============================================================

@app.get("/status/{job_id}", response_model=StatusResponse)
def status(job_id: str):

    job = get_job(job_id)

    if not job:
        raise HTTPException(
            status_code=404,
            detail="Job not found."
        )

    return StatusResponse(**job)


# ==============================================================
# POST /query
# ==============================================================

@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):

    session = get_session()

    doc = session.query(DocModel).filter_by(
        doc_id=req.doc_id
    ).first()

    session.close()

    if doc is None:
        raise HTTPException(
            status_code=404,
            detail="Document not found."
        )

    if doc.status != "done":
        raise HTTPException(
            status_code=400,
            detail=f"Document status is '{doc.status}'."
        )

    answer, sources = retrieve(
        query=req.question,
        doc_id=req.doc_id,
        base_image_url=req.base_image_url
        or "http://localhost:8000/images",
    )

    return QueryResponse(
        answer=answer,
        sources=sources,
    )


# ==============================================================
# GET /docs/{doc_id}
# ==============================================================

@app.get("/documents/{doc_id}")
def get_doc(doc_id: str):

    session = get_session()

    doc = session.query(DocModel).filter_by(
        doc_id=doc_id
    ).first()

    session.close()

    if doc is None:
        raise HTTPException(
            status_code=404,
            detail="Document not found."
        )

    return {
        "doc_id": doc.doc_id,
        "filename": doc.filename,
        "status": doc.status,
        "created_at": doc.created_at,
    }


# ==============================================================
# GET /docs
# ==============================================================

@app.get("/documents")
def list_docs():

    session = get_session()

    docs = session.query(DocModel).all()

    session.close()

    return [
        {
            "doc_id": d.doc_id,
            "filename": d.filename,
            "status": d.status,
            "created_at": d.created_at,
        }
        for d in docs
    ]


# ==============================================================
# DELETE /docs/{doc_id}
# ==============================================================
# NEW: lets you clear out a stale "processing"/"failed" row (e.g. from an
# interrupted run) without having to touch docs.db by hand every time.

@app.delete("/documents/{doc_id}")
def delete_doc(doc_id: str):

    session = get_session()

    doc = session.query(DocModel).filter_by(doc_id=doc_id).first()

    if doc is None:
        session.close()
        raise HTTPException(
            status_code=404,
            detail="Document not found."
        )

    session.delete(doc)
    session.commit()
    session.close()

    return {"doc_id": doc_id, "deleted": True}


# ==============================================================
# Health Check
# ==============================================================

@app.get("/health")
def health():
    return {
        "status": "ok"
    }


# ==============================================================
# Root Endpoint
# ==============================================================

@app.get("/", include_in_schema=False)
def root():
    return FileResponse("frontend.html")