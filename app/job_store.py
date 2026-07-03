# ==============================================================
# app/job_store.py
# Concern 5 — job state persisted to SQLite (survives server restart)
# ==============================================================

from app.storage import get_session, JobModel
from datetime import datetime


def create_job(job_id: str, doc_id: str):
    session = get_session()
    session.add(JobModel(
        job_id     = job_id,
        doc_id     = doc_id,
        status     = "started",
        progress   = "Queued...",
        error      = None,
        created_at = datetime.utcnow().isoformat(),
    ))
    session.commit()
    session.close()


def update_job(job_id: str, **kwargs):
    session = get_session()
    job = session.query(JobModel).filter_by(job_id=job_id).first()
    if job:
        for k, v in kwargs.items():
            setattr(job, k, v)
        session.commit()
    session.close()


def get_job(job_id: str) -> dict | None:
    session = get_session()
    job = session.query(JobModel).filter_by(job_id=job_id).first()
    session.close()
    if not job:
        return None
    return {
        "job_id"  : job.job_id,
        "doc_id"  : job.doc_id,
        "status"  : job.status,
        "progress": job.progress,
        "error"   : job.error,
    }
