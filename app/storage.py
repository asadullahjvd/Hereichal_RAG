# ==============================================================
# app/storage.py
# SQLAlchemy models + session factory
# ==============================================================

from sqlalchemy import create_engine, Column, String, Integer, Text, JSON
from sqlalchemy.orm import declarative_base, sessionmaker

DB_PATH = "storage/docs.db"
Base    = declarative_base()
engine  = create_engine(f"sqlite:///{DB_PATH}", echo=False, connect_args={"check_same_thread": False})
Session = sessionmaker(bind=engine)


def get_session():
    return Session()


def init_db():
    Base.metadata.create_all(engine)
    print(f"DB initialised at {DB_PATH}")


# ── Document registry ─────────────────────────────────────────
class DocModel(Base):
    __tablename__ = "docs"
    doc_id     = Column(String, primary_key=True)
    filename   = Column(String)
    hash       = Column(String, unique=True)   # Concern 1 — duplicate detection
    status     = Column(String)                # processing | done | failed
    created_at = Column(String)


# ── Job tracking ──────────────────────────────────────────────
# Concern 5 — persisted to SQLite, survives server restart
class JobModel(Base):
    __tablename__ = "jobs"
    job_id     = Column(String, primary_key=True)
    doc_id     = Column(String)
    status     = Column(String)   # started | processing | done | failed
    progress   = Column(String)
    error      = Column(Text)
    created_at = Column(String)


# ── Parents ───────────────────────────────────────────────────
class ParentModel(Base):
    __tablename__ = "parents"
    parent_id   = Column(String,  primary_key=True)
    doc_id      = Column(String,  nullable=False)
    text        = Column(Text,    nullable=False)
    page_num    = Column(Integer)
    token_count = Column(Integer)
    image_refs  = Column(JSON)    # list of relative image paths


# ── Children ──────────────────────────────────────────────────
class ChildModel(Base):
    __tablename__ = "children"
    child_id    = Column(String, primary_key=True)
    parent_id   = Column(String, nullable=False)
    doc_id      = Column(String, nullable=False)
    text        = Column(Text,   nullable=False)
    page_num    = Column(Integer)
    token_count = Column(Integer)
    is_atomic   = Column(String)  # "true" | "false"
