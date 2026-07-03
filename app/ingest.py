# ==============================================================
# app/ingest.py
# Updated to match actual notebook implementation:
# - el.category (not type(el).__name__)
# - Caption model: models/gemini-2.5-flash
# - LLM model: models/gemini-3.5-flash
# Concern 4 — full cleanup on failure (no orphaned partial data)
# Concern 7 — image paths stored as relative, not absolute
# Concern 8 — store the ORIGINAL uploaded filename, not the
#             on-disk saved-path basename (which is doc_id-based)
# ==============================================================

import os
import json
import base64
import dataclasses
import tiktoken
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional

import chromadb
import google.generativeai as genai
from PIL import Image as PILImage
from sentence_transformers import SentenceTransformer
from unstructured.partition.pdf import partition_pdf

from app.storage import get_session, ParentModel, ChildModel, DocModel
from app.job_store import update_job

# ── Config ────────────────────────────────────────────────────
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise RuntimeError(
        "GOOGLE_API_KEY environment variable is not set. "
        "Set it as a secret in your deployment environment."
    )
PARENT_MAX_TOKENS    = 1000
CHILD_TARGET_TOKENS  = 128
CHILD_OVERLAP_TOKENS = 20
ATOMIC_TYPES         = {"table", "chart", "image"}
EMBEDDING_MODEL      = "all-MiniLM-L6-v2"
CHROMA_PATH          = "storage/chroma"
IMAGE_BASE_DIR       = "storage/images"

genai.configure(api_key=GOOGLE_API_KEY)

# Caption model — matches notebook Cell 9
vlm = genai.GenerativeModel("models/gemini-2.5-flash")

embedder  = SentenceTransformer(EMBEDDING_MODEL)
tokenizer = tiktoken.get_encoding("cl100k_base")

# Element categories from unstructured — matches notebook Cell 11
TABLE_TYPES = {"Table"}
IMAGE_TYPES = {"Image", "Figure"}

CAPTION_PROMPT = """Describe this table or chart in detail for search retrieval purposes.
Include:
- All visible numbers, labels, and units exactly as shown
- The type of chart (bar, line, pie, etc.) if applicable
- Any trends, comparisons, or notable data points
- Row and column headers if it's a table, preserved as a markdown table
Be thorough — this description is the only way this image can be found by search."""


# ── Helpers ───────────────────────────────────────────────────
def count_tokens(text): return len(tokenizer.encode(text))
def tokenize(text):     return tokenizer.encode(text)
def detokenize(tokens): return tokenizer.decode(tokens)


@dataclass
class ElementSegment:
    text: str
    is_atomic: bool
    image_path: Optional[str] = None

@dataclass
class Parent:
    parent_id: str
    doc_id: str
    text: str
    page_num: int
    token_count: int
    image_refs: list
    segments: list = field(default_factory=list)

@dataclass
class Child:
    child_id: str
    parent_id: str
    doc_id: str
    text: str
    page_num: int
    token_count: int
    is_atomic: bool


# ── Phase 1 — Parse & Caption ─────────────────────────────────
def caption_image(image_path_abs):
    """Uses PIL Image directly — matches notebook Cell 9."""
    img = PILImage.open(image_path_abs)
    response = vlm.generate_content([CAPTION_PROMPT, img])
    return response.text


def parse_and_caption(pdf_path, doc_id, job_id):
    image_dir = os.path.join(IMAGE_BASE_DIR, doc_id)
    os.makedirs(image_dir, exist_ok=True)

    update_job(job_id, status="processing", progress="Phase 1: Extracting layout...")

    raw_elements = partition_pdf(
        filename=pdf_path,
        strategy="hi_res",
        extract_images_in_pdf=True,
        extract_image_block_types=["Table", "Image"],
        extract_image_block_output_dir=image_dir,
    )

    elements = []
    for i, el in enumerate(raw_elements):
        # Matches notebook Cell 11: uses el.category not type(el).__name__
        el_type_name = el.category
        page_num     = el.metadata.page_number if hasattr(el.metadata, "page_number") else None
        image_path   = getattr(el.metadata, "image_path", None)

        if el_type_name in TABLE_TYPES or el_type_name in IMAGE_TYPES:
            if image_path and os.path.exists(image_path):
                update_job(job_id, progress=f"Phase 1: Captioning element {i+1}/{len(raw_elements)}...")
                caption  = caption_image(image_path)
                el_class = "table" if el_type_name in TABLE_TYPES else "chart"
                # Concern 7 — store RELATIVE path only
                img_rel  = os.path.relpath(image_path, IMAGE_BASE_DIR)
                elements.append({
                    "el_id"     : f"el_{i}",
                    "type"      : el_class,
                    "text"      : caption,
                    "page_num"  : page_num,
                    "image_path": img_rel,
                })
            elif el.text and el.text.strip():
                elements.append({
                    "el_id": f"el_{i}", "type": "text",
                    "text": el.text, "page_num": page_num, "image_path": None
                })
        else:
            if el.text and el.text.strip():
                elements.append({
                    "el_id": f"el_{i}", "type": "text",
                    "text": el.text, "page_num": page_num, "image_path": None
                })

    # Save JSON for debugging
    with open(f"storage/json/{doc_id}.json", "w") as f:
        json.dump(elements, f, indent=2)

    return elements


# ── Phase 2 — Chunking ────────────────────────────────────────
# Matches notebook Cells 19 & 20 exactly
def build_parents(elements, doc_id):
    parents, parent_counter = [], 0
    current_segments, current_tokens, current_images = [], 0, []
    current_page = None

    def flush():
        nonlocal parent_counter, current_segments, current_tokens, current_images
        if not current_segments:
            return
        pid      = f"p_{doc_id}_{parent_counter}"
        combined = "\n\n".join(s.text for s in current_segments)
        parents.append(Parent(
            parent_id=pid, doc_id=doc_id, text=combined,
            page_num=current_page, token_count=current_tokens,
            image_refs=current_images.copy(), segments=current_segments.copy()
        ))
        parent_counter   += 1
        current_segments  = []
        current_tokens    = 0
        current_images    = []

    for el in elements:
        el_text   = el["text"] or ""
        el_type   = el["type"]
        el_tokens = count_tokens(el_text)
        el_image  = el.get("image_path")
        is_atomic = el_type in ATOMIC_TYPES
        current_page = el["page_num"]

        if is_atomic:
            current_segments.append(ElementSegment(el_text, True, el_image))
            current_tokens += el_tokens
            if el_image:
                current_images.append(el_image)
            if current_tokens >= PARENT_MAX_TOKENS:
                flush()
        else:
            if current_tokens + el_tokens > PARENT_MAX_TOKENS:
                flush()
            current_segments.append(ElementSegment(el_text, False))
            current_tokens += el_tokens

    flush()
    return parents


def build_children(parents):
    children = []

    for parent in parents:
        child_counter  = 0
        current_parts  = []
        current_tokens = 0
        overlap_buffer = []

        def flush_child():
            nonlocal child_counter, current_parts, current_tokens
            if not current_parts:
                return
            text = "\n\n".join(current_parts)
            children.append(Child(
                child_id=f"c_{parent.parent_id}_{child_counter}",
                parent_id=parent.parent_id, doc_id=parent.doc_id,
                text=text, page_num=parent.page_num,
                token_count=count_tokens(text), is_atomic=False,
            ))
            child_counter  += 1
            current_parts   = []
            current_tokens  = 0

        for seg in parent.segments:
            seg_tokens = count_tokens(seg.text)

            if seg.is_atomic:
                flush_child()
                children.append(Child(
                    child_id=f"c_{parent.parent_id}_{child_counter}",
                    parent_id=parent.parent_id, doc_id=parent.doc_id,
                    text=seg.text, page_num=parent.page_num,
                    token_count=seg_tokens, is_atomic=True,
                ))
                child_counter += 1
            else:
                carry = overlap_buffer + tokenize(seg.text)
                start = 0
                while start < len(carry):
                    end   = min(start + CHILD_TARGET_TOKENS, len(carry))
                    chunk = carry[start:end]
                    text  = detokenize(chunk)
                    if end == len(carry):
                        current_parts  = [text]
                        current_tokens = len(chunk)
                        overlap_buffer = chunk[-CHILD_OVERLAP_TOKENS:] if len(chunk) > CHILD_OVERLAP_TOKENS else chunk
                        break
                    else:
                        children.append(Child(
                            child_id=f"c_{parent.parent_id}_{child_counter}",
                            parent_id=parent.parent_id, doc_id=parent.doc_id,
                            text=text, page_num=parent.page_num,
                            token_count=len(chunk), is_atomic=False,
                        ))
                        child_counter  += 1
                        overlap_buffer  = chunk[-CHILD_OVERLAP_TOKENS:]
                        start          += CHILD_TARGET_TOKENS - CHILD_OVERLAP_TOKENS

        flush_child()
        overlap_buffer = []

    return children


def save_chunks(parents, children, doc_id):
    session = get_session()
    for p in parents:
        session.add(ParentModel(
            parent_id=p.parent_id, doc_id=p.doc_id, text=p.text,
            page_num=p.page_num, token_count=p.token_count, image_refs=p.image_refs,
        ))
    for c in children:
        session.add(ChildModel(
            child_id=c.child_id, parent_id=c.parent_id, doc_id=c.doc_id,
            text=c.text, page_num=c.page_num, token_count=c.token_count,
            is_atomic=str(c.is_atomic).lower(),
        ))
    session.commit()
    session.close()


# ── Phase 3 — Embed & Index ───────────────────────────────────
def embed_and_index(children, doc_id):
    chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
    try:
        chroma_client.delete_collection(f"children_{doc_id}")
    except Exception:
        pass

    collection = chroma_client.get_or_create_collection(
        f"children_{doc_id}", metadata={"hnsw:space": "cosine"}
    )

    ids        = [c.child_id for c in children]
    texts      = [c.text for c in children]
    metadatas  = [{
        "parent_id"  : c.parent_id,
        "doc_id"     : c.doc_id,
        "page_num"   : c.page_num or 0,
        "is_atomic"  : str(c.is_atomic).lower(),
        "token_count": c.token_count,
    } for c in children]

    embeddings = embedder.encode(texts, show_progress_bar=False).tolist()
    collection.add(ids=ids, documents=texts, embeddings=embeddings, metadatas=metadatas)
    return collection.count()


# ── Main pipeline ─────────────────────────────────────────────
def run_ingestion_pipeline(job_id: str, pdf_path: str, doc_id: str, pdf_hash: str, original_filename: str = None):

    print("\n" + "="*70)
    print("INGESTION PIPELINE STARTED")
    print("="*70)

    session = get_session()

    # Concern 8 — prefer the real uploaded filename. Only fall back to the
    # saved-path basename (which is typically "<doc_id>.pdf" or similar)
    # if the caller genuinely didn't pass one through, so we never crash
    # on old call sites mid-migration.
    display_filename = original_filename or os.path.basename(pdf_path)

    try:
        print("Step 1 -> Creating document entry...")

        session.add(
            DocModel(
                doc_id=doc_id,
                filename=display_filename,
                hash=pdf_hash,
                status="processing",
                created_at=datetime.utcnow().isoformat(),
            )
        )

        session.commit()

        print("✓ Document entry created")

        # ----------------------------------------------------

        print("Step 2 -> Parsing PDF...")

        update_job(
            job_id,
            status="processing",
            progress="Phase 1: Parsing & captioning..."
        )

        elements = parse_and_caption(pdf_path, doc_id, job_id)

        print(f"✓ Parsed {len(elements)} elements")

        # ----------------------------------------------------

        print("Step 3 -> Building parents...")

        parents = build_parents(elements, doc_id)

        print(f"✓ Parents = {len(parents)}")

        # ----------------------------------------------------

        print("Step 4 -> Building children...")

        children = build_children(parents)

        print(f"✓ Children = {len(children)}")

        # ----------------------------------------------------

        print("Step 5 -> Saving chunks...")

        save_chunks(parents, children, doc_id)

        print("✓ SQLite saved")

        # ----------------------------------------------------

        print("Step 6 -> Creating embeddings...")

        count = embed_and_index(children, doc_id)

        print(f"✓ Indexed {count} children")

        # ----------------------------------------------------

        print("Step 7 -> Updating document status...")

        doc = session.query(DocModel).filter_by(
            doc_id=doc_id
        ).first()

        doc.status = "done"

        session.commit()

        update_job(
            job_id,
            status="done",
            progress=f"Complete. {len(parents)} parents, {count} children indexed."
        )

        print("✓ DONE")

    except Exception as e:

        import traceback

        print("\n")
        print("="*70)
        print("PIPELINE FAILED")
        print("="*70)

        traceback.print_exc()

        try:
            session.query(ChildModel).filter_by(
                doc_id=doc_id
            ).delete()

            session.query(ParentModel).filter_by(
                doc_id=doc_id
            ).delete()

            doc = session.query(DocModel).filter_by(
                doc_id=doc_id
            ).first()

            if doc:
                doc.status = "failed"

            session.commit()

        except Exception:
            traceback.print_exc()

        try:
            chroma_client = chromadb.PersistentClient(
                path=CHROMA_PATH
            )

            chroma_client.delete_collection(
                f"children_{doc_id}"
            )

        except Exception:
            pass

        update_job(
            job_id,
            status="failed",
            error=str(e),
        )

        raise

    finally:

        session.close()

        print("="*70)
        print("PIPELINE FINISHED")
        print("="*70)
