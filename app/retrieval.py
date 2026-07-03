# ==============================================================
# app/retrieval.py
# Updated to match actual notebook implementation:
# - LLM model: models/gemini-3.5-flash (matches notebook Cell 35)
# - search_children caps n_results to collection size (avoids ChromaDB error)
# - image path: relative → absolute → URL conversion
# Concern 2 — query result caching
# Concern 7 — relative path → full URL conversion at serve time
# ==============================================================

import os
import base64
import chromadb
from sentence_transformers import SentenceTransformer
import google.generativeai as genai

from app.storage import get_session, ParentModel

# ── Config ────────────────────────────────────────────────────
CHROMA_PATH     = "storage/chroma"
IMAGE_BASE_DIR  = "storage/images"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
TOP_K_CHILDREN  = 10
TOP_N_PARENTS   = 3

GOOGLE_API_KEY  = os.environ.get("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise RuntimeError(
        "GOOGLE_API_KEY environment variable is not set. "
        "Set it as a secret in your deployment environment."
    )
genai.configure(api_key=GOOGLE_API_KEY)

# LLM model — matches notebook Cell 35
llm      = genai.GenerativeModel("models/gemini-3.5-flash")
embedder = SentenceTransformer(EMBEDDING_MODEL)

# Concern 2 — in-memory query cache
_query_cache: dict = {}


# ── Helpers ───────────────────────────────────────────────────
def image_rel_to_url(rel_path: str, base_url: str) -> str:
    """Concern 7 — relative storage path → full API URL."""
    return f"{base_url.rstrip('/')}/{rel_path}"


# ── Step 1: Search children ───────────────────────────────────
def search_children(query: str, doc_id: str, k: int = TOP_K_CHILDREN):
    chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection    = chroma_client.get_collection(f"children_{doc_id}")

    # Cap k to collection size — matches notebook behavior
    # (avoids ChromaDB error when k > number of documents)
    k = min(k, collection.count())

    query_vec = embedder.encode(query).tolist()
    results   = collection.query(
        query_embeddings=[query_vec],
        n_results=k,
        include=["documents", "metadatas", "distances"],
    )

    hits = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        hits.append({
            "text"      : doc,
            "parent_id" : meta["parent_id"],
            "page_num"  : meta["page_num"],
            "is_atomic" : meta["is_atomic"],
            "score"     : round(1 - dist, 4),
        })
    return hits


# ── Step 2: Resolve to parents ────────────────────────────────
def resolve_parents(hits: list, n: int = TOP_N_PARENTS):
    seen, ranked_ids = {}, []
    for hit in hits:
        pid = hit["parent_id"]
        if pid not in seen:
            seen[pid] = hit["score"]
            ranked_ids.append(pid)
        if len(ranked_ids) == n:
            break

    session = get_session()
    parents = []
    for pid in ranked_ids:
        p = session.query(ParentModel).filter_by(parent_id=pid).first()
        if p:
            parents.append({
                "parent_id" : p.parent_id,
                "text"      : p.text,
                "page_num"  : p.page_num,
                "image_refs": p.image_refs or [],
                "score"     : seen[pid],
            })
    session.close()
    return parents


# ── Step 3: Build multimodal prompt ───────────────────────────
# Matches notebook Cell 40 exactly
def build_prompt(query: str, parents: list):
    context_parts = []
    for i, p in enumerate(parents):
        block  = f"--- Source {i+1} (Parent: {p['parent_id']}, Page: {p['page_num']}) ---\n"
        block += p["text"]
        context_parts.append(block)

    context_text = "\n\n".join(context_parts)

    prompt = f"""You are a document analyst. Answer the user's question using ONLY
the provided document context below. Be precise with numbers and data.
If the answer involves a table or chart, refer to it explicitly.
Answer directly and naturally, as if you already know this information — do not
mention "Source 1", "the context", "the document provided", or similar labels.

DOCUMENT CONTEXT:
{context_text}

USER QUESTION:
{query}

ANSWER:"""

    content = [prompt]

    # Attach images — resolve relative path to absolute for file reading
    for p in parents:
        for img_rel in p["image_refs"]:
            img_abs = os.path.join(IMAGE_BASE_DIR, img_rel)
            if os.path.exists(img_abs):
                with open(img_abs, "rb") as f:
                    img_b64 = base64.b64encode(f.read()).decode()
                ext        = img_abs.split(".")[-1].lower()
                media_type = "image/png" if ext == "png" else "image/jpeg"
                content.append({"mime_type": media_type, "data": img_b64})

    return content


# ── Step 4: Generate answer ───────────────────────────────────
# Matches notebook Cell 41 — returns answer + sources
def generate_answer(query: str, content: list, parents: list, base_image_url: str):
    response = llm.generate_content(content)
    answer   = response.text

    sources = []
    for p in parents:
        # Concern 7 — convert relative paths to full image URLs for API response
        image_urls = [
            image_rel_to_url(ref, base_image_url)
            for ref in p["image_refs"]
        ]
        sources.append({
            "parent_id" : p["parent_id"],
            "page_num"  : p["page_num"],
            "score"     : p["score"],
            "snippet"   : p["text"][:200] + "...",
            "image_urls": image_urls,
        })

    return answer, sources


# ── Full retrieval pipeline ───────────────────────────────────
def retrieve(query: str, doc_id: str, base_image_url: str = "http://localhost:8000/images"):
    # Concern 2 — return cached result if available
    cache_key = f"{doc_id}:{query.lower().strip()}"
    if cache_key in _query_cache:
        print(f"Cache hit: '{query}'")
        return _query_cache[cache_key]

    hits    = search_children(query, doc_id)
    parents = resolve_parents(hits)
    content = build_prompt(query, parents)
    answer, sources = generate_answer(query, content, parents, base_image_url)

    _query_cache[cache_key] = (answer, sources)
    return answer, sources
