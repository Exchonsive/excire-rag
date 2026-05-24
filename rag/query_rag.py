# ============================================================
# EXCIRE - RAG Engine: Query / Fact-Checker
# Fungsi: Semantic search → cari berita relevan sebagai
#         konteks fact-checking (cosine similarity)
# ============================================================

import os
import streamlit as st
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone


try:
    PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]
except:
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX   = os.getenv("PINECONE_INDEX", "excire-knowledge-base")
EMBED_MODEL      = "paraphrase-multilingual-MiniLM-L12-v2"
EMBED_DIM        = 384
TOP_K            = 5     # ambil 5 berita paling relevan
MIN_SCORE        = 0.45  # threshold cosine similarity

# ── LAZY LOAD (efficient untuk Streamlit) ────────────────────
_embedder = None
_index    = None

def _get_embedder():
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(EMBED_MODEL)
    return _embedder

def _get_index():
    global _index
    if _index is None:
        pc     = Pinecone(api_key=PINECONE_API_KEY)
        _index = pc.Index(PINECONE_INDEX)
    return _index

# ── MAIN QUERY FUNCTION ──────────────────────────────────────
def fact_check(query: str, top_k: int = TOP_K, min_score: float = MIN_SCORE) -> dict:
    """
    Semantic search terhadap knowledge base Pinecone.

    Input : query string (teks dari user / OCR hasil gambar)
    Output: {
        "verdict"  : "VERIFIED" | "UNVERIFIED" | "NO_DATA",
        "score"    : float (similarity score tertinggi),
        "sources"  : list[dict] (berita relevan),
        "summary"  : str (ringkasan hasil)
    }
    """
    embedder = _get_embedder()
    index    = _get_index()

    # Embed query
    query_vector = embedder.encode(
        query,
        normalize_embeddings = True
    ).tolist()

    # Query Pinecone
    results = index.query(
        vector          = query_vector,
        top_k           = top_k,
        include_metadata = True
    )

    matches = results.get("matches", [])

    # Filter berdasarkan threshold
    relevant = [m for m in matches if m["score"] >= min_score]

    # Susun output sources
    sources = []
    for m in relevant:
        meta = m.get("metadata", {})
        sources.append({
            "title"     : meta.get("title",     ""),
            "url"       : meta.get("url",       ""),
            "source"    : meta.get("source",    ""),
            "published" : meta.get("published", ""),
            "content"   : meta.get("content",   ""),
            "score"     : round(m["score"], 4),
        })

    # Tentukan verdict
    if not matches:
        verdict = "NO_DATA"
        summary = "Tidak ditemukan berita terkait di knowledge base."
    elif not relevant:
        top_score = matches[0]["score"]
        verdict   = "UNVERIFIED"
        summary   = f"Berita paling mirip hanya memiliki similarity {top_score:.2f} (di bawah threshold {min_score}). Klaim tidak dapat diverifikasi."
    else:
        top_score = relevant[0]["score"]
        if top_score >= 0.75:
            verdict = "VERIFIED"
            summary = f"Ditemukan {len(relevant)} berita relevan dengan similarity tinggi ({top_score:.2f}). Klaim kemungkinan besar benar."
        else:
            verdict = "UNVERIFIED"
            summary = f"Ditemukan {len(relevant)} berita terkait namun similarity sedang ({top_score:.2f}). Perlu verifikasi lebih lanjut."

    return {
        "verdict" : verdict,
        "score"   : round(matches[0]["score"], 4) if matches else 0.0,
        "sources" : sources,
        "summary" : summary,
    }


if __name__ == "__main__":
    test_queries = [
        "Presiden Prabowo menandatangani undang-undang baru",
        "Vaksin COVID menyebabkan kematian massal",
        "Gempa bumi melanda Jakarta hari ini",
    ]

    print("="*55)
    print("EXCIRE RAG - Fact Checker Test")
    print("="*55)

    for q in test_queries:
        result = fact_check(q)
        print(f"\nQuery   : {q}")
        print(f"Verdict : {result['verdict']}")
        print(f"Score   : {result['score']}")
        print(f"Summary : {result['summary']}")
        if result["sources"]:
            print(f"Source  : {result['sources'][0]['title'][:60]}...")
