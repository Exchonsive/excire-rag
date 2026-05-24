# ============================================================
# EXCIRE - RAG Engine: Embedder
# Fungsi: Embed artikel → vektor, upsert ke Pinecone
# Model : paraphrase-multilingual-MiniLM-L12-v2 (768 dim)
# ============================================================

import os
import time
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec
from scraper import scrape_all_feeds

# ── CONFIG ───────────────────────────────────────────────────
try:
    PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]
except:
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX   = os.getenv("PINECONE_INDEX", "excire-knowledge-base")
EMBED_MODEL      = "paraphrase-multilingual-MiniLM-L12-v2"
EMBED_DIM        = 384
BATCH_SIZE       = 50   # upsert per batch

# ── INIT ─────────────────────────────────────────────────────
print("Loading embedding model...")
embedder = SentenceTransformer(EMBED_MODEL)

print("Connecting to Pinecone...")
pc = Pinecone(api_key=PINECONE_API_KEY)

# Buat index kalau belum ada (safety check)
existing = [idx.name for idx in pc.list_indexes()]
if PINECONE_INDEX not in existing:
    print(f"Index '{PINECONE_INDEX}' tidak ditemukan, membuat baru...")
    pc.create_index(
        name      = PINECONE_INDEX,
        dimension = EMBED_DIM,
        metric    = "cosine",
        spec      = ServerlessSpec(cloud="aws", region="us-east-1")
    )
    time.sleep(10)  # tunggu index siap

index = pc.Index(PINECONE_INDEX)
print(f"Index '{PINECONE_INDEX}' siap!")

# ── EMBED & UPSERT ───────────────────────────────────────────
def embed_and_upsert(articles: list[dict]) -> int:
    """
    Embed semua artikel dan upsert ke Pinecone.
    Return jumlah artikel yang berhasil di-upsert.
    """
    if not articles:
        print("Tidak ada artikel untuk di-embed.")
        return 0

    total_upserted = 0

    # Proses per batch
    for i in range(0, len(articles), BATCH_SIZE):
        batch    = articles[i : i + BATCH_SIZE]
        contents = [a["content"] for a in batch]

        # Embed semua teks dalam batch sekaligus
        vectors = embedder.encode(
            contents,
            batch_size      = 32,
            show_progress_bar = False,
            normalize_embeddings = True   # L2 normalize → cosine = dot product
        )

        # Format untuk Pinecone upsert
        upsert_data = []
        for article, vector in zip(batch, vectors):
            upsert_data.append({
                "id"     : article["id"],
                "values" : vector.tolist(),
                "metadata": {
                    "title"     : article["title"][:500],   # Pinecone metadata limit
                    "url"       : article["url"][:500],
                    "source"    : article["source"],
                    "published" : article["published"],
                    "content"   : article["content"][:1000] # simpan untuk retrieval
                }
            })

        # Upsert ke Pinecone
        index.upsert(vectors=upsert_data)
        total_upserted += len(upsert_data)
        print(f"  Batch {i//BATCH_SIZE + 1}: upsert {len(upsert_data)} artikel")

    return total_upserted


# ── MAIN ─────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "="*50)
    print("EXCIRE RAG - Scrape & Embed Pipeline")
    print("="*50)

    # 1. Scrape
    print("\n[1/2] Scraping RSS feeds...")
    articles = scrape_all_feeds(max_per_feed=20)

    # 2. Embed & Upsert
    print("\n[2/2] Embedding & upserting ke Pinecone...")
    total = embed_and_upsert(articles)

    # 3. Stats
    stats = index.describe_index_stats()
    print(f"\n✅ Selesai! Upserted: {total} artikel")
    print(f"📊 Total vektor di Pinecone: {stats['total_vector_count']}")
