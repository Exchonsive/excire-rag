# ============================================================
# EXCIRE - RAG Engine: Scraper
# Fungsi: Ambil berita terbaru dari RSS Feed portal resmi
# Source: CNN Indonesia, Kompas, Tempo
# ============================================================

import feedparser
import requests
import hashlib
import re
from bs4 import BeautifulSoup
from datetime import datetime, timezone

# ── RSS FEED SOURCES ─────────────────────────────────────────
RSS_FEEDS = {
    "cnn_indonesia": [
        "https://www.cnnindonesia.com/nasional/rss",
        "https://www.cnnindonesia.com/teknologi/rss",
        "https://www.cnnindonesia.com/politik/rss",
    ],
    "kompas": [
        "https://rss.kompas.com/nasional",
        "https://rss.kompas.com/tren",
        "https://rss.kompas.com/regional",
    ],
    "tempo": [
        "https://rss.tempo.co/nasional",
        "https://rss.tempo.co/politik",
        "https://rss.tempo.co/hukum",
    ],
}

# ── HELPER: Bersihkan teks HTML ──────────────────────────────
def clean_html(raw: str) -> str:
    soup = BeautifulSoup(raw, "html.parser")
    text = soup.get_text(separator=" ")
    text = re.sub(r"\s+", " ", text).strip()
    return text

# ── HELPER: Buat ID unik per artikel ────────────────────────
def make_article_id(url: str) -> str:
    return hashlib.md5(url.encode()).hexdigest()

# ── MAIN SCRAPER ─────────────────────────────────────────────
def scrape_all_feeds(max_per_feed: int = 20) -> list[dict]:
    """
    Scrape semua RSS feed dan return list artikel.
    
    Output per artikel:
    {
        "id"        : str  (MD5 hash dari URL, jadi unique key Pinecone),
        "title"     : str,
        "content"   : str  (title + summary digabung untuk embedding),
        "url"       : str,
        "source"    : str,
        "published" : str  (ISO format)
    }
    """
    articles = []
    seen_ids = set()

    for source, feed_urls in RSS_FEEDS.items():
        for feed_url in feed_urls:
            try:
                feed = feedparser.parse(feed_url)

                if feed.bozo and not feed.entries:
                    print(f"  ⚠️  Skip (parse error): {feed_url}")
                    continue

                count = 0
                for entry in feed.entries[:max_per_feed]:
                    url        = entry.get("link", "")
                    article_id = make_article_id(url)

                    # Skip duplikat
                    if article_id in seen_ids:
                        continue
                    seen_ids.add(article_id)

                    title   = clean_html(entry.get("title",   ""))
                    summary = clean_html(entry.get("summary", ""))

                    # Gabung title + summary → teks untuk embedding
                    content = f"{title}. {summary}".strip()

                    # Skip kalau konten terlalu pendek
                    if len(content) < 30:
                        continue

                    # Parse tanggal publikasi
                    published = ""
                    if hasattr(entry, "published_parsed") and entry.published_parsed:
                        dt        = datetime(*entry.published_parsed[:6], tzinfo=timezone.utc)
                        published = dt.isoformat()
                    else:
                        published = datetime.now(timezone.utc).isoformat()

                    articles.append({
                        "id"        : article_id,
                        "title"     : title,
                        "content"   : content,
                        "url"       : url,
                        "source"    : source,
                        "published" : published,
                    })
                    count += 1

                print(f"  ✅ {source} | {feed_url.split('/')[-1]} → {count} artikel")

            except Exception as e:
                print(f"  ❌ Error scraping {feed_url}: {e}")

    print(f"\nTotal artikel terkumpul: {len(articles)}")
    return articles


if __name__ == "__main__":
    results = scrape_all_feeds()
    print(f"\nSample artikel pertama:")
    print(results[0] if results else "Tidak ada hasil")