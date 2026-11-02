"""
ingest.py  (memory-safe + per-language documents)
-------------------------------------------------
Embeds all 6236 ayahs and loads them into ChromaDB.

Two separate changes from the original, for two separate problems:

  MEMORY. The original built all 6236 vectors in one list comprehension over
  `embedder.embed(documents, batch_size=256)`. On a 3 GB WSL2 VM already running
  chromadb + backend + streamlit, the ONNX activations for a 256-row batch plus
  the accumulated float lists push the container past its limit and the kernel
  OOM-killer takes it out silently. Now each chunk is embedded, inserted, and
  freed before the next starts, so peak memory stays flat.

  RELEVANCE. The original embedded `english_text + " " + bengali_text` as a
  single document. Mean pooling averages over every token, so a long verse
  dilutes its own topic across ~60 tokens while a two-word verse concentrates
  its meaning -- short verses then win every query regardless of subject, and
  half of each document is in a language the query is not written in.

  Each verse now produces TWO documents, one per translation, sharing the same
  metadata. An English question is compared against English text and a Bengali
  question against Bengali text, and neither is diluted by the other. The
  retriever collapses the two back into one verse per result.

  Collection size doubles to 12472 entries, but the total volume of embedded
  text is unchanged, so the run takes roughly as long as before.
"""

import gc
import json
import os
import time

import chromadb
from fastembed import TextEmbedding

CHROMA_HOST = os.getenv("CHROMA_HOST", "chromadb")
CHROMA_PORT = int(os.getenv("CHROMA_PORT", "8000"))
DATASET_PATH = os.getenv("DATASET_PATH", "/app/data/quran_full.json")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "quran_ayahs")
EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

# Two threads, not four: each ONNX thread keeps its own memory arena.
EMBED_THREADS = int(os.getenv("EMBED_THREADS", "2"))
# Rows sent to ONNX at once. 256 was the OOM trigger.
BATCH = int(os.getenv("EMBED_BATCH", "32"))
# Verses embedded and written to Chroma before memory is released.
CHUNK = int(os.getenv("CHUNK_SIZE", "250"))


def log(msg: str) -> None:
    print(msg, flush=True)


def wait_for_chroma(client, attempts: int = 60) -> None:
    for i in range(attempts):
        try:
            client.heartbeat()
            log("ChromaDB is up.")
            return
        except Exception as exc:
            log(f"  waiting for ChromaDB... ({i + 1}/{attempts}) {exc}")
            time.sleep(3)
    raise RuntimeError("ChromaDB did not become reachable.")


def row_to_metadata(r: dict, lang: str) -> dict:
    """Both language entries carry the full verse, so the UI card renders
    identically no matter which translation matched the query."""
    return {
        "surah_no": str(r["surah_no"]),
        "ayah_no": str(r["ayah_no"]),
        "surah_name_en": r["surah_name_en"],
        "surah_name_ar": r["surah_name_ar"],
        "arabic_text": r["arabic_text"],
        "english_text": r["english_text"],
        "bengali_text": r["bengali_text"],
        "lang": lang,
    }


def main() -> None:
    # ---- Vector store first, so a connection problem costs seconds ----
    client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
    wait_for_chroma(client)

    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        rows = json.load(f)
    log(f"Loaded {len(rows)} ayahs from {DATASET_PATH}")

    try:
        client.delete_collection(COLLECTION_NAME)
        log("Removed previous collection.")
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    log(f"Embedding with {EMBED_MODEL} "
        f"(ONNX, {EMBED_THREADS} threads, batch={BATCH}, chunk={CHUNK})...")
    embedder = TextEmbedding(model_name=EMBED_MODEL, threads=EMBED_THREADS)

    t0 = time.time()
    total = len(rows)
    written = 0

    for start in range(0, total, CHUNK):
        chunk = rows[start:start + CHUNK]

        documents, ids, metadatas = [], [], []
        for r in chunk:
            base = f"s{r['surah_no']}_a{r['ayah_no']}"

            english = (r.get("english_text") or "").strip()
            if english:
                documents.append(english)
                ids.append(f"{base}_en")
                metadatas.append(row_to_metadata(r, "en"))

            bengali = (r.get("bengali_text") or "").strip()
            if bengali:
                documents.append(bengali)
                ids.append(f"{base}_bn")
                metadatas.append(row_to_metadata(r, "bn"))

        vectors = [v.tolist() for v in embedder.embed(documents, batch_size=BATCH)]

        collection.add(
            ids=ids,
            documents=documents,
            embeddings=vectors,
            metadatas=metadatas,
        )
        written += len(ids)

        done = min(start + CHUNK, total)
        log(f"  {done}/{total} verses  ({written} entries)  "
            f"({time.time() - t0:.0f}s elapsed)")

        del vectors, documents, ids, metadatas, chunk
        gc.collect()

    count = collection.count()
    log(f"Ingestion complete. Collection holds {count} entries "
        f"across {total} ayahs.")
    if count != written:
        raise RuntimeError(f"Expected {written} entries in the collection, found {count}.")


if __name__ == "__main__":
    main()
