"""
retriever.py
------------
Verse retrieval over ChromaDB.

Two things here matter for the reported metrics:

1. Every field shown to the user -- Arabic, Bengali, English, Surah name and
   Ayah number -- is read from the vector store's metadata. The language model
   never produces a reference. That is what makes citation accuracy a property
   of the code rather than of the model.

2. SIM_THRESHOLD is the out-of-scope gate. If the best verse scores below it,
   nothing is returned, the chain refuses, and the model is never given a
   chance to invent an answer. Tune this on your own data: too low and
   irrelevant verses reach the user, too high and valid questions get refused.

CHANGED: the collection now holds one entry per verse PER LANGUAGE
(`s2_a153_en` and `s2_a153_bn`), so a query never compares against a document
half of which is in the other language. This retriever therefore over-fetches a
candidate pool, keeps the better-scoring language entry for each verse, and
returns one result per verse. Callers see the same shape as before.
"""

import os
import re
from typing import List, Dict

import chromadb
from fastembed import TextEmbedding

CHROMA_HOST = os.getenv("CHROMA_HOST", "chromadb")
CHROMA_PORT = int(os.getenv("CHROMA_PORT", "8000"))
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "quran_ayahs")
EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
EMBED_THREADS = int(os.getenv("EMBED_THREADS", "2"))

TOP_K = int(os.getenv("TOP_K", "3"))
SIM_THRESHOLD = float(os.getenv("SIM_THRESHOLD", "0.35"))
USE_HYBRID = os.getenv("USE_HYBRID", "false").lower() == "true"
BM25_WEIGHT = float(os.getenv("BM25_WEIGHT", "0.3"))

# How many raw entries to pull before collapsing languages and cutting to
# top_k. Both language entries of the same verse often land in the pool, so
# fetch well beyond what is needed.
POOL_MULTIPLIER = int(os.getenv("POOL_MULTIPLIER", "10"))
MIN_POOL = int(os.getenv("MIN_POOL", "30"))

_TOKEN = re.compile(r"\w+", re.UNICODE)
_LANG_SUFFIX = re.compile(r"_(en|bn)$")


def _tokenize(text: str) -> List[str]:
    return _TOKEN.findall(text.lower())


def _verse_key(entry_id: str) -> str:
    """`s2_a153_en` -> `s2_a153`. Safe on ids that carry no suffix."""
    return _LANG_SUFFIX.sub("", entry_id)


class QuranRetriever:
    def __init__(self) -> None:
        self.embedder = TextEmbedding(model_name=EMBED_MODEL, threads=EMBED_THREADS)
        self.collection = self._connect()
        self.bm25 = None
        if USE_HYBRID:
            self._build_bm25()

    @staticmethod
    def _connect(attempts: int = 40):
        """Retry rather than crash. On a slow disk the backend often wins the
        startup race against ChromaDB and the ingest job, and a container that
        exits on the first failure turns that into a restart loop."""
        import time

        last = None
        for i in range(attempts):
            try:
                client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
                return client.get_collection(name=COLLECTION_NAME)
            except Exception as exc:
                last = exc
                print(f"  waiting for ChromaDB collection... ({i + 1}/{attempts}) {exc}",
                      flush=True)
                time.sleep(5)
        raise RuntimeError(f"Could not reach ChromaDB collection: {last}")

    def _build_bm25(self) -> None:
        """Optional lexical index. Note the limitation kept from the original:
        fusion only re-ranks verses dense retrieval already surfaced, so BM25
        can raise a verse's position but never introduce one the dense pool
        missed. Widen POOL_MULTIPLIER if you want it to have more to work on."""
        from rank_bm25 import BM25Okapi

        dump = self.collection.get(include=["documents"])
        self._bm25_ids = [_verse_key(i) for i in dump["ids"]]
        self.bm25 = BM25Okapi([_tokenize(d) for d in dump["documents"]])

    def embed_query(self, question: str) -> List[float]:
        return list(self.embedder.embed([question]))[0].tolist()

    def search(self, question: str, top_k: int = TOP_K) -> List[Dict]:
        qvec = self.embed_query(question)
        pool = max(top_k * POOL_MULTIPLIER, MIN_POOL)

        res = self.collection.query(
            query_embeddings=[qvec],
            n_results=pool,
            include=["metadatas", "distances"],
        )

        # Collapse the per-language entries: keep whichever translation of a
        # verse scored higher against this question.
        best: Dict[str, tuple] = {}
        for i, meta in enumerate(res["metadatas"][0]):
            # Chroma cosine distance is in [0, 2]; similarity = 1 - distance.
            similarity = 1.0 - float(res["distances"][0][i])
            key = _verse_key(res["ids"][0][i])
            if key not in best or similarity > best[key][1]:
                best[key] = (meta, similarity)

        ranked = sorted(
            ((k, m, s) for k, (m, s) in best.items()),
            key=lambda t: t[2],
            reverse=True,
        )

        if self.bm25:
            ranked = self._fuse(question, ranked)

        out = []
        for vid, meta, similarity in ranked[:top_k]:
            if similarity < SIM_THRESHOLD:
                continue
            out.append({
                "verse_id": vid,
                "surah_no": meta["surah_no"],
                "ayah_no": meta["ayah_no"],
                "surah_name_en": meta["surah_name_en"],
                "surah_name_ar": meta["surah_name_ar"],
                "arabic_text": meta["arabic_text"],
                "english_text": meta["english_text"],
                "bengali_text": meta["bengali_text"],
                "similarity_score": round(similarity, 4),
            })
        return out

    def _fuse(self, question: str, dense: List) -> List:
        """Reciprocal-rank fusion, weighted so dense retrieval stays dominant."""
        scores, payload = {}, {}
        for rank, (vid, meta, sim) in enumerate(dense):
            scores[vid] = scores.get(vid, 0.0) + 1.0 / (60 + rank + 1)
            payload[vid] = (meta, sim)

        import numpy as np

        bm_scores = self.bm25.get_scores(_tokenize(question))
        seen = set()
        for rank, idx in enumerate(np.argsort(-bm_scores)[:40]):
            vid = self._bm25_ids[idx]
            if vid in seen:          # both language entries map to one verse
                continue
            seen.add(vid)
            if vid in payload:       # only re-rank verses dense retrieval surfaced
                scores[vid] += BM25_WEIGHT / (60 + len(seen))

        order = sorted(scores, key=scores.get, reverse=True)
        return [(v, payload[v][0], payload[v][1]) for v in order if v in payload]
