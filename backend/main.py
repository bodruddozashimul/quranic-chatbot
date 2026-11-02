"""
main.py
-------
FastAPI surface over the RAG chain.

The /chat contract is unchanged from the version described in the report, with
three fields added that the evaluation harness needs: `refused`, and the
retrieval / generation timings.
"""

import logging
import os
from contextlib import asynccontextmanager
from typing import List, Optional

import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from rag_chain import QuranRAGChain, MODEL_NAME, OLLAMA_URL

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

rag_chain: Optional[QuranRAGChain] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Build the chain once at startup and warm the model.

    The warmup matters for a live demo: the first Ollama call after a cold
    start pays for loading several gigabytes of weights from disk. Paying that
    here means the first question an examiner asks is not the slow one.
    """
    global rag_chain
    logger.info("Initializing Quran RAG chain...")
    rag_chain = QuranRAGChain()
    try:
        requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={"model": MODEL_NAME, "prompt": "ok", "stream": False,
                  "options": {"num_predict": 1}},
            timeout=300,
        )
        logger.info("Model warm.")
    except Exception as exc:
        logger.warning("Warmup skipped (%s). First request will be slower.", exc)
    logger.info("RAG chain ready.")
    yield


app = FastAPI(
    title="Quran RAG API",
    description="Local, offline RAG over the Holy Quran with Bengali and English support",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    question: str
    conversation_history: List[dict] = []


class AyahResult(BaseModel):
    verse_id: str
    surah_no: str
    ayah_no: str
    surah_name_en: str
    surah_name_ar: str
    arabic_text: str
    english_text: str
    bengali_text: str
    similarity_score: float


class ChatResponse(BaseModel):
    answer: str
    retrieved_ayahs: List[AyahResult]
    language_detected: str
    sources: List[str]
    refused: bool
    retrieval_time: float
    generation_time: float
    total_time: float


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    if rag_chain is None:
        raise HTTPException(status_code=503, detail="Service still starting up.")
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    if len(request.question) > 1000:
        raise HTTPException(status_code=400, detail="Question too long (max 1000 chars).")

    try:
        return ChatResponse(**rag_chain.query(request.question))
    except Exception as exc:
        logger.exception("RAG chain error")
        raise HTTPException(status_code=500, detail=f"Internal error: {exc}")


@app.get("/health")
async def health():
    """Reports on both dependencies so a failing demo can be diagnosed in one call."""
    status = {
        "status": "ok" if rag_chain is not None else "starting",
        "model": MODEL_NAME,
        "ollama_url": OLLAMA_URL,
        "chroma": "unknown",
        "ollama": "unknown",
        "verses_indexed": None,
    }
    if rag_chain is not None:
        try:
            status["verses_indexed"] = rag_chain.retriever.collection.count()
            status["chroma"] = "ok"
        except Exception as exc:
            status["chroma"] = f"error: {exc}"
    try:
        tags = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5).json()
        names = [m["name"] for m in tags.get("models", [])]
        status["ollama"] = "ok" if any(MODEL_NAME in n for n in names) else \
            f"reachable, but {MODEL_NAME} not pulled. Available: {names}"
    except Exception as exc:
        status["ollama"] = f"unreachable: {exc}"
    return status


@app.get("/")
async def root():
    return {"message": "Quran RAG API is running. POST to /chat to query."}
