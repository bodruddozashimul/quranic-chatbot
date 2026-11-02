"""
rag_chain.py
------------
Orchestrates: detect language -> retrieve -> refuse or explain.

Calls Ollama's /api/generate directly over HTTP. LangChain is not used: it added
a dependency tree without doing anything this single-step chain needs, and on a
memory-constrained host every avoided import counts.

Timings for each stage are returned with the response so the evaluation harness
can report real latency numbers instead of estimates.
"""

import logging
import os
import re
import time
from typing import Dict, List

import requests

from language_detect import detect_language, bengali_ratio
from retriever import QuranRetriever
from system_prompt import (
    EXPLANATION_PROMPT,
    LANGUAGE_INSTRUCTIONS,
    REFUSAL,
    build_context,
)

logger = logging.getLogger(__name__)

OLLAMA_URL = os.getenv("OLLAMA_BASE_URL", "http://host.docker.internal:11434")
MODEL_NAME = os.getenv("OLLAMA_MODEL", "qwen2.5:3b-instruct-q4_K_M")
NUM_PREDICT = int(os.getenv("NUM_PREDICT", "220"))
NUM_CTX = int(os.getenv("NUM_CTX", "2048"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.2"))
LLM_TIMEOUT = int(os.getenv("LLM_TIMEOUT", "180"))

# Leading filler small models like to emit despite being told not to.
_PREAMBLE = re.compile(
    r"^\s*(certainly|sure|of course|here is|here's|this verse|the verse|in this verse)"
    r"[ ,:.\-–—]*",
    re.IGNORECASE,
)


class QuranRAGChain:
    def __init__(self) -> None:
        self.retriever = QuranRetriever()
        logger.info("Retriever ready. LLM=%s at %s", MODEL_NAME, OLLAMA_URL)

    # ---------------------------------------------------------------- LLM
    def _generate(self, prompt: str) -> str:
        response = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={
                "model": MODEL_NAME,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": TEMPERATURE,
                    "top_p": 0.9,
                    "num_predict": NUM_PREDICT,
                    "num_ctx": NUM_CTX,
                    "repeat_penalty": 1.1,
                },
            },
            timeout=LLM_TIMEOUT,
        )
        response.raise_for_status()
        return response.json().get("response", "").strip()

    @staticmethod
    def _clean(text: str) -> str:
        text = _PREAMBLE.sub("", text.strip())
        text = re.sub(r"[*#`_]+", "", text)          # stray markdown
        text = re.sub(r"\n{2,}", " ", text)
        text = re.sub(r"\s{2,}", " ", text)
        return text.strip()

    # ------------------------------------------------------- fallback path
    @staticmethod
    def _deterministic_explanation(verses: List[Dict], language: str) -> str:
        """Used only when the model fails or answers in the wrong language.

        This is not a canned answer: it is assembled from the retrieved
        metadata at request time, and it deliberately says less rather than
        risking an ungrounded claim. If you see this text during a demo, the
        model call did not succeed -- say so if asked.
        """
        names = ", ".join(
            f"{v['surah_name_en']} {v['surah_no']}:{v['ayah_no']}" for v in verses
        )
        if language == "bengali":
            return (
                f"উপরে প্রদর্শিত আয়াত ({names}) আপনার প্রশ্নের সঙ্গে সবচেয়ে বেশি "
                "সাদৃশ্যপূর্ণ হিসেবে পাওয়া গেছে। অনুগ্রহ করে উপরের বাংলা অনুবাদটি "
                "পড়ুন — উত্তরটি সরাসরি সেখানেই রয়েছে। এই সিস্টেম সংরক্ষিত আয়াতের "
                "বাইরে কোনো তথ্য যোগ করে না।"
            )
        return (
            f"The verse shown above ({names}) was retrieved as the closest match to "
            "your question. Please read the translation above; the answer is contained "
            "directly in it. This system does not add anything beyond the retrieved text."
        )

    # ------------------------------------------------------------- public
    def query(self, question: str) -> Dict:
        language = detect_language(question)

        t0 = time.perf_counter()
        verses = self.retriever.search(question)
        retrieval_time = time.perf_counter() - t0

        if not verses:
            return {
                "answer": REFUSAL[language],
                "retrieved_ayahs": [],
                "language_detected": language,
                "sources": [],
                "refused": True,
                "retrieval_time": round(retrieval_time, 3),
                "generation_time": 0.0,
                "total_time": round(retrieval_time, 3),
            }

        prompt = EXPLANATION_PROMPT.format(
            language_instruction=LANGUAGE_INSTRUCTIONS[language],
            context=build_context(verses),
            question=question,
        )

        t1 = time.perf_counter()
        try:
            answer = self._clean(self._generate(prompt))

            # Small models sometimes drift to English on a Bengali question.
            # One retry with a blunter instruction, then the deterministic path.
            if language == "bengali" and bengali_ratio(answer) < 0.30:
                logger.warning("Bengali answer came back in the wrong script; retrying.")
                answer = self._clean(self._generate(
                    prompt + "\n\nReminder: the reply MUST be written in Bengali script (বাংলা)."
                ))
                if bengali_ratio(answer) < 0.30:
                    answer = self._deterministic_explanation(verses, language)

            if len(answer) < 25:
                answer = self._deterministic_explanation(verses, language)

        except Exception as exc:
            logger.error("Generation failed: %s", exc)
            answer = self._deterministic_explanation(verses, language)
        generation_time = time.perf_counter() - t1

        return {
            "answer": answer,
            "retrieved_ayahs": verses,
            "language_detected": language,
            # Sources are built from metadata, never from model output.
            "sources": [
                f"Surah {v['surah_name_en']} ({v['surah_name_ar']}) · Ayah {v['ayah_no']}"
                for v in verses
            ],
            "refused": False,
            "retrieval_time": round(retrieval_time, 3),
            "generation_time": round(generation_time, 3),
            "total_time": round(retrieval_time + generation_time, 3),
        }
