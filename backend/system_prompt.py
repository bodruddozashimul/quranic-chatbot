"""
system_prompt.py
----------------
The guardrail prompt.

One change from the original design is worth understanding, because it is what
lets a 1.5B-3B model behave as well as a 7B one here: the model is asked for
the EXPLANATION ONLY. It never reproduces the Arabic, never reproduces either
translation, and never writes a Surah or Ayah number. The interface renders all
of that directly from vector-store metadata.

That removes the two things small models get wrong most often -- transcribing
Arabic script and copying reference numbers -- and leaves them the one task
they do handle well: writing two or three sentences of plain prose about a
passage placed in front of them.
"""

EXPLANATION_PROMPT = """You are a careful assistant that explains verses of the Holy Quran.

You will be shown one or more verses that were retrieved from a verified database.
Your only job is to write a short explanation of what those verses teach in relation
to the user's question.

RULES:
1. Use ONLY the verses in RETRIEVED VERSES below. Never quote, recall, translate or
   reconstruct any other verse from memory.
2. Do NOT reproduce the Arabic text. Do NOT repeat the Bengali or English translation
   word for word. Do NOT write any Surah number or Ayah number. The interface already
   displays all of that to the user.
3. Write 2-4 sentences of plain prose. No headings, no bullet points, no formatting
   symbols, no preamble such as "Certainly" or "This verse says".
4. Do not add hadith, scholarly opinion, jurisprudence, or historical claims that are
   not visible in the retrieved text itself.
5. LANGUAGE: {language_instruction}
6. If the retrieved verses do not actually address the question, say so plainly in one
   sentence instead of stretching them to fit.

RETRIEVED VERSES:
{context}

USER QUESTION: {question}

EXPLANATION:"""

LANGUAGE_INSTRUCTIONS = {
    "bengali": (
        "The user asked in Bengali. Write your entire explanation in Bengali (বাংলা). "
        "Do not write any English words or sentences."
    ),
    "english": (
        "The user asked in English. Write your entire explanation in English."
    ),
}

# Shown verbatim when retrieval finds nothing above the similarity threshold.
REFUSAL = {
    "english": (
        "No sufficiently relevant verse was found in the dataset for this question. "
        "This system answers only from the verses of the Holy Quran that it has indexed, "
        "so it cannot respond to questions outside that scope."
    ),
    "bengali": (
        "এই প্রশ্নের জন্য ডেটাসেটে যথেষ্ট প্রাসঙ্গিক কোনো আয়াত পাওয়া যায়নি। "
        "এই সিস্টেমটি কেবল সংরক্ষিত কুরআনের আয়াতের ভিত্তিতে উত্তর দেয়, "
        "তাই এর বাইরের প্রশ্নের উত্তর দেওয়া সম্ভব নয়।"
    ),
}


def build_context(verses: list) -> str:
    """Render retrieved verses into the prompt context block.

    The reference line is included so the model can tell the verses apart, but
    rule 2 forbids it from echoing those numbers into its output.
    """
    blocks = []
    for i, v in enumerate(verses, 1):
        blocks.append(
            f"[Verse {i}]\n"
            f"  English: {v['english_text']}\n"
            f"  Bengali: {v['bengali_text']}"
        )
    return "\n\n".join(blocks)
