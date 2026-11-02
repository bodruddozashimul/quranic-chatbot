"""
app.py
------
Streamlit interface. Laid out to match Figure "fig_ui_screenshot" in the report:
green header bar, sidebar with history and sample questions, a verse card with
separate Arabic / Bengali / English panels and a reference badge, and a separate
explanation panel underneath.

The verse card is rendered entirely from the API's `retrieved_ayahs` metadata.
The model's text goes only into the explanation panel. Nothing the model writes
can change a Surah name or an Ayah number on screen.

FIXED in this version:

  1. Errors survive. The original called `st.error(...)` and then `st.rerun()`,
     which discards the run the error was drawn into. Every failure -- backend
     down, timeout, HTTP 500 -- rendered as a blank screen. Errors now go into
     session_state and are drawn at the top of the next run.

  2. Sample questions send in one click. The original only prefilled the text
     box; you still had to press Send. Clicking a sample now asks it.

  3. A backend status line is always visible, so a dead backend is obvious
     before you type anything rather than after.

  4. The request timeout is 600s and the HTTP error path reports the status
     code and the API's `detail` field instead of a generic message.
"""

import html
import os
from datetime import datetime

import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://backend:8000")
REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "600"))

st.set_page_config(
    page_title="Quran RAG Assistant",
    page_icon="🕌",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Amiri:wght@400;700&family=Hind+Siliguri:wght@400;600&family=Inter:wght@400;500;600;700&display=swap');

body, .stApp { font-family: 'Inter', sans-serif; }
.block-container { padding-top: 1rem; max-width: 1100px; }
#MainMenu, footer, header { visibility: hidden; }

.app-header {
    background: #1a6b3c;
    color: #ffffff;
    padding: 18px 28px;
    border-radius: 10px;
    margin-bottom: 18px;
    display: flex;
    align-items: center;
    justify-content: space-between;
}
.app-header .title { font-size: 26px; font-weight: 700; letter-spacing: -0.01em; }
.app-header .tag   { font-size: 14px; opacity: 0.92; }

.arabic-verse {
    font-family: 'Amiri', 'Scheherazade New', serif;
    font-size: 30px;
    direction: rtl;
    text-align: right;
    line-height: 2.3;
    color: #1a6b3c;
    padding: 18px 20px;
    background: #f0fdf4;
    border-radius: 12px;
    border-right: 4px solid #16a34a;
    margin: 6px 0;
}
.bengali-text {
    font-family: 'Hind Siliguri', sans-serif;
    font-size: 17px; line-height: 1.9; color: #1e3a5f;
    padding: 14px 18px; background: #eff6ff;
    border-radius: 8px; margin: 6px 0;
}
.english-text {
    font-size: 15px; line-height: 1.7; color: #374151; font-style: italic;
    padding: 14px 18px; background: #fafaf9;
    border-left: 3px solid #d1d5db; border-radius: 4px; margin: 6px 0;
}
.reference-badge {
    display: inline-block; background: #1a6b3c; color: #fff;
    padding: 7px 18px; border-radius: 999px;
    font-size: 14px; font-weight: 600; margin: 10px 0 2px 0;
}
.ayah-card {
    border: 1px solid #e5e7eb; border-radius: 16px; padding: 22px;
    margin: 10px 0; background: #fff; box-shadow: 0 1px 4px rgba(0,0,0,0.06);
}
.section-label {
    font-size: 11px; font-weight: 600; text-transform: uppercase;
    letter-spacing: 0.08em; color: #9ca3af; margin: 12px 0 4px 0;
}
.explanation-box {
    font-family: 'Hind Siliguri', 'Inter', sans-serif;
    background: #f9fafb; border: 1px solid #eef0f2; border-radius: 12px;
    padding: 16px 20px; font-size: 15px; line-height: 1.85;
    color: #1f2937; margin-top: 14px;
}
.user-bubble {
    background: #1a6b3c; color: #fff; padding: 13px 20px;
    border-radius: 18px 18px 4px 18px; margin: 14px 0 6px auto;
    max-width: 72%; width: fit-content; font-size: 15px;
}
.meta-line { font-size: 13px; color: #6b7280; margin: 2px 0 8px 0; }
.refusal-box {
    background: #fffbeb; border: 1px solid #fde68a; border-radius: 12px;
    padding: 16px 20px; font-size: 15px; line-height: 1.8; color: #78350f;
    font-family: 'Hind Siliguri', 'Inter', sans-serif;
}
.stTextInput > div > div > input {
    border-radius: 24px !important; border: 2px solid #d1fae5 !important;
    font-size: 15px !important; padding: 12px 20px !important;
}
.stButton > button {
    border-radius: 24px !important; background: #1a6b3c !important;
    color: #fff !important; border: none !important; font-weight: 600 !important;
}
section[data-testid="stSidebar"] .stButton > button {
    background: #f0fdf4 !important; color: #14532d !important;
    border: 1px solid #bbf7d0 !important; text-align: left !important;
    font-weight: 500 !important; font-size: 13px !important; width: 100%;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="app-header">
  <div class="title">🕌 Quran RAG Assistant</div>
  <div class="tag">Local AI &nbsp;•&nbsp; Zero-Hallucination Policy</div>
</div>
""", unsafe_allow_html=True)

if "messages" not in st.session_state:
    st.session_state.messages = []
if "history" not in st.session_state:
    st.session_state.history = []

SAMPLES = [
    ("About patience", "What does the Quran say about patience?"),
    ("About forgiveness", "What does the Quran say about forgiving others?"),
    ("Ramadan virtues (BN)", "রমজান মাস ও কোরআন নাজিলের কথা কোন আয়াতে আছে?"),
    ("Description of Paradise", "How does the Quran describe Paradise?"),
    ("Kindness to parents (BN)", "পিতা-মাতার সাথে আচরণ সম্পর্কে কোরআনে কী বলা হয়েছে?"),
    ("Out-of-scope check", "What is the capital city of Australia?"),
]


def backend_health() -> dict:
    """Returns the /health payload, or {'_error': ...} if it cannot be read."""
    try:
        r = requests.get(f"{API_URL}/health", timeout=5)
        r.raise_for_status()
        return r.json()
    except Exception as exc:
        return {"_error": f"{type(exc).__name__}: {exc}"}


def ask_backend(question: str) -> None:
    """POST to /chat and record either the answer or a readable failure.

    Nothing here draws to the screen: the caller reruns afterwards and the
    results are rendered from session_state, so no message can be lost to a
    discarded run.
    """
    st.session_state.messages.append({"role": "user", "content": question})
    response = None
    try:
        response = requests.post(
            f"{API_URL}/chat",
            json={"question": question, "conversation_history": []},
            timeout=REQUEST_TIMEOUT,
        )
        response.raise_for_status()
        st.session_state.messages.append({"role": "assistant", "data": response.json()})
        st.session_state.history.append({
            "question": question,
            "timestamp": datetime.now().isoformat(),
        })
    except requests.exceptions.ConnectionError as exc:
        st.session_state["error"] = (
            f"Cannot reach the backend at {API_URL}. "
            f"Check `docker compose ps`. ({exc})"
        )
    except requests.exceptions.Timeout:
        st.session_state["error"] = (
            f"No answer within {REQUEST_TIMEOUT}s. The model is probably still "
            "loading into RAM, or NUM_PREDICT is too high for this machine."
        )
    except requests.exceptions.HTTPError:
        detail = ""
        if response is not None:
            try:
                detail = response.json().get("detail", "")
            except Exception:
                detail = response.text[:400]
            st.session_state["error"] = f"Backend returned {response.status_code}: {detail}"
        else:
            st.session_state["error"] = "Backend returned an HTTP error."
    except Exception as exc:
        st.session_state["error"] = f"{type(exc).__name__}: {exc}"


def render_ayah_card(ayah: dict) -> None:
    """Every field here comes from vector-store metadata, escaped before display."""
    e = lambda k: html.escape(str(ayah.get(k, "")))
    reference = (f"Surah {e('surah_name_en')} · Ayah {e('ayah_no')} "
                 f"· Surah {e('surah_no')}")
    st.markdown(f"""
    <div class="ayah-card">
      <div class="section-label">Arabic Verse</div>
      <div class="arabic-verse">{e('arabic_text')}</div>
      <div class="section-label">Bengali Translation</div>
      <div class="bengali-text">{e('bengali_text')}</div>
      <div class="section-label">English Translation</div>
      <div class="english-text">“{e('english_text')}”</div>
      <div><span class="reference-badge">📖 {reference}</span></div>
    </div>
    """, unsafe_allow_html=True)


def render_assistant_response(result: dict) -> None:
    verses = result.get("retrieved_ayahs", [])

    if result.get("refused") or not verses:
        st.markdown(
            f'<div class="refusal-box">🛡️ {html.escape(result.get("answer", ""))}</div>',
            unsafe_allow_html=True,
        )
        st.caption(f"Correctly refused · no verse above similarity threshold "
                   f"· {result.get('total_time', 0):.1f}s")
        return

    top = verses[0].get("similarity_score", 0.0)
    noun = "verse" if len(verses) == 1 else "verses"
    st.markdown(
        f'<div class="meta-line">📖 Retrieved {len(verses)} grounded {noun} '
        f'&nbsp;•&nbsp; similarity {top:.2f}</div>',
        unsafe_allow_html=True,
    )

    for ayah in verses:
        render_ayah_card(ayah)

    st.markdown('<div class="section-label">Explanation</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="explanation-box">{html.escape(result.get("answer", ""))}</div>',
        unsafe_allow_html=True,
    )

    flag = "🇧🇩 Bengali query detected" if result.get("language_detected") == "bengali" \
        else "🇬🇧 English query detected"
    st.caption(f"{flag} · retrieval {result.get('retrieval_time', 0):.2f}s "
               f"· generation {result.get('generation_time', 0):.1f}s")


# ─── Sidebar ──────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 📋 Chat History")
    if st.session_state.history:
        for item in reversed(st.session_state.history[-15:]):
            st.markdown(f"<div style='font-size:13px;color:#374151;padding:4px 0'>"
                        f"💬 {html.escape(item['question'][:60])}</div>",
                        unsafe_allow_html=True)
    else:
        st.caption("No history yet. Start asking!")

    st.divider()
    st.markdown("### 💡 Sample Questions")
    for label, question in SAMPLES:
        # One click asks the question outright.
        if st.button(f"• {label}", key=f"s_{label}"):
            st.session_state["ask"] = question
            st.rerun()

    st.divider()
    if st.button("🗑️ Clear Conversation"):
        st.session_state.messages = []
        st.session_state.history = []
        st.session_state.pop("error", None)
        st.rerun()

    st.divider()
    with st.expander("System status", expanded=True):
        h = backend_health()
        if "_error" in h:
            st.error(f"Backend unreachable: {h['_error']}")
        else:
            st.write(f"**Model:** `{h.get('model')}`")
            st.write(f"**Verses indexed:** {h.get('verses_indexed')}")
            st.write(f"**ChromaDB:** {h.get('chroma')}")
            st.write(f"**Ollama:** {h.get('ollama')}")

# ─── Errors from the previous run, drawn where they cannot be lost ────
if st.session_state.get("error"):
    st.error(f"❌ {st.session_state.pop('error')}")

# ─── Pending question: do the blocking call, then rerun to render ─────
queued = st.session_state.pop("ask", "")
if queued:
    for message in st.session_state.messages:
        if message["role"] == "user":
            st.markdown(f'<div class="user-bubble">{html.escape(message["content"])}</div>',
                        unsafe_allow_html=True)
        else:
            render_assistant_response(message["data"])
    st.markdown(f'<div class="user-bubble">{html.escape(queued)}</div>',
                unsafe_allow_html=True)
    with st.spinner("🔍 Searching the Quran... (first call after an idle period "
                    "reloads the model and can take a minute)"):
        ask_backend(queued)
    st.rerun()

# ─── Conversation ─────────────────────────────────────────────────────
for message in st.session_state.messages:
    if message["role"] == "user":
        st.markdown(f'<div class="user-bubble">{html.escape(message["content"])}</div>',
                    unsafe_allow_html=True)
    else:
        render_assistant_response(message["data"])

with st.form("chat_form", clear_on_submit=True):
    col1, col2 = st.columns([8, 1])
    with col1:
        user_input = st.text_input(
            "Ask about the Quran",
            placeholder="Ask about the Quran...  /  বাংলায় প্রশ্ন করতে পারেন",
            label_visibility="collapsed",
        )
    with col2:
        submitted = st.form_submit_button("Send ➤")

if submitted and user_input.strip():
    st.session_state["ask"] = user_input.strip()
    st.rerun()
