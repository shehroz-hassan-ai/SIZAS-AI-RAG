import html
import json
import re
import time
from datetime import datetime
from pathlib import Path

import streamlit as st

from src.chunker import chunk_pages
from src.config import (
    CHUNK_OVERLAP, CHUNK_SIZE, NOT_FOUND_MESSAGE,
    SIMILARITY_THRESHOLD, TOP_K, validate_config,
)
from src.embedder import embed_texts
from src.generator import GenerationError, generate_answer
from src.loader import PDFLoadError, load_pdf
from src.retriever import retrieve
from src.vector_store import (
    VectorStoreError, delete_namespace, get_index,
    list_namespaces, make_namespace, upsert_chunks,
)

LOG_FILE = Path("logs/queries.jsonl")

# Edit these to match the PDF you demo with.
EXAMPLES = [
    "When was Zenith Tech founded?",
    "Who is eligible for remote work?",
    "Summarize all types of leave",
    "What does health insurance cover?",
]

st.set_page_config(page_title="SIZAS AI | RAG PDF Assistant", page_icon="📑", layout="wide")

# ---------------------------------------------------------------- login
if not st.user.is_logged_in:
    st.title("SIZAS AI")
    st.write("Please sign in to continue.")
    st.button("Sign in with Google", on_click=st.login)
    st.stop()

st.sidebar.write(f"Signed in as {st.user.name}")
st.sidebar.button("Log out", on_click=st.logout)

try:
    validate_config()
except ValueError as e:
    st.error(str(e))
    st.stop()

# ---------------------------------------------------------------- styling
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,600;12..96,800&family=Figtree:wght@400;500;600&display=swap');

:root{
  --ink:#14183A; --ink-deep:#0E1130; --ink-soft:#1E2350;
  --paper:#F2F4FF; --paper-text:#1B1F45; --paper-line:#D9DDF5;
  --marker:#FFD84D; --coral:#FF6B8A; --muted:#A3A9D9;
}

[data-testid="stApp"]{
  background-color:var(--ink);
  background-image:radial-gradient(rgba(163,169,217,.14) 1px, transparent 1px);
  background-size:22px 22px;
  color:#E9EBFF;
}
[data-testid="stApp"] p, [data-testid="stApp"] li, [data-testid="stApp"] label,
[data-testid="stApp"] input, [data-testid="stApp"] textarea, [data-testid="stApp"] button{
  font-family:'Figtree', system-ui, sans-serif;
}
h1,h2,h3{font-family:'Bricolage Grotesque', 'Figtree', sans-serif !important; letter-spacing:-.02em;}

#MainMenu, footer{visibility:hidden;}
header[data-testid="stHeader"]{background:transparent;}
.block-container{max-width:980px; padding-top:1.6rem; padding-bottom:8rem;}

/* ---------- sidebar ---------- */
[data-testid="stSidebar"]{background:var(--ink-deep); border-right:1px solid rgba(163,169,217,.15);}
[data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3{
  font-size:1.05rem; color:#fff; margin-top:.4rem;
}
[data-testid="stFileUploaderDropzone"]{
  background:var(--ink-soft); border:1.5px dashed rgba(163,169,217,.5); border-radius:12px;
}
[data-baseweb="tag"]{background:var(--marker) !important; border-radius:6px !important;}
[data-baseweb="tag"] *{color:var(--ink) !important; font-weight:600;}

/* primary action = marker yellow, everything else = outline */
button[kind="primary"], [data-testid="stBaseButton-primary"]{
  background:var(--marker) !important; color:var(--ink) !important;
  border:none !important; font-weight:700 !important; border-radius:10px !important;
}
button[kind="primary"]:hover, [data-testid="stBaseButton-primary"]:hover{background:#FFE580 !important;}
button[kind="secondary"], [data-testid="stBaseButton-secondary"]{
  background:transparent !important; color:#E9EBFF !important;
  border:1.5px solid rgba(163,169,217,.45) !important; border-radius:999px !important;
}
button[kind="secondary"]:hover, [data-testid="stBaseButton-secondary"]:hover{
  border-color:var(--marker) !important; color:var(--marker) !important;
}

/* ---------- top bar ---------- */
.topbar{display:flex; align-items:center; justify-content:space-between; gap:16px; flex-wrap:wrap; margin-bottom:1.2rem;}
.wordmark{
  font-family:'Bricolage Grotesque', sans-serif; font-weight:800; font-size:1.6rem; color:#fff;
  letter-spacing:-.03em; padding:0 .25rem;
  background:linear-gradient(transparent 58%, rgba(255,216,77,.6) 58%) no-repeat;
  background-size:100% 100%;
  animation:swipe .8s .2s cubic-bezier(.2,.8,.2,1) backwards;
}
@keyframes swipe{from{background-size:0% 100%;}}
.pills{display:flex; gap:8px; flex-wrap:wrap;}
.pill{
  font-size:.82rem; font-weight:600; color:var(--ink); background:var(--paper);
  padding:4px 12px; border-radius:999px;
}
.pill.dim{background:transparent; color:var(--muted); border:1px solid rgba(163,169,217,.35);}

/* ---------- hero ---------- */
.hero{display:grid; grid-template-columns:1.35fr 1fr; gap:32px; align-items:center; margin:1.2rem 0 1.6rem;}
.hero h1{
  font-size:clamp(2.4rem, 5.6vw, 4.3rem); line-height:1.02; font-weight:800;
  letter-spacing:-.035em; color:#fff; margin:0 0 1rem; padding:0;
}
.hero h1 span{display:block; animation:rise .75s cubic-bezier(.2,.8,.2,1) backwards;}
.hero h1 .l2{animation-delay:.12s;}
.hero p{color:var(--muted); font-size:1.08rem; line-height:1.6; max-width:48ch; margin:0 0 1rem; animation:rise .75s .3s cubic-bezier(.2,.8,.2,1) backwards;}
.badges{display:flex; gap:8px; flex-wrap:wrap; animation:rise .75s .42s cubic-bezier(.2,.8,.2,1) backwards;}
.badges span{
  font-size:.82rem; font-weight:600; color:var(--muted); padding:4px 12px; border-radius:999px;
  border:1px solid rgba(163,169,217,.4);
}
.badges span:first-child{background:var(--marker); color:var(--ink); border-color:var(--marker);}
@keyframes rise{from{opacity:0; transform:translateY(26px);}}

.stack{position:relative; height:250px;}
.stack-inner{position:absolute; inset:0; animation:float 6s ease-in-out 1.6s infinite;}
@keyframes float{0%,100%{transform:translateY(0);} 50%{transform:translateY(-9px);}}
.stack .sheet{
  position:absolute; left:50%; top:10px; width:200px; height:236px; margin-left:-100px;
  background:var(--paper); border-radius:6px; box-shadow:0 2px 0 rgba(0,0,0,.25);
  animation:fan 1s .1s cubic-bezier(.2,.8,.2,1) backwards;
}
.stack .s3{transform:rotate(-10deg); opacity:.55; animation-delay:.1s;}
.stack .s2{transform:rotate(-5deg); opacity:.8; animation-delay:.2s;}
.stack .s1{transform:rotate(3deg); padding:44px 20px 0; animation-delay:.3s;}
@keyframes fan{from{transform:rotate(0deg) translateY(30px); opacity:0;}}
.stack .pg{
  position:absolute; top:-12px; right:16px; background:var(--ink-deep); color:var(--marker);
  font-family:'Bricolage Grotesque', sans-serif; font-weight:800; font-size:.85rem;
  padding:4px 12px; border-radius:999px; border:1px solid rgba(255,216,77,.5);
  animation:pop .5s 1.5s cubic-bezier(.3,1.6,.5,1) backwards;
}
@keyframes pop{from{transform:scale(0); opacity:0;}}
.stack .ln{display:block; height:8px; background:#C9CEEE; border-radius:4px; margin:15px 0;}
.stack .w90{width:90%;} .stack .w80{width:80%;} .stack .w70{width:70%;} .stack .w50{width:50%;}
.stack .ln.hl{background:var(--marker); width:100%; animation:sweep 1.1s 1s ease-out backwards;}
@keyframes sweep{from{width:0;}}
@media (max-width:760px){ .hero{grid-template-columns:1fr;} .stack{height:210px;} }

.callout{
  border:1.5px dashed rgba(163,169,217,.5); border-radius:14px; padding:18px 20px;
  color:#E9EBFF; background:rgba(30,35,80,.6); margin-bottom:1rem;
}
.nudge{color:var(--muted); font-size:.98rem; margin:.4rem 0 .6rem;}

/* example question chips */
[data-testid="stMain"] .stButton > button, .main .stButton > button{
  transition:transform .18s ease, border-color .18s ease, color .18s ease;
  text-align:left; justify-content:flex-start;
}
[data-testid="stMain"] .stButton > button:hover, .main .stButton > button:hover{transform:translateY(-3px);}

/* thinking state: a highlighter scanning skeleton lines */
.thinking{margin:.4rem 0 1.4rem; max-width:520px;}
.thinking .bar{position:relative; height:10px; background:#262B63; border-radius:5px; overflow:hidden; margin:10px 0;}
.thinking .bar::after{
  content:""; position:absolute; top:0; bottom:0; left:0; width:35%; background:var(--marker);
  animation:scan 1.2s ease-in-out infinite;
}
.thinking .bar:nth-child(2){width:80%;} .thinking .bar:nth-child(2)::after{animation-delay:.2s;}
.thinking .bar:nth-child(3){width:60%;} .thinking .bar:nth-child(3)::after{animation-delay:.4s;}
@keyframes scan{0%{transform:translateX(-100%);} 100%{transform:translateX(300%);}}
.thinking .say{color:var(--muted); font-size:.95rem; margin-top:.5rem;}

/* ---------- chat ---------- */
[data-testid="stChatMessage"]{background:transparent; padding:0; margin-bottom:1.5rem; gap:0;}
[data-testid="stChatMessageAvatarUser"], [data-testid="stChatMessageAvatarAssistant"],
[data-testid="chatAvatarIcon-user"], [data-testid="chatAvatarIcon-assistant"]{display:none;}

/* the question */
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]),
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]){
  border-left:5px solid var(--marker); padding:2px 0 2px 16px; margin-bottom:.9rem;
}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) p,
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) p{
  font-family:'Bricolage Grotesque', sans-serif; font-weight:600; font-size:1.55rem;
  line-height:1.25; color:#fff; letter-spacing:-.02em;
}

/* the answer = a sheet of paper */
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]),
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]){
  background:var(--paper); color:var(--paper-text); border-radius:6px 20px 20px 20px;
  padding:1.3rem 1.5rem 1.1rem; box-shadow:0 2px 0 rgba(0,0,0,.3);
}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) *,
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) *{color:var(--paper-text);}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) p,
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) p{font-size:1.08rem; line-height:1.65;}
[data-testid="stChatMessage"] [data-testid="stExpander"]{
  background:transparent; border:1.5px dashed #B9BFE6; border-radius:12px;
}

/* match strength ruler */
.ruler{display:flex; align-items:center; gap:12px; margin:.9rem 0 1rem;}
.ruler-label{font-size:.85rem; font-weight:600; color:#4A5085 !important; white-space:nowrap;}
.ruler-track{position:relative; flex:1; height:12px; background:var(--paper-line); border-radius:3px; overflow:hidden;}
.ruler-fill{height:100%; background:var(--marker); animation:grow 1.1s .2s cubic-bezier(.2,.8,.2,1) backwards;}
@keyframes grow{from{width:0 !important;}}
.ruler-track::after{
  content:""; position:absolute; inset:0;
  background:repeating-linear-gradient(90deg, transparent 0 9px, rgba(27,31,69,.28) 9px 10px);
}
.ruler-val{font-family:'Bricolage Grotesque', sans-serif; font-weight:800; font-size:1rem; min-width:3ch; text-align:right;}

/* evidence sheets */
.sheets{display:grid; gap:20px; margin:14px 4px 8px;}
.sheet-card{
  position:relative; background:#fff; border:1px solid var(--paper-line); border-radius:6px;
  padding:20px 20px 12px;
  box-shadow:0 1px 0 var(--paper-line), 0 6px 0 -3px #E9ECFB, 0 7px 0 -3px var(--paper-line);
}
.sheet-card{animation:deal .55s cubic-bezier(.2,.8,.2,1) backwards;}
.sheet-card:nth-child(2){animation-delay:.12s;} .sheet-card:nth-child(3){animation-delay:.24s;}
.sheet-card:nth-child(even){transform:rotate(.35deg);}
@keyframes deal{from{opacity:0; transform:translateY(-16px) rotate(-2.5deg);}}
.sheet-card .tab{
  position:absolute; top:-13px; right:18px; background:var(--ink); color:var(--marker) !important;
  font-family:'Bricolage Grotesque', sans-serif; font-weight:800; font-size:.85rem;
  padding:3px 12px; border-radius:999px;
}
.sheet-body{font-size:.97rem; line-height:1.7; color:#2B3060 !important;}
.sheet-body mark{
  background:linear-gradient(transparent 10%, var(--marker) 10%, var(--marker) 92%, transparent 92%);
  color:var(--paper-text) !important; padding:0 2px; border-radius:2px;
  -webkit-box-decoration-break:clone; box-decoration-break:clone;
}
.sheet-meta{display:flex; justify-content:space-between; gap:12px; margin-top:10px; font-size:.8rem; color:#6B71A3 !important;}
.hint{margin-top:.8rem; font-size:.95rem; color:#4A5085 !important;}

/* chat input */
[data-testid="stBottom"] > div{background:transparent;}
[data-testid="stChatInput"]{border-radius:16px; transition:box-shadow .2s ease;}
[data-testid="stChatInput"]:focus-within{box-shadow:0 0 0 3px rgba(255,216,77,.55);}
[data-testid="stChatInput"]{background:transparent;}
[data-testid="stChatInput"] textarea{color:#fff !important; -webkit-text-fill-color:#fff !important; caret-color:var(--marker) !important;}
[data-testid="stChatInput"] textarea::placeholder{color:var(--muted) !important; -webkit-text-fill-color:var(--muted) !important; opacity:1 !important;}
@media (prefers-reduced-motion:reduce){*{animation:none !important; transition:none !important;}}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

HERO = (
    '<div class="hero">'
    '<div class="hero-text">'
    '<h1><span class="l1">Ask your PDF.</span><span class="l2">Get the page.</span></h1>'
    "<p>SIZAS AI is a RAG assistant. It retrieves the right passages from your document and answers "
    "only from them, with the page and the exact sentence it used.</p>"
    '<div class="badges"><span>Retrieval-Augmented Generation</span><span>Pinecone</span><span>Groq</span></div>'
    "</div>"
    '<div class="stack" aria-hidden="true"><div class="stack-inner">'
    '<div class="sheet s3"></div><div class="sheet s2"></div>'
    '<div class="sheet s1"><span class="pg">p. 2</span>'
    '<i class="ln w90"></i><i class="ln w70"></i><i class="ln hl"></i>'
    '<i class="ln w80"></i><i class="ln w50"></i></div>'
    "</div></div></div>"
)

THINKING = (
    '<div class="thinking"><div class="bar"></div><div class="bar"></div><div class="bar"></div>'
    '<div class="say">Reading the PDF and finding the page…</div></div>'
)

# ---------------------------------------------------------------- helpers
_WORD = re.compile(r"[a-z0-9]{4,}")


def log_query(query, namespaces, top_score, answered):
    LOG_FILE.parent.mkdir(exist_ok=True)
    entry = {
        "time": datetime.now().isoformat(timespec="seconds"),
        "query": query,
        "documents": namespaces,
        "top_score": top_score,
        "answered": answered,
    }
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def highlight(text, answer, n=2):
    """Escape the chunk and wrap the sentences that overlap most with the answer in <mark>."""
    text = " ".join(text.split())
    if len(text) > 700:
        text = text[:700].rsplit(" ", 1)[0] + " …"
    sentences = re.split(r"(?<=[.!?])\s+", text)
    ans_words = set(_WORD.findall(answer.lower()))
    scored = sorted(
        ((len(set(_WORD.findall(s.lower())) & ans_words), i) for i, s in enumerate(sentences)),
        reverse=True,
    )
    best = {i for score, i in scored[:n] if score >= 2}
    parts = []
    for i, s in enumerate(sentences):
        safe = html.escape(s)
        parts.append(f"<mark>{safe}</mark>" if i in best else safe)
    return " ".join(parts)


def strength_html(score):
    pct = int(min(max(score, 0.0), 1.0) * 100)
    return (
        '<div class="ruler"><span class="ruler-label">Match strength</span>'
        f'<div class="ruler-track"><div class="ruler-fill" style="width:{pct}%"></div></div>'
        f'<span class="ruler-val">{pct}%</span></div>'
    )


def sources_html(sources, answer):
    cards = []
    for s in sources:
        cards.append(
            '<div class="sheet-card">'
            f'<span class="tab">p. {html.escape(str(s["page"]))}</span>'
            f'<div class="sheet-body">{highlight(s["text"], answer)}</div>'
            f'<div class="sheet-meta"><span>{html.escape(str(s["doc_name"]))}</span>'
            f'<span>similarity {s["score"]:.2f}</span></div>'
            "</div>"
        )
    return '<div class="sheets">' + "".join(cards) + "</div>"


def word_stream(text):
    parts = re.split(r"(\s+)", text)
    delay = min(0.03, 2.0 / max(len(parts) / 2, 1))
    for part in parts:
        yield part
        time.sleep(delay)


def render_message(msg, expanded=False, stream=False):
    if msg["role"] == "user":
        with st.chat_message("user"):
            st.markdown(msg["content"])
        return
    with st.chat_message("assistant"):
        if stream:
            st.write_stream(word_stream(msg["content"]))
        else:
            st.markdown(msg["content"])
        if msg.get("confidence") is not None:
            st.markdown(strength_html(msg["confidence"]), unsafe_allow_html=True)
        sources = msg.get("sources")
        if sources:
            label = f"Where this came from ({len(sources)})"
            with st.expander(label, expanded=expanded):
                st.markdown(sources_html(sources, msg["content"]), unsafe_allow_html=True)
        elif msg.get("answered") is False:
            st.markdown(
                '<div class="hint">Nothing in the selected PDF answers this. Try different words, '
                "or lower the match threshold under Fine-tune search.</div>",
                unsafe_allow_html=True,
            )


def set_pending(q):
    st.session_state.pending = q


if "messages" not in st.session_state:
    st.session_state.messages = []

# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("Add a PDF")
    uploaded = st.file_uploader("Choose a PDF (max 20 MB)", type=["pdf"], label_visibility="collapsed")

    with st.expander("Fine-tune search"):
        chunk_size = st.slider("Chunk size", 300, 1500, CHUNK_SIZE, 50)
        chunk_overlap = st.slider("Chunk overlap", 0, 300, CHUNK_OVERLAP, 10)
        top_k = st.slider("Excerpts to retrieve (top-k)", 1, 10, TOP_K)
        threshold = st.slider("Match threshold", 0.0, 1.0, SIMILARITY_THRESHOLD, 0.05)
        use_page_filter = st.checkbox("Limit to a page range")
        page_range = None
        if use_page_filter:
            lo = st.number_input("From page", 1, value=1)
            hi = st.number_input("To page", 1, value=10)
            page_range = (int(lo), int(hi))

    if uploaded and st.button("Index this PDF", type="primary", use_container_width=True):
        try:
            with st.status("Indexing…", expanded=True) as status:
                st.write("Reading the text")
                pages = load_pdf(uploaded.getvalue(), uploaded.name)
                ns = make_namespace(uploaded.name)

                st.write("Splitting into chunks")
                chunks = chunk_pages(pages, ns, chunk_size, chunk_overlap)

                st.write(f"Embedding {len(chunks)} chunks")
                vectors = embed_texts([c["text"] for c in chunks])

                st.write("Saving to Pinecone")
                index = get_index()
                delete_namespace(index, ns)  # remove old chunks of the same file
                upsert_chunks(index, ns, chunks, vectors)
                status.update(
                    label=f"Indexed {len(pages)} pages, {len(chunks)} chunks", state="complete"
                )
        except (PDFLoadError, VectorStoreError) as e:
            st.error(str(e))

    st.header("Search in")
    try:
        available = list_namespaces(get_index())
    except VectorStoreError as e:
        st.error(str(e))
        available = []
    selected = st.multiselect(
        "Documents", available, default=available, label_visibility="collapsed"
    )

    if st.button("Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# ---------------------------------------------------------------- main
pills = "".join(f'<span class="pill">{html.escape(n)}</span>' for n in selected)
if not pills:
    pills = '<span class="pill dim">no document selected</span>'
st.markdown(
    f'<div class="topbar"><span class="wordmark">SIZAS AI</span><span class="pills">{pills}</span></div>',
    unsafe_allow_html=True,
)

if not st.session_state.messages:
    st.markdown(HERO, unsafe_allow_html=True)
    if not selected:
        st.markdown(
            '<div class="callout">Add a PDF in the sidebar and press <b>Index this PDF</b>. '
            "Then ask your first question.</div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown('<div class="nudge">Not sure what to ask? Start with one of these.</div>', unsafe_allow_html=True)
        cols = st.columns(2)
        for i, q in enumerate(EXAMPLES):
            cols[i % 2].button(q, key=f"example_{i}", on_click=set_pending, args=(q,), use_container_width=True)

for msg in st.session_state.messages:
    render_message(msg)

query = st.chat_input("Ask a question about your PDF")
if not query and st.session_state.get("pending"):
    query = st.session_state.pop("pending")

if query:
    if not query.strip():
        st.warning("Type a question first.")
    elif not selected:
        st.warning("Add a PDF and pick it under Search in first.")
    else:
        user_msg = {"role": "user", "content": query}
        st.session_state.messages.append(user_msg)
        render_message(user_msg)

        thinking = st.empty()
        thinking.markdown(THINKING, unsafe_allow_html=True)
        try:
            chunks = retrieve(query, selected, top_k, threshold, page_range)
            answer = generate_answer(query, chunks)
            thinking.empty()

            answered = answer.strip() != NOT_FOUND_MESSAGE
            sources = chunks if answered else []
            confidence = (
                min(max((c["score"] for c in chunks), default=0.0), 1.0) if answered else None
            )

            assistant_msg = {
                "role": "assistant",
                "content": answer,
                "sources": sources,
                "confidence": confidence,
                "answered": answered,
            }
            st.session_state.messages.append(assistant_msg)
            render_message(assistant_msg, expanded=True, stream=True)
            log_query(query, selected, confidence, answered)

        except (ValueError, VectorStoreError, GenerationError) as e:
            thinking.empty()
            st.error(str(e))