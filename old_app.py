import json
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

st.set_page_config(page_title="PDF RAG Assistant", page_icon="📄", layout="wide")

try:
    validate_config()
except ValueError as e:
    st.error(str(e))
    st.stop()


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


if "messages" not in st.session_state:
    st.session_state.messages = []

# ---------------- Sidebar ----------------
with st.sidebar:
    st.header("⚙️ Settings")
    chunk_size = st.slider("Chunk size", 300, 1500, CHUNK_SIZE, 50)
    chunk_overlap = st.slider("Chunk overlap", 0, 300, CHUNK_OVERLAP, 10)
    top_k = st.slider("Top-k", 1, 10, TOP_K)
    threshold = st.slider("Similarity threshold", 0.0, 1.0, SIMILARITY_THRESHOLD, 0.05)

    st.divider()
    st.header("📤 Upload PDF")
    uploaded = st.file_uploader("PDF chuno (max 20 MB)", type=["pdf"])

    if uploaded and st.button("Index karo", use_container_width=True):
        try:
            with st.status("Processing...", expanded=True) as status:
                st.write("Text nikal raha hoon...")
                pages = load_pdf(uploaded.getvalue(), uploaded.name)
                ns = make_namespace(uploaded.name)

                st.write("Chunks bana raha hoon...")
                chunks = chunk_pages(pages, ns, chunk_size, chunk_overlap)

                st.write(f"{len(chunks)} chunks ke embeddings...")
                vectors = embed_texts([c["text"] for c in chunks])

                st.write("Pinecone mein save...")
                index = get_index()
                delete_namespace(index, ns)  # purane chunks hatao
                upsert_chunks(index, ns, chunks, vectors)
                status.update(label=f"Done: {len(pages)} pages, {len(chunks)} chunks", state="complete")
        except (PDFLoadError, VectorStoreError) as e:
            st.error(str(e))

    st.divider()
    st.header("📚 Documents")
    try:
        available = list_namespaces(get_index())
    except VectorStoreError as e:
        st.error(str(e))
        available = []

    selected = st.multiselect("Kin documents mein dhoondna hai?", available, default=available)

    use_page_filter = st.checkbox("Page filter")
    page_range = None
    if use_page_filter:
        lo, hi = st.number_input("Page se", 1, value=1), st.number_input("Page tak", 1, value=10)
        page_range = (int(lo), int(hi))

    if st.button("🗑️ Chat clear", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# ---------------- Main ----------------
st.title("📄 PDF RAG Assistant")
st.caption("Jawab sirf aapki PDF ke content se aata hai.")


def show_sources(sources):
    if not sources:
        return
    with st.expander(f"📎 Sources ({len(sources)})"):
        for s in sources:
            st.markdown(f"**{s['doc_name']} — Page {s['page']}** · Similarity: `{s['score']:.2f}`")
            st.caption(s["text"][:400] + ("..." if len(s["text"]) > 400 else ""))
            st.divider()


for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("confidence") is not None:
            st.progress(msg["confidence"], text=f"Confidence: {msg['confidence']:.0%}")
        show_sources(msg.get("sources"))

query = st.chat_input("PDF ke baare mein sawal poochho...")

if query:
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        if not query.strip():
            st.warning("Sawal khali hai.")
        elif not selected:
            st.warning("Pehle PDF index karo aur document select karo.")
        else:
            try:
                with st.spinner("Soch raha hoon..."):
                    chunks = retrieve(query, selected, top_k, threshold, page_range)
                    answer = generate_answer(query, chunks)

                answered = answer.strip() != NOT_FOUND_MESSAGE
                sources = chunks if answered else []
                confidence = max((c["score"] for c in chunks), default=0.0) if answered else None

                st.markdown(answer)
                if confidence is not None:
                    st.progress(min(confidence, 1.0), text=f"Confidence: {confidence:.0%}")
                show_sources(sources)

                st.session_state.messages.append(
                    {"role": "assistant", "content": answer, "sources": sources,
                     "confidence": min(confidence, 1.0) if confidence is not None else None}
                )
                log_query(query, selected, confidence, answered)

            except (ValueError, VectorStoreError, GenerationError) as e:
                st.error(str(e))