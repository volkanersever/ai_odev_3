"""Streamlit chat UI for the local Wikipedia RAG assistant.

Run with::

    streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st  # noqa: E402

from config import DEFAULT_TOP_K, MAX_TOP_K  # noqa: E402
from src.embedder import OllamaError, ping  # noqa: E402
from src.rag import RAGPipeline  # noqa: E402


@st.cache_resource(show_spinner=True)
def get_pipeline() -> RAGPipeline:
    return RAGPipeline()


def init_state() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "show_context" not in st.session_state:
        st.session_state.show_context = True
    if "top_k" not in st.session_state:
        st.session_state.top_k = DEFAULT_TOP_K


def render_sidebar(pipeline: RAGPipeline) -> None:
    st.sidebar.title("Local Wikipedia RAG")
    st.sidebar.markdown(
        "Answers come from a local index of Wikipedia articles, generated "
        "by a local LLM via Ollama.  No external APIs are used."
    )

    st.sidebar.subheader("Index")
    summary = pipeline.index_summary()
    st.sidebar.write(
        f"**People chunks**: {summary['people_chunks']}  \n"
        f"**Places chunks**: {summary['places_chunks']}  \n"
        f"**Embedding dim**: {summary['people_dim'] or summary['places_dim']}"
    )
    if not pipeline.is_ready:
        st.sidebar.warning(
            "The index is empty. Run `python -m scripts.ingest` then "
            "`python -m scripts.build_index` from a terminal."
        )

    st.sidebar.subheader("Settings")
    st.session_state.top_k = st.sidebar.slider(
        "Top-K chunks",
        min_value=1,
        max_value=MAX_TOP_K,
        value=st.session_state.top_k,
    )
    st.session_state.show_context = st.sidebar.checkbox(
        "Show retrieved context",
        value=st.session_state.show_context,
    )

    st.sidebar.subheader("Session")
    if st.sidebar.button("Reset chat"):
        st.session_state.messages = []
        st.rerun()


def render_message(msg: dict) -> None:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("meta"):
            meta = msg["meta"]
            st.caption(
                f"classified={meta['classification']} | "
                f"stores={', '.join(meta['searched_stores']) or 'none'} | "
                f"chunks={meta['num_chunks']} | latency={meta['latency']:.2f}s"
            )
        if msg.get("hits") and st.session_state.get("show_context", True):
            with st.expander("Retrieved context", expanded=False):
                for i, hit in enumerate(msg["hits"], 1):
                    st.markdown(
                        f"**[{i}] {hit['title']}** "
                        f"(score `{hit['score']:.3f}`, type `{hit['entity_type']}`)  \n"
                        f"{hit['text']}"
                    )


def main() -> None:
    st.set_page_config(page_title="Local Wikipedia RAG", page_icon="W", layout="wide")
    init_state()

    if not ping():
        st.error(
            "Ollama runtime is not reachable at http://localhost:11434. "
            "Start it with `ollama serve` and pull `llama3.2:3b` and "
            "`nomic-embed-text` before continuing."
        )

    pipeline = get_pipeline()
    render_sidebar(pipeline)

    st.title("Local Wikipedia RAG Assistant")
    st.caption(
        "Ask about famous people or famous places.  Answers are grounded in "
        "the local Wikipedia snapshot only -- the model will reply 'I don't "
        "know' when the answer is not in the corpus."
    )

    for msg in st.session_state.messages:
        render_message(msg)

    user_input = st.chat_input("Ask a question...")
    if not user_input:
        return

    st.session_state.messages.append({"role": "user", "content": user_input})
    render_message(st.session_state.messages[-1])

    if not pipeline.is_ready:
        reply = (
            "The index is empty. Please run the ingestion and build steps "
            "from the terminal first."
        )
        st.session_state.messages.append({"role": "assistant", "content": reply})
        render_message(st.session_state.messages[-1])
        return

    with st.chat_message("assistant"):
        placeholder = st.empty()
        meta_placeholder = st.empty()

        try:
            retrieval = pipeline.retrieve(user_input, top_k=st.session_state.top_k)
        except OllamaError as exc:
            placeholder.error(f"Ollama error during retrieval: {exc}")
            return

        if not retrieval.hits:
            answer_text = "I don't know. The local index does not contain relevant information."
            placeholder.markdown(answer_text)
            meta = {
                "classification": retrieval.classification.query_type.value,
                "searched_stores": retrieval.searched_stores,
                "num_chunks": 0,
                "latency": 0.0,
            }
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": answer_text,
                    "meta": meta,
                    "hits": [],
                }
            )
            return

        from src import generator as gen

        start = time.time()
        accumulated = ""
        try:
            for token in gen.stream(user_input, retrieval.hits):
                accumulated += token
                placeholder.markdown(accumulated + "\u2588")
        except OllamaError as exc:
            placeholder.error(f"Ollama error during generation: {exc}")
            return
        latency = time.time() - start
        placeholder.markdown(accumulated)

        meta = {
            "classification": retrieval.classification.query_type.value,
            "searched_stores": retrieval.searched_stores,
            "num_chunks": len(retrieval.hits),
            "latency": latency,
        }
        meta_placeholder.caption(
            f"classified={meta['classification']} | "
            f"stores={', '.join(meta['searched_stores']) or 'none'} | "
            f"chunks={meta['num_chunks']} | latency={latency:.2f}s"
        )

        hits_payload = [
            {
                "title": hit.entry.canonical_title or hit.entry.source_title,
                "text": hit.entry.text,
                "score": hit.score,
                "entity_type": hit.entry.entity_type,
            }
            for hit in retrieval.hits
        ]
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": accumulated,
                "meta": meta,
                "hits": hits_payload,
            }
        )

        if st.session_state.show_context:
            with st.expander("Retrieved context", expanded=False):
                for i, hit in enumerate(hits_payload, 1):
                    st.markdown(
                        f"**[{i}] {hit['title']}** "
                        f"(score `{hit['score']:.3f}`, type `{hit['entity_type']}`)  \n"
                        f"{hit['text']}"
                    )


if __name__ == "__main__":
    main()
