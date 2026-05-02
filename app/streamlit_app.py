"""Streamlit chat UI for the local Wikipedia RAG assistant.

Run with::

    streamlit run app/streamlit_app.py

Features
--------
- Streaming token output from the local LLM.
- Per-message metadata: classifier verdict, stores searched, latency,
  Top-K, and retrieved-chunk count.
- Retrieved-context expander with score, source, and chunk text.
- One-click example queries grouped by category (people / places /
  mixed / failure cases).
- Index browser: list every ingested entity, see its chunk count and
  preview the first chunk.
- Settings sidebar for Top-K and "show context" toggle.
- Conversation export as JSON or Markdown.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st  # noqa: E402

from config import (  # noqa: E402
    DEFAULT_TOP_K,
    EMBED_MODEL,
    LLM_MODEL,
    MAX_TOP_K,
    PEOPLE,
    PLACES,
)
from src.embedder import OllamaError, ping  # noqa: E402
from src import generator as gen  # noqa: E402
from src.rag import RAGPipeline  # noqa: E402


# ---------------------------------------------------------------------------
# Example queries grouped by category, lifted directly from the brief so the
# user can validate the assignment requirements with one click.
# ---------------------------------------------------------------------------
EXAMPLE_QUERIES: dict[str, list[str]] = {
    "People": [
        "Who was Albert Einstein and what is he known for?",
        "What did Marie Curie discover?",
        "Why is Nikola Tesla famous?",
        "Compare Lionel Messi and Cristiano Ronaldo.",
        "What is Frida Kahlo known for?",
    ],
    "Places": [
        "Where is the Eiffel Tower located?",
        "Why is the Great Wall of China important?",
        "What is Machu Picchu?",
        "What was the Colosseum used for?",
        "Where is Mount Everest?",
    ],
    "Mixed": [
        "Which famous place is located in Turkey?",
        "Which person is associated with electricity?",
        "Compare Albert Einstein and Nikola Tesla.",
        "Compare the Eiffel Tower and the Statue of Liberty.",
    ],
    "Failure cases": [
        "Who is the president of Mars?",
        "Tell me about a random unknown person John Doe.",
    ],
}


# ---------------------------------------------------------------------------
# Pipeline construction is expensive (loads two NumPy matrices) so cache it
# across reruns.  st.cache_resource is the canonical pattern.
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading vector indices...")
def get_pipeline() -> RAGPipeline:
    return RAGPipeline()


def init_state() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "show_context" not in st.session_state:
        st.session_state.show_context = True
    if "top_k" not in st.session_state:
        st.session_state.top_k = DEFAULT_TOP_K
    if "pending_query" not in st.session_state:
        st.session_state.pending_query = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _latency_emoji(seconds: float) -> str:
    if seconds < 4:
        return "fast"
    if seconds < 12:
        return "ok"
    return "slow"


def _conversation_to_markdown(messages: list[dict]) -> str:
    lines: list[str] = ["# Local Wikipedia RAG -- Chat Transcript", ""]
    for msg in messages:
        role = "User" if msg["role"] == "user" else "Assistant"
        lines.append(f"## {role}")
        lines.append("")
        lines.append(msg["content"])
        lines.append("")
        if msg.get("meta"):
            m = msg["meta"]
            lines.append(
                f"_classified={m['classification']} | "
                f"stores={', '.join(m['searched_stores']) or 'none'} | "
                f"chunks={m['num_chunks']} | latency={m['latency']:.2f}s_"
            )
            lines.append("")
        if msg.get("hits"):
            lines.append("### Retrieved context")
            for i, hit in enumerate(msg["hits"], 1):
                lines.append(
                    f"- **[{i}] {hit['title']}** (score `{hit['score']:.3f}`, "
                    f"type `{hit['entity_type']}`)"
                )
                lines.append("")
                lines.append("    " + hit["text"].replace("\n", "\n    "))
                lines.append("")
        lines.append("---")
        lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
def render_sidebar(pipeline: RAGPipeline) -> None:
    st.sidebar.title("Local Wikipedia RAG")
    st.sidebar.markdown(
        "Answers come from a local index of Wikipedia articles, generated "
        "by a local LLM via Ollama.  No external APIs are used."
    )

    with st.sidebar.expander("Index", expanded=True):
        summary = pipeline.index_summary()
        col_p, col_q = st.columns(2)
        col_p.metric("People chunks", summary["people_chunks"])
        col_q.metric("Places chunks", summary["places_chunks"])
        st.caption(
            f"Embedding dim: **{summary['people_dim'] or summary['places_dim']}** | "
            f"LLM: `{LLM_MODEL}` | Embedder: `{EMBED_MODEL}`"
        )
        if not pipeline.is_ready:
            st.warning(
                "The index is empty. Run `python -m scripts.ingest` then "
                "`python -m scripts.build_index` from a terminal."
            )

    with st.sidebar.expander("Settings", expanded=True):
        st.session_state.top_k = st.slider(
            "Top-K chunks",
            min_value=1,
            max_value=MAX_TOP_K,
            value=st.session_state.top_k,
            help="Number of chunks retrieved before being passed to the LLM.",
        )
        st.session_state.show_context = st.checkbox(
            "Show retrieved context",
            value=st.session_state.show_context,
        )

    with st.sidebar.expander("Session"):
        if st.button("Reset chat", use_container_width=True):
            st.session_state.messages = []
            st.session_state.pending_query = None
            st.rerun()
        if st.session_state.messages:
            md = _conversation_to_markdown(st.session_state.messages)
            jl = json.dumps(st.session_state.messages, indent=2, ensure_ascii=False)
            st.download_button(
                "Download chat (Markdown)",
                data=md,
                file_name="rag_chat.md",
                mime="text/markdown",
                use_container_width=True,
            )
            st.download_button(
                "Download chat (JSON)",
                data=jl,
                file_name="rag_chat.json",
                mime="application/json",
                use_container_width=True,
            )

    with st.sidebar.expander("Indexed entities"):
        tab_p, tab_pl = st.tabs(["People (20)", "Places (20)"])
        with tab_p:
            for name in PEOPLE:
                st.markdown(f"- {name}")
        with tab_pl:
            for name in PLACES:
                st.markdown(f"- {name}")


# ---------------------------------------------------------------------------
# Top of page: example-query quick buttons
# ---------------------------------------------------------------------------
def render_example_panel() -> None:
    with st.expander("Example queries (one click to send)", expanded=False):
        for category, queries in EXAMPLE_QUERIES.items():
            st.markdown(f"**{category}**")
            cols = st.columns(min(len(queries), 3))
            for i, q in enumerate(queries):
                col = cols[i % len(cols)]
                if col.button(q, key=f"ex_{category}_{i}", use_container_width=True):
                    st.session_state.pending_query = q
                    st.rerun()


# ---------------------------------------------------------------------------
# Message rendering
# ---------------------------------------------------------------------------
def render_message(msg: dict) -> None:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("meta"):
            m = msg["meta"]
            st.caption(
                f"classified `{m['classification']}` | "
                f"stores: {', '.join(m['searched_stores']) or 'none'} | "
                f"{m['num_chunks']} chunks | "
                f"{_latency_emoji(m['latency'])} {m['latency']:.2f}s"
            )
        if msg.get("hits") and st.session_state.get("show_context", True):
            with st.expander(
                f"Retrieved context ({len(msg['hits'])} chunks)", expanded=False
            ):
                for i, hit in enumerate(msg["hits"], 1):
                    st.markdown(
                        f"**[{i}] {hit['title']}** "
                        f"(score `{hit['score']:.3f}`, type `{hit['entity_type']}`)"
                    )
                    st.markdown(
                        f"> {hit['text'][:600].replace(chr(10), ' ')}"
                        f"{'...' if len(hit['text']) > 600 else ''}"
                    )
                    st.divider()


# ---------------------------------------------------------------------------
# The actual answer cycle
# ---------------------------------------------------------------------------
def answer_query(pipeline: RAGPipeline, user_input: str) -> None:
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

        retrieve_start = time.time()
        try:
            retrieval = pipeline.retrieve(
                user_input, top_k=st.session_state.top_k
            )
        except OllamaError as exc:
            placeholder.error(f"Ollama error during retrieval: {exc}")
            return
        retrieve_latency = time.time() - retrieve_start

        if not retrieval.hits:
            answer_text = (
                "I don't know. The local index does not contain relevant information."
            )
            placeholder.markdown(answer_text)
            meta = {
                "classification": retrieval.classification.query_type.value,
                "searched_stores": retrieval.searched_stores,
                "num_chunks": 0,
                "latency": retrieve_latency,
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

        gen_start = time.time()
        accumulated = ""
        try:
            for token in gen.stream(user_input, retrieval.hits):
                accumulated += token
                placeholder.markdown(accumulated + "\u2588")
        except OllamaError as exc:
            placeholder.error(f"Ollama error during generation: {exc}")
            return
        gen_latency = time.time() - gen_start
        placeholder.markdown(accumulated)

        latency = retrieve_latency + gen_latency
        meta = {
            "classification": retrieval.classification.query_type.value,
            "searched_stores": retrieval.searched_stores,
            "num_chunks": len(retrieval.hits),
            "latency": latency,
            "retrieve_latency": retrieve_latency,
            "gen_latency": gen_latency,
        }
        meta_placeholder.caption(
            f"classified `{meta['classification']}` | "
            f"stores: {', '.join(meta['searched_stores']) or 'none'} | "
            f"{meta['num_chunks']} chunks | "
            f"retrieve {retrieve_latency:.2f}s + generate {gen_latency:.2f}s = "
            f"{_latency_emoji(latency)} {latency:.2f}s"
        )

        hits_payload = [
            {
                "title": hit.entry.canonical_title or hit.entry.source_title,
                "text": hit.entry.text,
                "score": hit.score,
                "entity_type": hit.entry.entity_type,
                "url": hit.entry.url,
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
            with st.expander(
                f"Retrieved context ({len(hits_payload)} chunks)", expanded=False
            ):
                for i, hit in enumerate(hits_payload, 1):
                    title_label = (
                        f"[{hit['title']}]({hit['url']})" if hit["url"] else hit["title"]
                    )
                    st.markdown(
                        f"**[{i}] {title_label}** "
                        f"(score `{hit['score']:.3f}`, type `{hit['entity_type']}`)"
                    )
                    st.markdown(
                        f"> {hit['text'][:600].replace(chr(10), ' ')}"
                        f"{'...' if len(hit['text']) > 600 else ''}"
                    )
                    st.divider()


# ---------------------------------------------------------------------------
# Main page
# ---------------------------------------------------------------------------
def main() -> None:
    st.set_page_config(
        page_title="Local Wikipedia RAG",
        page_icon="W",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    init_state()

    pipeline = get_pipeline()

    if not ping():
        st.error(
            "Ollama runtime is not reachable at http://localhost:11434. "
            f"Start it with `ollama serve` and pull `{LLM_MODEL}` and "
            f"`{EMBED_MODEL}` before continuing."
        )

    render_sidebar(pipeline)

    st.title("Local Wikipedia RAG Assistant")
    st.caption(
        "Ask about famous people or famous places.  Answers are grounded "
        "in the local Wikipedia snapshot only -- the model will reply "
        "**I don't know.** when the answer is not in the corpus."
    )

    render_example_panel()

    # Render history first so new messages append below it visually.
    for msg in st.session_state.messages:
        render_message(msg)

    # If a quick-button populated a pending query, consume it now.
    pending = st.session_state.pop("pending_query", None) if "pending_query" in st.session_state else None
    if pending:
        answer_query(pipeline, pending)

    user_input = st.chat_input("Ask a question about a person or a place...")
    if user_input:
        answer_query(pipeline, user_input)


if __name__ == "__main__":
    main()
