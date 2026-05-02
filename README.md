# Local Wikipedia RAG Assistant

A from-scratch retrieval-augmented-generation (RAG) system that answers
questions about famous people and famous places using only **local**
resources: a local LLM via Ollama, a local embedding model, a hand-rolled
NumPy vector store, and a SQLite metadata database.

> Built for **BLG483E – Project 3**.  No external LLM API, no managed
> vector store, no LangChain.  The only third-party Python deps are
> NumPy and Streamlit.

---

## 1. Architecture at a glance

```
                +----------------------------------------------+
                |                  Streamlit / CLI             |
                +----------------------------------------------+
                                    |
                                    v
                          +--------------------+
                          |    RAGPipeline     |
                          +--------------------+
                            |                |
              +-------------+                +-----------+
              v                                          v
     +---------------+                          +-----------------+
     | HybridRetriever|<---classify (rule-based)|  query: str     |
     +---------------+                          +-----------------+
        |        |
        v        v
  +---------+ +---------+
  | People  | | Places  |    NumPy vector stores  (data/index/...)
  | store   | | store   |    cosine similarity = dot product on
  +---------+ +---------+    L2-normalised vectors
                                    |
                                    v
                         +-------------------+
                         |  Ollama generate  |   llama3.2:3b
                         +-------------------+
```

End-to-end pipeline:

1. **Ingest** – `scripts/ingest.py` pulls plain-text articles from the
   public MediaWiki API for 20 people + 20 places, saves to `data/raw/`.
2. **Chunk** – `src/chunker.py` splits documents into ~900-char overlapping
   chunks (paragraph-aware, sentence-snapping fallback).
3. **Embed + Store** – `scripts/build_index.py` calls Ollama
   (`nomic-embed-text`) and persists L2-normalised vectors via
   `src/vector_store.py` to two stores (people / places).  Article and
   chunk metadata are mirrored in SQLite.
4. **Classify + Retrieve** – `src/classifier.py` decides whether the query
   is about a person, a place, or both, using entity-name matching and
   lexical cues.  `src/retriever.py` runs **hybrid retrieval**: cosine
   similarity on the dense vectors + an IDF-weighted keyword overlap +
   a flat boost per proper-noun match.  Comparison queries split top-K
   evenly across the named entities so each subject is represented.
5. **Generate** – `src/generator.py` calls `llama3.2:3b` with a strict
   "answer only from context, otherwise say 'I don't know'" system prompt.

### Why two vector stores (Option A)?

The brief offers a choice between two stores or one store with metadata
filtering.  We picked **Option A: two stores** because:

- It expresses the routing decision in the *index layout itself*, so a
  mis-classification never silently leaks people-chunks into a places
  search and vice versa.
- Cosine search is a single matmul; doubling it once into two smaller
  matrices is essentially free at this scale (~80 chunks total).
- Adding a new entity type later (e.g. "events") is a one-line change:
  spin up another `VectorStore` with its own index path.

When the classifier is uncertain we still query *both* stores and let
ranking arbitrate, which gives us the best of both designs.

---

## 2. Prerequisites

- **macOS / Linux** (Windows works too but commands below assume bash)
- **Python 3.9+** (tested on 3.9 and 3.11)
- **Ollama**, the local LLM runtime: <https://ollama.com/download>
- ~3 GB of free disk for the model weights

---

## 3. Install

```bash
# 1. clone the repo and step into it
git clone <your-fork-url> ai_odev_3
cd ai_odev_3

# 2. create a virtual environment (recommended)
python3 -m venv .venv
source .venv/bin/activate

# 3. install Python deps
pip install -r requirements.txt
```

---

## 4. Run the local model (Ollama)

In a separate terminal start the Ollama runtime and pull both models:

```bash
ollama serve                          # leave this running
ollama pull llama3.2:3b               # answer generation (~2 GB)
ollama pull nomic-embed-text          # embeddings (~270 MB)
```

You can swap the LLM via `RAG_LLM_MODEL=phi3` or `RAG_LLM_MODEL=mistral`
without changing any code.

---

## 5. Ingest Wikipedia data

```bash
python -m scripts.ingest          # ~1 min, polite 0.5 s/page rate-limit
```

This downloads plain-text bodies for the 20 people and 20 places listed
in `config.py` (which includes every required entity from the brief)
into `data/raw/`.

Re-run with `--refresh` to force a fresh download.

---

## 6. Build the vector index

```bash
python -m scripts.build_index     # depends on Ollama embeddings
```

Outputs:

- `data/index/people/{vectors.npy,meta.jsonl}`
- `data/index/places/{vectors.npy,meta.jsonl}`
- `data/db/rag.sqlite` – article + chunk metadata

Re-run with `--reset` to wipe and rebuild from scratch.

---

## 7. Start the assistant

### Option A – Streamlit web UI

```bash
streamlit run app/streamlit_app.py
```

Open the URL Streamlit prints (default `http://localhost:8501`).  You'll
get a chat interface with:

- streaming token output
- adjustable Top-K and "show context" toggle in the sidebar
- a **Reset chat** button that clears the conversation (the index is
  unchanged)
- per-answer metadata: classification, stores searched, latency

### Option B – CLI

```bash
python -m app.cli
```

Slash commands:

| Command           | Effect                                       |
|-------------------|----------------------------------------------|
| `/context on/off` | Show retrieved chunks alongside the answer   |
| `/reset`          | Clear chat history (index unchanged)         |
| `/stats`          | Print index statistics                       |
| `/help`           | Show the command list                        |
| `/exit`           | Quit                                         |

---

## 8. Example queries

Direct from the brief:

**People**
- *Who was Albert Einstein and what is he known for?*
- *What did Marie Curie discover?*
- *Why is Nikola Tesla famous?*
- *Compare Lionel Messi and Cristiano Ronaldo.*
- *What is Frida Kahlo known for?*

**Places**
- *Where is the Eiffel Tower located?*
- *Why is the Great Wall of China important?*
- *What is Machu Picchu?*
- *What was the Colosseum used for?*
- *Where is Mount Everest?*

**Mixed**
- *Which famous place is located in Turkey?*  (→ Hagia Sophia)
- *Which person is associated with electricity?*  (→ Tesla)
- *Compare Albert Einstein and Nikola Tesla.*
- *Compare the Eiffel Tower and the Statue of Liberty.*

**Failure cases (must reply "I don't know")**
- *Who is the president of Mars?*
- *Tell me about a random unknown person John Doe.*

---

## 9. Repository layout

```
ai_odev_3/
├── README.md                # this file
├── Product_prd.md           # product requirements document
├── recommendation.md        # production deployment recommendations
├── requirements.txt         # only numpy + streamlit
├── config.py                # paths, models, entity lists, knobs
├── src/
│   ├── ingest.py            # MediaWiki API ingestion (urllib only)
│   ├── chunker.py           # paragraph + sentence chunker
│   ├── embedder.py          # Ollama embeddings client
│   ├── vector_store.py      # NumPy vector store + persistence
│   ├── classifier.py        # rule-based query router
│   ├── retriever.py         # store routing + top-k cosine search
│   ├── generator.py         # Ollama generate / stream client
│   ├── rag.py               # high-level pipeline orchestrator
│   └── db.py                # SQLite metadata schema + helpers
├── scripts/
│   ├── ingest.py            # python -m scripts.ingest
│   └── build_index.py       # python -m scripts.build_index
├── app/
│   ├── cli.py               # python -m app.cli
│   └── streamlit_app.py     # streamlit run app/streamlit_app.py
└── data/
    ├── raw/{people,places}/ # cached Wikipedia text
    ├── index/{people,places}/ # vectors.npy + meta.jsonl
    └── db/rag.sqlite          # SQLite metadata
```

---

## 10. Configuration knobs

All settings live in `config.py` and are also overridable via env vars:

| Env var               | Default              | Notes                                  |
|-----------------------|----------------------|----------------------------------------|
| `OLLAMA_HOST`         | `http://localhost:11434` | Where to find Ollama               |
| `RAG_LLM_MODEL`       | `llama3.2:3b`        | Try `phi3`, `mistral` for speed/quality |
| `RAG_EMBED_MODEL`     | `nomic-embed-text`   | Any local embedding model in Ollama    |
| `RAG_LLM_TEMPERATURE` | `0.1`                | Low for grounded factual answers       |
| `RAG_LLM_NUM_CTX`     | `4096`               | Increase if Top-K * chunk_size grows   |

Chunk size, overlap, and Top-K are at the top of `config.py`.

---

## 11. Troubleshooting

| Symptom                                     | Fix                                                         |
|---------------------------------------------|-------------------------------------------------------------|
| `Cannot reach Ollama at http://localhost:11434` | Run `ollama serve` in another terminal                  |
| `model 'llama3.2:3b' not found`             | `ollama pull llama3.2:3b`                                   |
| Streamlit re-loads the index every prompt   | Restart the app once; `@st.cache_resource` will memoise it |
| Embedding dim mismatch on build             | `python -m scripts.build_index --reset`                     |
| `urllib.error.HTTPError: 429`               | Wikipedia rate-limit – the script already retries; wait    |

---

## 12. Demo video

A 5-minute walkthrough is available at: **<https://youtu.be/REPLACE_ME>**
*(replace with your unlisted YouTube / Loom link before submitting).*

The video covers system overview, ingestion + indexing, live chat,
technical decisions, tradeoffs, and possible improvements.

---

## 13. License

MIT – feel free to fork and reuse.
