# Product Requirements Document — Local Wikipedia RAG Assistant

> Audience: an AI engineer (or autonomous coding agent) tasked with
> reproducing this project from scratch.  This document describes the
> *what* and *why*; the README describes the *how*.

---

## 1. Problem statement

Course assistants and students need a privacy-respecting, offline-capable
question-answering system that can reliably answer factual questions
about famous people and famous places, drawing only on a curated subset
of Wikipedia.

Existing chat assistants (ChatGPT, Gemini, Claude) require external API
keys, send prompts off-device, and may hallucinate when knowledge cutoffs
fail.  We want a system that:

- Runs **entirely** on a student laptop with no external calls.
- Returns **grounded** answers backed by retrieved Wikipedia text.
- Refuses to answer when the corpus has no information (no hallucination).

## 2. Goals (success criteria)

| #   | Goal                                                                      | Measurement                                       |
|-----|---------------------------------------------------------------------------|---------------------------------------------------|
| G1  | Ingest 20 famous people and 20 famous places from Wikipedia               | `data/raw/people` ≥ 20 files, `places` ≥ 20 files |
| G2  | Run end-to-end on `localhost` only                                        | No outbound network call after ingestion          |
| G3  | Correctly answer the assignment's example questions                       | Manual eval, see §10                              |
| G4  | Reply **"I don't know"** for the failure-case questions                   | Manual eval, see §10                              |
| G5  | Deliver responses in < 30 s on a 16 GB MacBook                            | Latency log per response                          |
| G6  | Make the system runnable from README alone, without extra guidance        | Instructor walkthrough                            |

## 3. Non-goals

- General-purpose chit-chat or coding help.
- Multi-turn reasoning over conversation history (out of scope; chat
  history is shown but not fed back into retrieval).
- Real-time Wikipedia updates – snapshots are static after ingestion.
- Multilingual support – English Wikipedia only.

## 4. Personas

| Persona              | Need                                                                   |
|----------------------|------------------------------------------------------------------------|
| **Student**          | Demo the project locally, ask grading-style questions.                 |
| **Instructor**       | Run the README from scratch, validate the failure cases.               |
| **Developer (you)**  | Re-run ingestion, swap the LLM, extend to a new entity type.           |

## 5. Functional requirements

### 5.1 Ingestion

- **F-1**: Pull plain-text article body for each title in `config.PEOPLE`
  and `config.PLACES` using the public MediaWiki API.
- **F-2**: Cache successful downloads in `data/raw/{people,places}/<slug>.txt`
  and a sibling `<slug>.json` with metadata.
- **F-3**: Re-runs MUST be idempotent: cached files are reused unless
  `--refresh` is passed.
- **F-4**: Must follow Wikipedia redirects (so "Pyramids of Giza" resolves
  to "Giza pyramid complex").

### 5.2 Chunking

- **F-5**: Split each article into chunks of ≤ 900 characters with ~150
  characters of overlap.
- **F-6**: Chunking SHOULD respect paragraph and sentence boundaries; only
  fall back to hard cuts when a single sentence exceeds the budget.
- **F-7**: Prepend the canonical title to the first chunk so the embedding
  has a strong document-level signal.

### 5.3 Embedding & storage

- **F-8**: Embed every chunk using a **local** model
  (`nomic-embed-text` via Ollama by default).
- **F-9**: Persist vectors as a single `vectors.npy` matrix per store,
  paired with a `meta.jsonl` for chunk metadata.
- **F-10**: Implement **two stores** (Option A): one for people, one for
  places.  Justification belongs in README.md §1.
- **F-11**: Mirror article + chunk metadata in SQLite (`data/db/rag.sqlite`)
  so administrative tasks (counts, deletions) don't require loading
  the vectors.
- **F-12**: Support incremental adds and full reset (`--reset`) of the index.

### 5.4 Query classification

- **F-13**: Classify every query into `{PERSON, PLACE, BOTH}` before
  retrieval using the rules in `src/classifier.py`:
    1. Direct entity-name match (highest priority).
    2. Lexical cues (`who/where/born/located/...`).
    3. Default to `BOTH` when evidence is weak.
- **F-14**: Comparison queries that name two entities of different types
  must classify as `BOTH`.

### 5.5 Retrieval

- **F-15**: **Hybrid retrieval**.  The base score is cosine similarity
  (dot product on normalised vectors).  On top of that the retriever adds
  (a) an IDF-weighted keyword overlap and (b) a flat boost per proper
  noun in the query that appears in the chunk.  This recovers cases where
  the dense embedding is too diffuse -- e.g. *"Which famous place is
  located in Turkey"* needs a strong "Turkey" cue to surface Hagia Sophia.
- **F-16**: Top-K configurable (default 5, max 10 in UI).
- **F-17**: When the classifier identifies a specific entity by name, the
  retriever applies a **strict** title filter so unrelated articles can't
  win on partial similarity, and skips the keyword re-rank.
- **F-18**: When the classifier returns `BOTH`, query both stores and
  merge results by score before truncating to Top-K.
- **F-19**: For comparison queries that name two entities of the same
  type, allocate Top-K equally across the named entities so neither
  subject is drowned by the other's better cosine alignment.

### 5.6 Generation

- **F-20**: Use a **local** LLM via Ollama (`llama3.2:3b` default;
  swappable to `phi3` or `mistral` via env var).
- **F-21**: System prompt enforces:
    - Answer only from supplied context.
    - Reply *"I don't know."* if the context lacks the answer; the
      reply must be exactly that string, with no surrounding text.
    - Cite source titles in square brackets.
- **F-22**: Temperature ≤ 0.2 to minimise creative hallucination.
- **F-23**: Streaming output is supported via `src/generator.stream`.

### 5.7 Chat interface

- **F-24**: Streamlit web UI with:
    - chat-style message stream
    - sidebar for Top-K, "show context" toggle, and reset
    - per-answer metadata footer (classification, latency, store(s))
- **F-25**: CLI with parity for the core flow plus `/context`, `/reset`,
  `/stats`, `/help`, `/exit` slash-commands.

## 6. Non-functional requirements

| #    | Requirement                                                                |
|------|----------------------------------------------------------------------------|
| NF-1 | No third-party dependency exceeds NumPy + Streamlit (and Ollama runtime).  |
| NF-2 | Cold start (load index, embed query, retrieve) ≤ 2 s on M-series Mac.      |
| NF-3 | Fits in 4 GB free RAM with the 3B-parameter LLM loaded.                    |
| NF-4 | All paths configurable so the project is portable across machines.         |
| NF-5 | Code passes `python -m py_compile` on Python 3.9+.                         |

## 7. System overview

See README §1 for the architecture diagram.  Key data flows:

1. **Build-time**: `Wikipedia API → ingest → chunker → embedder → vector_store + db`.
2. **Query-time**: `query → classifier → retriever → generator → answer`.

Two physical stores (`data/index/people` and `data/index/places`)
mirror the `entity_type` axis used in the classifier.

## 8. Data model

### 8.1 Vector store on disk

- `vectors.npy` – `float32` matrix of shape `(N, D)`; rows are
  L2-normalised.
- `meta.jsonl` – one JSON object per row, with `id`, `text`,
  `source_title`, `canonical_title`, `url`, `entity_type`, `chunk_index`.

### 8.2 SQLite schema (`data/db/rag.sqlite`)

```sql
CREATE TABLE articles (
    title TEXT PRIMARY KEY, canonical_title TEXT, url TEXT,
    entity_type TEXT CHECK (entity_type IN ('person','place')),
    char_count INTEGER, chunk_count INTEGER,
    fetched_at TEXT DEFAULT (datetime('now'))
);
CREATE TABLE chunks (
    id TEXT PRIMARY KEY, article_title TEXT REFERENCES articles(title)
        ON DELETE CASCADE,
    chunk_index INTEGER, entity_type TEXT, text TEXT
);
```

## 9. APIs / interfaces

### 9.1 Internal Python API

```python
from src.rag import RAGPipeline
pipeline = RAGPipeline()
answer = pipeline.answer("What did Marie Curie discover?")
print(answer.text, answer.classification.query_type, answer.latency_seconds)
```

### 9.2 External APIs consumed

- **Wikipedia REST API** (`en.wikipedia.org/w/api.php`) – read-only.
- **Ollama HTTP API** (`localhost:11434`) – `/api/embeddings` and
  `/api/generate`.

No other external service is contacted during runtime.

## 10. Acceptance test plan

Run after `ingest` and `build_index` complete.  All must pass.

| Test ID  | Query                                                     | Expected                                                       |
|----------|-----------------------------------------------------------|----------------------------------------------------------------|
| T-P-01   | *Who was Albert Einstein and what is he known for?*       | Mentions theoretical physics, relativity, Nobel Prize          |
| T-P-02   | *What did Marie Curie discover?*                          | Mentions polonium and/or radium and radioactivity              |
| T-P-03   | *Why is Nikola Tesla famous?*                             | Mentions AC electrical system / induction motor                |
| T-P-04   | *Compare Lionel Messi and Cristiano Ronaldo.*             | Both names appear; some attribute contrast                     |
| T-P-05   | *What is Frida Kahlo known for?*                          | Mentions Mexican painter and self-portraits                    |
| T-PL-01  | *Where is the Eiffel Tower located?*                      | Paris / France                                                 |
| T-PL-02  | *Why is the Great Wall of China important?*               | Defensive purpose / dynasties                                  |
| T-PL-03  | *What is Machu Picchu?*                                   | Inca citadel in Peru                                           |
| T-PL-04  | *What was the Colosseum used for?*                        | Gladiator contests / Roman entertainment                       |
| T-PL-05  | *Where is Mount Everest?*                                 | Himalayas / Nepal–Tibet border                                 |
| T-MX-01  | *Which famous place is located in Turkey?*                | Hagia Sophia                                                   |
| T-MX-02  | *Which person is associated with electricity?*            | Nikola Tesla (or Edison rivalry mention)                       |
| T-MX-03  | *Compare Albert Einstein and Nikola Tesla.*               | Both names referenced from people store                         |
| T-MX-04  | *Compare the Eiffel Tower and the Statue of Liberty.*     | Both names referenced from places store                         |
| T-FAIL-01| *Who is the president of Mars?*                           | "I don't know" or unambiguous refusal                          |
| T-FAIL-02| *Tell me about a random unknown person John Doe.*         | "I don't know" or unambiguous refusal                          |

## 11. Risks and mitigations

| Risk                                              | Impact          | Mitigation                                                    |
|---------------------------------------------------|-----------------|---------------------------------------------------------------|
| Wikipedia title drift (e.g. page renamed)         | Missing entity  | Use redirects + ingest cache + manual override in `config.py` |
| Ollama not installed / wrong model name           | System unusable | Pre-flight `ping()`, clear error messages in CLI/UI           |
| Hallucination by the small 3B model               | Wrong answers   | Strict system prompt, low temperature, "I don't know" rule    |
| Embedding dim mismatch after model swap           | Build crash     | `--reset` flag and explicit dim assertion in `VectorStore`    |
| Long Wikipedia articles → context overflow        | Truncated facts | Chunk + Top-K retrieval + `num_ctx=4096`                      |

## 12. Out-of-scope features (potential extensions)

- Multi-document summarisation across all 40 articles.
- Per-paragraph citations in the answer (currently per-source).
- Re-ranking of retrieved chunks with a small cross-encoder.
- Fine-grained access control or multi-user sessions.
- Multilingual ingestion / translation pipeline.

## 13. Glossary

- **RAG** – Retrieval-Augmented Generation.  Combine a retrieval step with
  an LLM so the answer is grounded in supplied text rather than the
  model's parametric memory.
- **Embedding** – Dense numeric representation of text where similar
  meanings yield close vectors.
- **Top-K** – The number of most-similar chunks pulled from the index.
- **Cosine similarity** – Cosine of the angle between two vectors;
  identical to a dot product when both vectors are L2-normalised.
