# Production Deployment Recommendations

This document explains how the Local Wikipedia RAG project would change
if we wanted to take it from a single-laptop demo to a production-grade
service.  It is meant as a thinking exercise, not a refactoring plan –
the demo intentionally keeps everything local and simple.

---

## 1. Where the demo would break in production

| Concern                | Demo today                                              | Why it would break at scale                                                                  |
|------------------------|---------------------------------------------------------|----------------------------------------------------------------------------------------------|
| Concurrency            | Single-process Streamlit / CLI                           | A second user blocks on the first user's Ollama generation.                                  |
| State persistence      | Local NumPy `.npy` + SQLite                              | Filesystem state on a pod is ephemeral; index disappears on restart.                         |
| Index size             | ~80 chunks, fits in RAM                                  | Wikipedia (6 M articles) won't; brute-force matmul becomes O(seconds) per query.             |
| Cold start             | Re-runs `ingest` + `build_index` manually                | Operators want zero-touch redeploys.                                                         |
| Observability          | `print` statements                                       | No way to tell whether a slow answer is retrieval, generation, or network.                   |
| Failure mode           | Crashes if Ollama is down                                | Production needs graceful degradation and alerting.                                          |
| Trust                  | LLM is asked nicely to ground answers                    | Users will paste prompts that try to bypass the system prompt.                               |

---

## 2. Recommended target architecture

```
              +-----------------------------+
              |   Frontend (Next.js / web)  |
              +--------------+--------------+
                             |
                       (HTTPS, JWT)
                             v
              +-----------------------------+
              |  API Gateway / FastAPI      |
              |  - rate limiting            |
              |  - auth                     |
              |  - prompt-injection filter  |
              +--+--------+--------+--------+
                 |        |        |
                 v        v        v
         +-----------+ +------+ +-----------------+
         | Retriever | | Cache| | Generator pool  |
         | service   | |(redis| | (vLLM / TGI)    |
         +-----+-----+ +------+ +--------+--------+
               |                          |
               v                          v
        +-------------+            +-------------+
        | Vector DB   |            | Object store|
        | (pgvector / |            | (S3) for    |
        | Qdrant)     |            | raw articles|
        +-------------+            +-------------+
               ^
               |
        +-------------+
        | Ingestion   |
        | worker      |
        | (Airflow /  |
        |  Temporal)  |
        +-------------+
```

### 2.1 LLM serving

- Replace single-process Ollama with **vLLM** or **Text-Generation-Inference**
  behind a load balancer.  Both support continuous batching, which is the
  single biggest throughput win for a RAG workload (many short prompts).
- Pin a 7B–13B instruction-tuned model (e.g. Llama-3.1-8B-Instruct,
  Mistral-7B-Instruct).  3B is fine for a demo but trips on multi-hop
  comparison questions.
- Run the model on a GPU node pool (A10G or L4 are usually enough for
  8B-class models with quantisation).
- Cache responses keyed by `(retrieved_chunk_ids, query_normalised)` in
  Redis with a TTL of a few minutes.  Hit rate is high for grading
  scenarios where the same questions repeat.

### 2.2 Embeddings

- Move embeddings to a dedicated service (separate pod), again behind
  load balancing, so query embedding doesn't compete with answer
  generation for GPU.
- For a much larger corpus, consider switching to a stronger embedding
  model (e.g. `bge-large-en-v1.5`, `mxbai-embed-large`).

### 2.3 Vector store

- **pgvector** if you already run Postgres – simplest, transactional,
  metadata filtering for free.
- **Qdrant** or **Weaviate** if you need >10 M vectors, hybrid search,
  or multi-tenant collections.
- Either way, keep the *abstraction*.  The current `VectorStore` class
  should grow a `BaseVectorStore` interface and a `PgVectorStore`
  subclass; the rest of the pipeline doesn't change.
- Build approximate-nearest-neighbour indexes (HNSW) once you cross
  ~100 k vectors, so latency stops scaling linearly with corpus size.

### 2.4 Ingestion

- Move from "run once at install time" to a scheduled **Airflow** /
  **Temporal** workflow that:
    1. Pulls a list of titles from a content team's spreadsheet / DB.
    2. Fetches Wikipedia articles, computes a hash, skips unchanged
       articles.
    3. Re-chunks and re-embeds *only the deltas*.
    4. Writes new vectors to the staging collection, swaps to live
       atomically.
- Store raw article text in S3 (versioned) so we can rebuild any past
  state of the index.

### 2.5 API layer

- Replace Streamlit with a **FastAPI** backend exposing
    - `POST /ask`  – body `{query, top_k}`, returns `{answer, sources, latency}`
    - `GET  /healthz`
    - `GET  /metrics` (Prometheus)
- Add a thin **Next.js** frontend so the UX is browser-native and
  CDN-cacheable.
- Authentication via the institutional SSO (SAML / OIDC).

### 2.6 Safety and grounding

- Add a **prompt-injection filter** on inbound queries: reject queries
  that try to override the system prompt ("ignore the above instructions
  and ...").  Even simple keyword + heuristic checks help.
- Run a **groundedness classifier** on the generated answer: take the
  answer + retrieved chunks and ask a small auxiliary model "is this
  answer supported by the chunks? yes/no".  If `no`, fall back to
  *I don't know.*
- Surface citations *in the UI* (clickable footnote markers), not just
  inline brackets, to give the user an audit trail.

### 2.7 Observability

- **Tracing**: OpenTelemetry around `retrieve` and `generate` stages so
  we can visualise per-stage latency per request.
- **Metrics**: histogram of `embedding_latency_seconds`,
  `retrieval_latency_seconds`, `generation_latency_seconds`,
  `top_k_score_first`, `idk_rate`.  Alert when `idk_rate` spikes (often
  signals a stale or broken index).
- **Eval harness**: run the acceptance test plan from Product_prd.md
  against every release artifact; fail the deploy if score regresses.

### 2.8 Cost management

- Quantise the LLM (Q4_K_M GGUF or AWQ) – fits 8B in ≤ 8 GB VRAM.
- Use **streaming responses** so the user sees first tokens fast and the
  perceived latency drops by ~70 %.
- Apply per-user **rate limits** and **token budgets**; abusive prompts
  are surprisingly common even on internal tools.

---

## 3. Migration path from this demo

A reasonable rollout, smallest blast radius first:

1. **Containerise** today's code (Ollama in one container, Streamlit in
   another, mounted volumes for `data/`).
2. **Replace Streamlit with FastAPI** + a static React frontend; keep
   the same `RAGPipeline` class on the backend so the swap is invisible.
3. **Move the vector store to pgvector**, behind the existing
   `VectorStore` interface.  Backfill from the NumPy index.
4. **Move ingestion** into a scheduled job, with delta detection.
5. **Add tracing, metrics, eval harness**, then deploy a managed LLM
   (vLLM on a GPU node) and finally retire the local Ollama.

Each step is independently revertible.

---

## 4. What I would *not* change

- Two-store architecture (people / places).  It scales fine: at 100
  entity types we'd just rename to "collections" and let the metadata
  filter handle it, but the design stays the same.
- Strict "answer only from context, otherwise I don't know" prompt.  In
  production this is a feature, not a limitation.
- Treating retrieved chunks as the source-of-truth surface.  Even if the
  generation model gets stronger, the retrieval contract is what users
  trust.

---

## 5. Open questions

- **Personalisation**: should the corpus be filtered per user (e.g. a
  course only sees the entities its syllabus covers)?  That would push
  toward multi-tenant collections and per-user API keys.
- **Multilingual**: do we need TR / ES / DE Wikipedia mirrors?  If yes,
  the embedding model has to change to a multilingual one (e.g.
  `multilingual-e5-large`).
- **Continuous evaluation**: who owns the gold question set?  Without a
  living eval set, retrieval quality silently degrades as the corpus grows.

These are decisions for product, not engineering, and should be settled
before serious production investment.
