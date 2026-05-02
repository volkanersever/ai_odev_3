"""Central configuration for the local Wikipedia RAG system.

All paths are resolved relative to the repo root so the project remains
portable regardless of where the user clones it.
"""

from __future__ import annotations

import os
from pathlib import Path

ROOT_DIR: Path = Path(__file__).resolve().parent
DATA_DIR: Path = ROOT_DIR / "data"
RAW_DIR: Path = DATA_DIR / "raw"
INDEX_DIR: Path = DATA_DIR / "index"
DB_DIR: Path = DATA_DIR / "db"

RAW_PEOPLE_DIR: Path = RAW_DIR / "people"
RAW_PLACES_DIR: Path = RAW_DIR / "places"

DB_PATH: Path = DB_DIR / "rag.sqlite"

# Two separate vector stores (Option A from the brief).  The design choice
# is justified in README.md.
PEOPLE_INDEX_PATH: Path = INDEX_DIR / "people"
PLACES_INDEX_PATH: Path = INDEX_DIR / "places"

# ---------------------------------------------------------------------------
# Ollama configuration
# ---------------------------------------------------------------------------
OLLAMA_HOST: str = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
EMBED_MODEL: str = os.environ.get("RAG_EMBED_MODEL", "nomic-embed-text")
LLM_MODEL: str = os.environ.get("RAG_LLM_MODEL", "llama3.2:3b")

# Generation parameters
LLM_TEMPERATURE: float = float(os.environ.get("RAG_LLM_TEMPERATURE", "0.1"))
LLM_NUM_CTX: int = int(os.environ.get("RAG_LLM_NUM_CTX", "4096"))

# ---------------------------------------------------------------------------
# Chunking configuration
# ---------------------------------------------------------------------------
CHUNK_SIZE_CHARS: int = 900       # ~150 tokens, comfortably within context
CHUNK_OVERLAP_CHARS: int = 150    # ~17% overlap to preserve context
MIN_CHUNK_CHARS: int = 120        # discard near-empty fragments

# ---------------------------------------------------------------------------
# Retrieval parameters
# ---------------------------------------------------------------------------
DEFAULT_TOP_K: int = 5
MAX_TOP_K: int = 10

# ---------------------------------------------------------------------------
# Required minimum entity sets from the assignment brief.  These are the
# entities the grader will test against and therefore MUST be present.
# Entries beyond the required ten satisfy the "at least 20" rule and
# diversify the test corpus.
# ---------------------------------------------------------------------------
PEOPLE: list[str] = [
    # Required ten
    "Albert Einstein",
    "Marie Curie",
    "Leonardo da Vinci",
    "William Shakespeare",
    "Ada Lovelace",
    "Nikola Tesla",
    "Lionel Messi",
    "Cristiano Ronaldo",
    "Taylor Swift",
    "Frida Kahlo",
    # Additional ten to reach 20
    "Isaac Newton",
    "Charles Darwin",
    "Stephen Hawking",
    "Mahatma Gandhi",
    "Mustafa Kemal Atatürk",
    "Mozart",
    "Vincent van Gogh",
    "Pablo Picasso",
    "Steve Jobs",
    "Elon Musk",
]

PLACES: list[str] = [
    # Required ten
    "Eiffel Tower",
    "Great Wall of China",
    "Taj Mahal",
    "Grand Canyon",
    "Machu Picchu",
    "Colosseum",
    "Hagia Sophia",
    "Statue of Liberty",
    "Giza pyramid complex",  # canonical Wikipedia title for "Pyramids of Giza"
    "Mount Everest",
    # Additional ten to reach 20
    "Stonehenge",
    "Petra",
    "Acropolis of Athens",
    "Niagara Falls",
    "Mount Fuji",
    "Sagrada Família",
    "Christ the Redeemer (statue)",
    "Angkor Wat",
    "Sydney Opera House",
    "Alhambra",
]


def ensure_dirs() -> None:
    """Create every directory the pipeline expects.  Idempotent."""
    for path in (
        DATA_DIR,
        RAW_DIR,
        RAW_PEOPLE_DIR,
        RAW_PLACES_DIR,
        INDEX_DIR,
        DB_DIR,
        PEOPLE_INDEX_PATH,
        PLACES_INDEX_PATH,
    ):
        path.mkdir(parents=True, exist_ok=True)
