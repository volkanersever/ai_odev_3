"""Pull every required Wikipedia page and persist it to ``data/raw/``.

Usage:
    python -m scripts.ingest               # ingest everything
    python -m scripts.ingest --refresh     # force re-download
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

# Make repo root importable when running ``python scripts/ingest.py``.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import (  # noqa: E402
    PEOPLE,
    PLACES,
    RAW_PEOPLE_DIR,
    RAW_PLACES_DIR,
    ensure_dirs,
)
from src.ingest import ingest_titles  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Ingest Wikipedia articles")
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Delete existing raw articles before fetching",
    )
    args = parser.parse_args()

    ensure_dirs()

    if args.refresh:
        for d in (RAW_PEOPLE_DIR, RAW_PLACES_DIR):
            if d.exists():
                shutil.rmtree(d)
                d.mkdir(parents=True, exist_ok=True)
        print("Cleared existing raw data\n")

    print(f"Ingesting {len(PEOPLE)} people...")
    people = ingest_titles(PEOPLE, RAW_PEOPLE_DIR, "person")
    print(f"\nIngesting {len(PLACES)} places...")
    places = ingest_titles(PLACES, RAW_PLACES_DIR, "place")

    print(
        f"\nDone. {len(people)}/{len(PEOPLE)} people and "
        f"{len(places)}/{len(PLACES)} places stored under data/raw/."
    )
    return 0 if people and places else 1


if __name__ == "__main__":
    raise SystemExit(main())
