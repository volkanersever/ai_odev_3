"""Wikipedia ingestion.

Pulls plain-text article bodies straight from the public MediaWiki API using
``urllib`` from the standard library.  Sticking to ``urllib`` keeps the
dependency surface small and respects the brief's preference for
language-native functionality over heavyweight wrappers.
"""

from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path

WIKI_API: str = "https://en.wikipedia.org/w/api.php"
USER_AGENT: str = "LocalWikipediaRAG/1.0 (educational; contact: student@example.com)"


@dataclass
class Article:
    """A single Wikipedia article persisted to disk."""

    title: str
    canonical_title: str
    url: str
    text: str
    entity_type: str  # "person" | "place"


def _http_get_json(url: str, params: dict[str, str], retries: int = 3) -> dict:
    """GET a JSON-returning endpoint with light retry handling."""
    qs = urllib.parse.urlencode(params)
    full_url = f"{url}?{qs}"
    request = urllib.request.Request(full_url, headers={"User-Agent": USER_AGENT})
    last_exc: Exception | None = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                payload = response.read().decode("utf-8")
                return json.loads(payload)
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as exc:
            last_exc = exc
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"Failed to fetch {full_url}: {last_exc}")


def fetch_article(title: str) -> Article | None:
    """Fetch a single Wikipedia article as plain text.

    Uses the MediaWiki ``extracts`` property with ``explaintext=1`` to get the
    article body without HTML markup.  Follows redirects automatically so
    informal titles (e.g. ``Pyramids of Giza``) resolve to canonical pages.
    """

    params = {
        "action": "query",
        "format": "json",
        "prop": "extracts|info",
        "explaintext": "1",
        "redirects": "1",
        "inprop": "url",
        "titles": title,
    }
    data = _http_get_json(WIKI_API, params)
    pages = data.get("query", {}).get("pages", {})
    if not pages:
        return None
    page = next(iter(pages.values()))
    if page.get("missing") is not None:
        return None
    extract = (page.get("extract") or "").strip()
    if not extract:
        return None
    return Article(
        title=title,
        canonical_title=page.get("title", title),
        url=page.get("fullurl", f"https://en.wikipedia.org/wiki/{urllib.parse.quote(title)}"),
        text=_clean_text(extract),
        entity_type="",  # filled in by the caller
    )


def _clean_text(text: str) -> str:
    """Light normalisation: collapse whitespace, drop "See also"/refs noise."""
    # Strip section markers that the API leaves in plain-text mode.
    text = re.sub(r"=+\s*See also\s*=+.*", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"=+\s*References\s*=+.*", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"=+\s*Notes\s*=+.*", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"=+\s*External links\s*=+.*", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"=+\s*Further reading\s*=+.*", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"=+\s*Bibliography\s*=+.*", "", text, flags=re.DOTALL | re.IGNORECASE)
    # The plain-text API still emits "==" style headings -- normalise to a
    # readable form so the LLM gets clean section breaks.
    text = re.sub(r"={2,}\s*([^=\n]+?)\s*={2,}", r"\n\n\1\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def safe_filename(title: str) -> str:
    """Produce a filesystem-safe representation of a Wikipedia title."""
    cleaned = re.sub(r"[^A-Za-z0-9]+", "_", title).strip("_")
    return cleaned or "untitled"


def save_article(article: Article, target_dir: Path) -> Path:
    """Persist an article to ``target_dir`` as ``<slug>.txt`` plus ``<slug>.json``."""
    target_dir.mkdir(parents=True, exist_ok=True)
    slug = safe_filename(article.title)
    text_path = target_dir / f"{slug}.txt"
    meta_path = target_dir / f"{slug}.json"
    text_path.write_text(article.text, encoding="utf-8")
    meta_path.write_text(
        json.dumps(
            {
                "title": article.title,
                "canonical_title": article.canonical_title,
                "url": article.url,
                "entity_type": article.entity_type,
                "char_count": len(article.text),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return text_path


def ingest_titles(titles: list[str], target_dir: Path, entity_type: str) -> list[Article]:
    """Fetch and persist every title under ``target_dir``.

    Already-downloaded articles are reused so reruns stay cheap.
    """
    target_dir.mkdir(parents=True, exist_ok=True)
    results: list[Article] = []
    for title in titles:
        slug = safe_filename(title)
        text_path = target_dir / f"{slug}.txt"
        meta_path = target_dir / f"{slug}.json"
        if text_path.exists() and meta_path.exists():
            text = text_path.read_text(encoding="utf-8")
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            results.append(
                Article(
                    title=meta.get("title", title),
                    canonical_title=meta.get("canonical_title", title),
                    url=meta.get("url", ""),
                    text=text,
                    entity_type=entity_type,
                )
            )
            print(f"  [cached] {title}")
            continue

        try:
            article = fetch_article(title)
        except RuntimeError as exc:
            print(f"  [error]  {title}: {exc}")
            continue
        if article is None:
            print(f"  [miss]   {title} (no Wikipedia page)")
            continue
        article.entity_type = entity_type
        save_article(article, target_dir)
        results.append(article)
        print(f"  [ok]     {title} -> {len(article.text):,} chars")
        time.sleep(0.5)  # be polite to Wikipedia
    return results
