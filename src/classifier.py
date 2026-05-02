"""Person / place / mixed query classification.

The brief allows a simple rule-based classifier.  We use three signals, each
of which is independently sufficient evidence:

1. **Entity name match.**  If the query mentions any indexed person or place
   by name (or surname), classify directly.
2. **Lexical cues.**  Words like *who*, *born*, *invented*, *singer* lean
   person; *where*, *located*, *built*, *city*, *mountain* lean place.
3. **Comparison detection.**  Queries containing *compare*, *vs*, *and* with
   two named entities of different types are classified as ``BOTH``.

When evidence is weak we fall back to ``BOTH`` and let semantic search
arbitrate -- the cost of a slightly broader retrieval is small.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from enum import Enum


class QueryType(str, Enum):
    PERSON = "person"
    PLACE = "place"
    BOTH = "both"


@dataclass
class ClassificationResult:
    query_type: QueryType
    matched_people: list[str]
    matched_places: list[str]
    reason: str


PERSON_KEYWORDS = {
    "who", "whom", "born", "biography", "biographies", "scientist", "scientists",
    "physicist", "physicists", "chemist", "chemists", "artist", "artists",
    "painter", "painters", "writer", "writers", "author", "authors",
    "inventor", "inventors", "musician", "musicians", "singer", "singers",
    "composer", "composers", "footballer", "footballers", "actor", "actors",
    "actress", "actresses", "leader", "leaders", "philosopher", "philosophers",
    "engineer", "engineers", "mathematician", "mathematicians",
    "person", "people", "she", "he", "him", "her", "his", "hers",
    "married", "husband", "wife", "child", "children", "father", "mother",
    "discovered", "invented", "wrote", "painted", "won", "awarded",
}

PLACE_KEYWORDS = {
    "where", "located", "location", "city", "country", "river", "mountain",
    "mountains", "wonder", "wonders", "monument", "monuments", "building",
    "buildings", "site", "sites", "tower", "wall", "bridge", "temple",
    "cathedral", "palace", "fortress", "ruins", "park", "canyon", "valley",
    "island", "lake", "ocean", "sea", "continent", "place", "places",
    "visit", "tourists", "tourist", "landmark", "landmarks",
    "constructed", "built", "founded", "destination",
}

COMPARE_TOKENS = {"compare", "comparison", "versus", "vs", "vs.", "between", "and"}


def _normalise(text: str) -> str:
    text = unicodedata.normalize("NFKD", text)
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _tokens(text: str) -> set[str]:
    return set(_normalise(text).split())


# Tokens that appear in multiple entity names or are otherwise too generic
# to use as a single-word match.  Two roles: (1) common-noun "surnames"
# (statue, tower, ...) and (2) category prefixes that begin many names
# (mount, lake, ...).
_AMBIGUOUS_SHORT_TOKENS = {
    # Common-noun trailing words
    "statue", "tower", "wall", "park", "city", "river", "valley", "lake",
    "bridge", "temple", "palace", "church", "complex", "falls", "wonder",
    "athens", "china",
    # Category prefixes
    "mount", "lake", "river", "saint", "sir", "dame", "lord", "lady",
}


def _name_variants(name: str) -> list[str]:
    """Return forms of ``name`` we can search for in a normalised query.

    Wikipedia titles often carry a parenthetical disambiguator
    (e.g. ``Christ the Redeemer (statue)``).  We strip that part before
    deriving variants so the disambiguator can't masquerade as a surname.
    """
    primary = re.sub(r"\([^)]*\)", " ", name)
    base = _normalise(primary)
    parts = base.split()
    out = {base}
    if len(parts) >= 2:
        # Surname-only is the most common reference form.
        last = parts[-1]
        first = parts[0]
        if last not in _AMBIGUOUS_SHORT_TOKENS and len(last) >= 4:
            out.add(last)
        if first not in _AMBIGUOUS_SHORT_TOKENS and len(first) >= 4:
            out.add(first)
    return [v for v in out if v]


def _find_matches(query_norm: str, candidates: list[str]) -> list[str]:
    tokens = set(query_norm.split())
    matched: list[str] = []
    for original in candidates:
        for variant in _name_variants(original):
            if " " in variant:
                if variant in query_norm:
                    matched.append(original)
                    break
            else:
                if variant in tokens:
                    matched.append(original)
                    break
    return matched


def classify(
    query: str,
    *,
    people: list[str],
    places: list[str],
) -> ClassificationResult:
    """Decide whether ``query`` is about people, places, or both."""
    query_norm = _normalise(query)
    tokens = set(query_norm.split())

    matched_people = _find_matches(query_norm, people)
    matched_places = _find_matches(query_norm, places)

    is_comparison = any(t in tokens for t in COMPARE_TOKENS) or " vs " in f" {query_norm} "

    # 1. Strong signal: entities of both kinds named explicitly.
    if matched_people and matched_places:
        return ClassificationResult(
            query_type=QueryType.BOTH,
            matched_people=matched_people,
            matched_places=matched_places,
            reason="Entities of both types named in query",
        )

    # 2. Comparison phrasing with multiple matches of one kind -- still that kind.
    if matched_people and not matched_places:
        return ClassificationResult(
            query_type=QueryType.PERSON,
            matched_people=matched_people,
            matched_places=[],
            reason="Person name(s) detected: " + ", ".join(matched_people),
        )
    if matched_places and not matched_people:
        return ClassificationResult(
            query_type=QueryType.PLACE,
            matched_people=[],
            matched_places=matched_places,
            reason="Place name(s) detected: " + ", ".join(matched_places),
        )

    # 3. No entity match -- lean on lexical cues.
    person_score = len(tokens & PERSON_KEYWORDS)
    place_score = len(tokens & PLACE_KEYWORDS)
    if person_score > place_score and person_score > 0:
        return ClassificationResult(
            query_type=QueryType.PERSON,
            matched_people=[],
            matched_places=[],
            reason=f"Person-leaning keywords (score {person_score} vs {place_score})",
        )
    if place_score > person_score and place_score > 0:
        return ClassificationResult(
            query_type=QueryType.PLACE,
            matched_people=[],
            matched_places=[],
            reason=f"Place-leaning keywords (score {place_score} vs {person_score})",
        )

    # 4. Last resort: search both stores.  Comparison queries also land here
    # when neither side is named explicitly.
    return ClassificationResult(
        query_type=QueryType.BOTH,
        matched_people=[],
        matched_places=[],
        reason=(
            "Comparison phrasing without specific entities"
            if is_comparison
            else "Insufficient signal -- searching both stores"
        ),
    )
