"""
Hybrid retriever — BM25 + vector search merged with Reciprocal Rank Fusion.

RRF score: 1/(k + rank_bm25) + 1/(k + rank_vector),  k = 60
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Optional

import chromadb
from chromadb.utils import embedding_functions

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
INDEX_DIR = ROOT / "index"
BM25_PATH = INDEX_DIR / "bm25.pkl"
CHROMA_DIR = INDEX_DIR / "chroma"

# Minimum RRF score for a result to be considered a strong match.
# Below this threshold the diagnosis layer should treat it as "no strong precedent."
MIN_RELEVANCE = 0.3


# ── Lazy singletons ───────────────────────────────────────────────────────────
_bm25_data: Optional[dict] = None
_collection: Optional[chromadb.Collection] = None
_incidents_by_id: Optional[dict] = None


def _load_incidents() -> dict:
    global _incidents_by_id
    if _incidents_by_id is None:
        incidents = json.loads((DATA_DIR / "incidents.json").read_text())
        _incidents_by_id = {i["id"]: i for i in incidents}
    return _incidents_by_id


def _load_bm25() -> dict:
    global _bm25_data
    if _bm25_data is None:
        if not BM25_PATH.exists():
            raise FileNotFoundError(
                f"BM25 index not found at {BM25_PATH}. Run: python -m rag.ingest"
            )
        with open(BM25_PATH, "rb") as f:
            _bm25_data = pickle.load(f)
    return _bm25_data


def _load_collection() -> chromadb.Collection:
    global _collection
    if _collection is None:
        if not CHROMA_DIR.exists():
            raise FileNotFoundError(
                f"ChromaDB not found at {CHROMA_DIR}. Run: python -m rag.ingest"
            )
        client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        ef = embedding_functions.DefaultEmbeddingFunction()
        _collection = client.get_collection("incidents", embedding_function=ef)
    return _collection


# ── Core retrieval ────────────────────────────────────────────────────────────
def hybrid_retrieve(query: str, top_k: int = 3, k: int = 60) -> list[dict]:
    """
    Merge BM25 and vector rankings using Reciprocal Rank Fusion.

    Returns a list of dicts:
        {
            "incident": <full incident dict>,
            "rrf_score": float,          # higher is more relevant
            "bm25_rank": int | None,
            "vector_rank": int | None,
        }

    Results with rrf_score < MIN_RELEVANCE are still returned (so callers can
    see the score), but the caller should treat them as "no strong precedent."
    """
    import re

    def tokenise(text: str) -> list[str]:
        return re.findall(r"\w+", text.lower())

    bm25_data = _load_bm25()
    collection = _load_collection()
    incidents_by_id = _load_incidents()

    bm25 = bm25_data["bm25"]
    incident_ids = bm25_data["incident_ids"]
    n_total = len(incident_ids)

    # ── BM25 ranking ─────────────────────────────────────────────────────────
    bm25_scores = bm25.get_scores(tokenise(query))
    # rank 0 = highest score
    bm25_ranked = sorted(
        enumerate(bm25_scores), key=lambda x: x[1], reverse=True
    )
    bm25_rank_map: dict[str, int] = {
        incident_ids[idx]: rank for rank, (idx, _) in enumerate(bm25_ranked)
    }

    # ── Vector ranking ────────────────────────────────────────────────────────
    vec_results = collection.query(
        query_texts=[query],
        n_results=n_total,
        include=["distances"],
    )
    vec_ids_ordered: list[str] = vec_results["ids"][0]  # closest first
    vec_rank_map: dict[str, int] = {
        inc_id: rank for rank, inc_id in enumerate(vec_ids_ordered)
    }

    # ── Reciprocal Rank Fusion ────────────────────────────────────────────────
    all_ids = set(bm25_rank_map) | set(vec_rank_map)
    rrf_scores: dict[str, float] = {}
    for inc_id in all_ids:
        bm25_r = bm25_rank_map.get(inc_id, n_total)   # unseen → worst rank
        vec_r = vec_rank_map.get(inc_id, n_total)
        rrf_scores[inc_id] = 1 / (k + bm25_r) + 1 / (k + vec_r)

    # Sort descending, take top_k
    top_ids = sorted(rrf_scores, key=lambda x: rrf_scores[x], reverse=True)[:top_k]

    results = []
    for inc_id in top_ids:
        results.append(
            {
                "incident": incidents_by_id[inc_id],
                "rrf_score": round(rrf_scores[inc_id], 6),
                "bm25_rank": bm25_rank_map.get(inc_id),
                "vector_rank": vec_rank_map.get(inc_id),
            }
        )

    return results


def has_strong_precedent(results: list[dict]) -> bool:
    """True if at least one result exceeds the minimum relevance threshold."""
    return any(r["rrf_score"] >= MIN_RELEVANCE for r in results)
