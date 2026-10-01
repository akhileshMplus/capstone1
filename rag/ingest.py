"""
RAG Ingest — builds and persists the BM25 + ChromaDB retrieval indexes.

Run once (or re-run to rebuild):
    python -m rag.ingest
"""
from __future__ import annotations

import json
import os
import pickle
import re
from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions
from rank_bm25 import BM25Okapi

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
INDEX_DIR = ROOT / "index"
INDEX_DIR.mkdir(exist_ok=True)

BM25_PATH = INDEX_DIR / "bm25.pkl"
CHROMA_DIR = INDEX_DIR / "chroma"


# ── Helpers ───────────────────────────────────────────────────────────────────
def tokenise(text: str) -> list[str]:
    """Whitespace + punctuation tokeniser for BM25."""
    return re.findall(r"\w+", text.lower())


def incident_text(inc: dict) -> str:
    """Canonical embedding text: error + root_cause + tags."""
    return f"{inc['error']}. {inc['root_cause']}. Tags: {', '.join(inc['tags'])}"


# ── Build indexes ─────────────────────────────────────────────────────────────
def build_indexes(incidents: list[dict]) -> tuple[BM25Okapi, chromadb.Collection]:
    """Build and persist BM25 + ChromaDB indexes from incident list."""

    # --- BM25 ---
    bm25_corpus = [
        tokenise(
            f"{inc['error']} {inc['root_cause']} {inc['pipeline']} {' '.join(inc['tags'])}"
        )
        for inc in incidents
    ]
    bm25 = BM25Okapi(bm25_corpus)
    with open(BM25_PATH, "wb") as f:
        pickle.dump({"bm25": bm25, "incident_ids": [i["id"] for i in incidents]}, f)
    print(f"✓ BM25 index built — {len(bm25_corpus)} documents → {BM25_PATH}")

    # --- ChromaDB (persistent) ---
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    ef = embedding_functions.DefaultEmbeddingFunction()

    # Drop and recreate so re-runs are idempotent
    try:
        client.delete_collection("incidents")
    except Exception:
        pass
    collection = client.get_or_create_collection("incidents", embedding_function=ef)

    docs = [incident_text(i) for i in incidents]
    metadatas = [
        {
            "id": i["id"],
            "pipeline": i["pipeline"],
            "severity": i["severity"],
            "resolution": i["resolution"],
            "fix_sql": i["fix_sql"],
            "resolved_mins": i["resolved_mins"],
            "tags": ",".join(i["tags"]),
        }
        for i in incidents
    ]
    ids = [i["id"] for i in incidents]

    collection.add(documents=docs, metadatas=metadatas, ids=ids)
    print(f"✓ Vector store built — {collection.count()} documents → {CHROMA_DIR}")

    return bm25, collection


def main() -> None:
    incidents = json.loads((DATA_DIR / "incidents.json").read_text())
    build_indexes(incidents)
    print("✓ Ingest complete")


if __name__ == "__main__":
    main()
