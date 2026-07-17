"""
rag/retriever.py
----------------
Phase 2 — RAG Retrieval Layer.

Implements retrieve_grounding(query) which:
  1. Connects to the persistent ChromaDB collection.
  2. Runs cosine-similarity search for glossary definitions.
  3. Runs cosine-similarity search for relevant few-shot SQL examples.
  4. Returns a structured dict ready for injection into an agent's context.

Usage:
    from rag.retriever import retrieve_grounding

    result = retrieve_grounding("What does effectiveness mean?")
    # result["definitions"] -> list of glossary hits
    # result["examples"]    -> list of few-shot SQL hits
"""

import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import chromadb
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------------------------
# Bootstrap: project root on sys.path
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import CHROMA_PATH, LOG_LEVEL

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
COLLECTION_NAME: str = "metric_glossary"
EMBED_MODEL: str = "all-MiniLM-L6-v2"

# Module-level singletons (lazy-initialised on first call)
_client: Optional[chromadb.PersistentClient] = None
_collection: Optional[chromadb.Collection] = None
_model: Optional[SentenceTransformer] = None


# ---------------------------------------------------------------------------
# Lazy Initialisation
# ---------------------------------------------------------------------------

def _get_client() -> chromadb.PersistentClient:  # type: ignore[return]
    """Return (or create) the singleton ChromaDB client."""
    global _client
    if _client is None:
        chroma_path = PROJECT_ROOT / CHROMA_PATH
        log.debug("Connecting to ChromaDB at: %s", chroma_path)
        _client = chromadb.PersistentClient(path=str(chroma_path))
    return _client


def _get_collection() -> chromadb.Collection:  # type: ignore[return]
    """Return (or fetch) the singleton ChromaDB collection."""
    global _collection
    if _collection is None:
        client = _get_client()
        _collection = client.get_collection(name=COLLECTION_NAME)
        log.debug("Loaded collection: %s", COLLECTION_NAME)
    return _collection


def _get_model() -> SentenceTransformer:  # type: ignore[return]
    """Return (or load) the singleton sentence-transformer model."""
    global _model
    if _model is None:
        log.debug("Loading embedding model: %s", EMBED_MODEL)
        _model = SentenceTransformer(EMBED_MODEL)
    return _model


# ---------------------------------------------------------------------------
# Core Retrieval
# ---------------------------------------------------------------------------

def _embed(query: str) -> List[float]:
    """Encode a query string into a float embedding vector."""
    return _get_model().encode([query], show_progress_bar=False)[0].tolist()


def _query_collection(
    query_embedding: List[float],
    doc_type: str,
    top_k: int,
) -> List[Dict[str, Any]]:
    """
    Run a filtered similarity search on the collection.

    Args:
        query_embedding: Pre-computed query vector.
        doc_type: ChromaDB metadata filter value ('glossary' or 'few_shot').
        top_k: Number of results to return.

    Returns:
        List of result dicts with document text, metadata, and distance score.
    """
    collection = _get_collection()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        where={"type": doc_type},
        include=["documents", "metadatas", "distances"],
    )

    hits: List[Dict[str, Any]] = []
    if not results["documents"] or not results["documents"][0]:
        return hits

    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        hits.append(
            {
                "document": doc,
                "metadata": meta,
                "similarity_score": round(1 - dist, 4),  # cosine: 1 - distance
            }
        )

    return hits


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def retrieve_grounding(
    query: str,
    top_k_definitions: int = 3,
    top_k_examples: int = 3,
) -> Dict[str, Any]:
    """
    Retrieve grounding context for a natural-language query.

    Runs two filtered similarity searches against the metric_glossary
    ChromaDB collection:
      - Glossary definitions  → semantic match on business metric terms
      - Few-shot SQL examples → semantic match on question patterns

    Args:
        query: The user's natural-language question.
        top_k_definitions: Number of glossary definitions to retrieve (default 3).
        top_k_examples: Number of few-shot examples to retrieve (default 3).

    Returns:
        {
            "query": str,
            "definitions": [
                {
                    "term": str,
                    "definition": str,
                    "formula": str,
                    "similarity_score": float
                },
                ...
            ],
            "examples": [
                {
                    "question": str,
                    "question_type": str,
                    "sql": str,
                    "explanation": str,
                    "similarity_score": float
                },
                ...
            ]
        }
    """
    log.info("Retrieving grounding for query: %r", query)

    query_vec = _embed(query)

    # ---- Glossary definitions ----------------------------------------------
    raw_defs = _query_collection(query_vec, "glossary", top_k_definitions)
    definitions: List[Dict[str, Any]] = []
    for hit in raw_defs:
        meta = hit["metadata"]
        definitions.append(
            {
                "term": meta.get("term", ""),
                "definition": meta.get("definition", ""),
                "formula": meta.get("formula", ""),
                "similarity_score": hit["similarity_score"],
            }
        )
    log.info("  Retrieved %d definition(s)", len(definitions))

    # ---- Few-shot examples --------------------------------------------------
    raw_examples = _query_collection(query_vec, "few_shot", top_k_examples)
    examples: List[Dict[str, Any]] = []
    for hit in raw_examples:
        meta = hit["metadata"]
        examples.append(
            {
                "question": meta.get("question", ""),
                "question_type": meta.get("question_type", ""),
                "sql": meta.get("sql", ""),
                "explanation": meta.get("explanation", ""),
                "similarity_score": hit["similarity_score"],
            }
        )
    log.info("  Retrieved %d example(s)", len(examples))

    return {
        "query": query,
        "definitions": definitions,
        "examples": examples,
    }


# ---------------------------------------------------------------------------
# Validation CLI
# ---------------------------------------------------------------------------

def _print_result(result: Dict[str, Any]) -> None:
    """Pretty-print a retrieval result to stdout."""
    print(f"\n{'=' * 60}")
    print(f"QUERY: {result['query']}")
    print(f"{'=' * 60}")

    print(f"\n📘 DEFINITIONS ({len(result['definitions'])} results):")
    for i, d in enumerate(result["definitions"], 1):
        print(f"  [{i}] Term       : {d['term']}")
        print(f"      Score      : {d['similarity_score']}")
        print(f"      Definition : {d['definition'][:120]}...")
        print(f"      Formula    : {d['formula']}")
        print()

    print(f"💡 FEW-SHOT EXAMPLES ({len(result['examples'])} results):")
    for i, ex in enumerate(result["examples"], 1):
        print(f"  [{i}] Type       : {ex['question_type']}")
        print(f"      Score      : {ex['similarity_score']}")
        print(f"      Question   : {ex['question']}")
        print(f"      SQL        : {ex['sql'][:120].strip()}...")
        print()


if __name__ == "__main__":
    # Validation queries
    test_queries = [
        "What does effectiveness mean?",
        "Which region performed best during promotion?",
    ]

    for q in test_queries:
        result = retrieve_grounding(q, top_k_definitions=3, top_k_examples=2)
        _print_result(result)
