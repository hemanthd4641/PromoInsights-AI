"""
rag/build_index.py
------------------
Phase 2 — ChromaDB Index Builder.

Responsibilities:
  1. Load glossary.json and few_shot_bank.json from rag/.
  2. Generate sentence embeddings via sentence-transformers (all-MiniLM-L6-v2).
  3. Build (or safely rebuild) the ChromaDB 'metric_glossary' collection.
  4. Store all documents with structured metadata.
  5. Log indexed document counts and print a success summary.

Run:
    python rag/build_index.py
"""

import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List

import chromadb
from chromadb.config import Settings
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
RAG_DIR: Path = PROJECT_ROOT / "rag"
GLOSSARY_PATH: Path = RAG_DIR / "glossary.json"
FEW_SHOT_PATH: Path = RAG_DIR / "few_shot_bank.json"


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_glossary() -> List[Dict[str, Any]]:
    """Load and validate glossary entries from glossary.json."""
    log.info("Loading glossary from: %s", GLOSSARY_PATH)
    with open(GLOSSARY_PATH, encoding="utf-8") as f:
        data: List[Dict[str, Any]] = json.load(f)
    log.info("  Loaded %d glossary entries", len(data))
    return data


def load_few_shots() -> List[Dict[str, Any]]:
    """Load and validate few-shot examples from few_shot_bank.json."""
    log.info("Loading few-shot bank from: %s", FEW_SHOT_PATH)
    with open(FEW_SHOT_PATH, encoding="utf-8") as f:
        data: List[Dict[str, Any]] = json.load(f)
    log.info("  Loaded %d few-shot examples", len(data))
    return data


# ---------------------------------------------------------------------------
# Document Builders
# ---------------------------------------------------------------------------

def build_glossary_documents(
    entries: List[Dict[str, Any]],
) -> tuple[List[str], List[str], List[Dict[str, str]]]:
    """
    Convert glossary entries into (ids, texts, metadatas) tuples
    ready for ChromaDB ingestion.
    """
    ids: List[str] = []
    texts: List[str] = []
    metadatas: List[Dict[str, str]] = []

    for entry in entries:
        term = entry["term"]
        definition = entry["definition"]
        formula = entry.get("formula", "")

        ids.append(f"glossary_{term}")
        # Combine term + definition + formula for richer embedding signal
        texts.append(
            f"Term: {term}\nDefinition: {definition}\nFormula: {formula}"
        )
        metadatas.append(
            {
                "type": "glossary",
                "term": term,
                "formula": formula,
                "definition": definition,
            }
        )

    return ids, texts, metadatas


def build_few_shot_documents(
    examples: List[Dict[str, Any]],
) -> tuple[List[str], List[str], List[Dict[str, str]]]:
    """
    Convert few-shot examples into (ids, texts, metadatas) tuples
    ready for ChromaDB ingestion.
    """
    ids: List[str] = []
    texts: List[str] = []
    metadatas: List[Dict[str, str]] = []

    for ex in examples:
        ex_id = ex["id"]
        question = ex["question"]
        sql = ex["sql"]
        explanation = ex.get("explanation", "")
        q_type = ex.get("type", "")

        ids.append(f"fewshot_{ex_id}")
        # Embed question + explanation for semantic matching
        texts.append(
            f"Question: {question}\nType: {q_type}\nExplanation: {explanation}"
        )
        metadatas.append(
            {
                "type": "few_shot",
                "question_type": q_type,
                "question": question,
                "sql": sql,
                "explanation": explanation,
            }
        )

    return ids, texts, metadatas


# ---------------------------------------------------------------------------
# ChromaDB Builder
# ---------------------------------------------------------------------------

def build_chroma_index(
    glossary_entries: List[Dict[str, Any]],
    few_shot_examples: List[Dict[str, Any]],
) -> None:
    """
    Build (or safely rebuild) the ChromaDB metric_glossary collection.
    Deletes and recreates the collection if it already exists to avoid duplicates.
    """
    chroma_path = PROJECT_ROOT / CHROMA_PATH
    chroma_path.mkdir(parents=True, exist_ok=True)
    log.info("ChromaDB path: %s", chroma_path)

    # Connect to persistent ChromaDB
    client = chromadb.PersistentClient(path=str(chroma_path))

    # Safely rebuild — delete existing collection to avoid duplicates
    existing = [c.name for c in client.list_collections()]
    if COLLECTION_NAME in existing:
        log.info("  Deleting existing collection: %s", COLLECTION_NAME)
        client.delete_collection(COLLECTION_NAME)

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )
    log.info("  Created collection: %s", COLLECTION_NAME)

    # Load embedding model
    log.info("Loading embedding model: %s", EMBED_MODEL)
    model = SentenceTransformer(EMBED_MODEL)

    # ---- Glossary documents ------------------------------------------------
    g_ids, g_texts, g_metas = build_glossary_documents(glossary_entries)
    log.info("Generating glossary embeddings (%d docs)...", len(g_texts))
    g_embeddings = model.encode(g_texts, show_progress_bar=False).tolist()

    collection.add(
        ids=g_ids,
        documents=g_texts,
        embeddings=g_embeddings,
        metadatas=g_metas,
    )
    log.info("  ✓ Indexed %d glossary documents", len(g_ids))

    # ---- Few-shot documents ------------------------------------------------
    fs_ids, fs_texts, fs_metas = build_few_shot_documents(few_shot_examples)
    log.info("Generating few-shot embeddings (%d docs)...", len(fs_texts))
    fs_embeddings = model.encode(fs_texts, show_progress_bar=False).tolist()

    collection.add(
        ids=fs_ids,
        documents=fs_texts,
        embeddings=fs_embeddings,
        metadatas=fs_metas,
    )
    log.info("  ✓ Indexed %d few-shot documents", len(fs_ids))

    total = len(g_ids) + len(fs_ids)
    log.info("=" * 60)
    log.info("ChromaDB index built successfully.")
    log.info("  Collection  : %s", COLLECTION_NAME)
    log.info("  Total docs  : %d (%d glossary + %d few-shot)", total, len(g_ids), len(fs_ids))
    log.info("=" * 60)


# ---------------------------------------------------------------------------
# Main Entry Point
# ---------------------------------------------------------------------------

def main() -> None:
    log.info("=" * 60)
    log.info("Phase 2 — Building ChromaDB Vector Index")
    log.info("=" * 60)

    try:
        glossary = load_glossary()
        few_shots = load_few_shots()
        build_chroma_index(glossary, few_shots)
        log.info("Phase 2 index build COMPLETE.")
    except Exception as exc:
        log.exception("Index build failed: %s", exc)
        raise


if __name__ == "__main__":
    main()
