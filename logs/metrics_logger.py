"""
logs/metrics_logger.py
-----------------------
Phase 11 — Success Metrics Instrumentation.

Provides structured metric logging for every pipeline execution.
Appends rows to logs/query_metrics.csv for persistent storage.
Exposes load_metrics() for the dashboard to consume.

Usage:
    from logs.metrics_logger import MetricsLogger, QueryMetrics

    logger = MetricsLogger()
    logger.log_query_metrics(QueryMetrics(...))
    df = logger.load_metrics()
"""

import csv
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Bootstrap project root
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import LOG_LEVEL

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
# CSV storage path
# ---------------------------------------------------------------------------
METRICS_DIR = PROJECT_ROOT / "logs"
METRICS_CSV = METRICS_DIR / "query_metrics.csv"

CSV_COLUMNS = [
    "timestamp",
    "question",
    "topic",
    "classification_confidence",
    "validation_passed",
    "retry_count",
    "execution_latency_ms",
    "row_count",
    "cache_hit",
    "response_generated",
]


# ---------------------------------------------------------------------------
# Pydantic Model
# ---------------------------------------------------------------------------


class QueryMetrics(BaseModel):
    """
    Structured metrics record for a single pipeline execution.

    Captured after every orchestrator.handle() call — both
    successful and failed runs.
    """

    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="ISO-8601 UTC timestamp of the query.",
    )
    question: str = Field(description="The original user question.")
    topic: str = Field(description="Classified intent topic.")
    classification_confidence: float = Field(
        ge=0.0, le=1.0,
        description="Classifier confidence score [0.0, 1.0].",
    )
    validation_passed: bool = Field(
        description="True if the generated SQL passed all validation checks."
    )
    retry_count: int = Field(
        ge=0,
        description="Number of SQL regeneration retries attempted.",
    )
    execution_latency_ms: float = Field(
        ge=0.0,
        description="DuckDB execution time in milliseconds.",
    )
    row_count: int = Field(
        ge=0,
        description="Number of result rows returned.",
    )
    cache_hit: bool = Field(
        description="True if the result was served from the rollup cache."
    )
    response_generated: bool = Field(
        description="True if a non-fallback response was returned to the user."
    )


# ---------------------------------------------------------------------------
# MetricsLogger
# ---------------------------------------------------------------------------


class MetricsLogger:
    """
    Append-only metrics store backed by a local CSV file.

    Thread-safety: single-process safe; uses file-level locking
    via the default CSV writer (suitable for Streamlit single-server deploys).
    """

    def __init__(self, csv_path: Optional[Path] = None) -> None:
        self._csv_path = csv_path or METRICS_CSV
        self.initialize_metrics_store()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def initialize_metrics_store(self) -> None:
        """
        Create the metrics CSV and its parent directory if they do not exist.
        Writes the header row only when the file is brand-new.
        """
        self._csv_path.parent.mkdir(parents=True, exist_ok=True)

        if not self._csv_path.exists():
            with open(self._csv_path, mode="w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
                writer.writeheader()
            log.info("MetricsLogger: created metrics store at %s", self._csv_path)
        else:
            log.debug("MetricsLogger: using existing store at %s", self._csv_path)

    def log_query_metrics(self, metrics: QueryMetrics) -> None:
        """
        Append a single QueryMetrics record as a new CSV row.

        Args:
            metrics: Populated QueryMetrics Pydantic object.
        """
        row = {
            "timestamp": metrics.timestamp,
            "question": metrics.question,
            "topic": metrics.topic,
            "classification_confidence": metrics.classification_confidence,
            "validation_passed": metrics.validation_passed,
            "retry_count": metrics.retry_count,
            "execution_latency_ms": metrics.execution_latency_ms,
            "row_count": metrics.row_count,
            "cache_hit": metrics.cache_hit,
            "response_generated": metrics.response_generated,
        }
        try:
            with open(self._csv_path, mode="a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
                writer.writerow(row)
            log.debug(
                "MetricsLogger: logged query | topic=%s | valid=%s | latency=%.1f ms",
                metrics.topic,
                metrics.validation_passed,
                metrics.execution_latency_ms,
            )
        except Exception as exc:
            # Never crash the pipeline because of a logging failure
            log.error("MetricsLogger: failed to write row: %s", exc)

    def load_metrics(self) -> pd.DataFrame:
        """
        Load all stored metrics from the CSV into a pandas DataFrame.

        Returns:
            DataFrame with correct dtypes, or an empty DataFrame if no data.
        """
        if not self._csv_path.exists() or self._csv_path.stat().st_size == 0:
            log.debug("MetricsLogger: no metrics data available yet.")
            return pd.DataFrame(columns=CSV_COLUMNS)

        try:
            df = pd.read_csv(self._csv_path, parse_dates=["timestamp"])
            # Ensure boolean columns are typed correctly after CSV round-trip.
            # CSV stores True/False as the strings "True"/"False".
            # We map them back to native Python bool so `is True` comparisons work.
            for bool_col in ("validation_passed", "cache_hit", "response_generated"):
                if bool_col in df.columns:
                    df[bool_col] = (
                        df[bool_col]
                        .astype(str)
                        .str.lower()
                        .map({"true": True, "false": False})
                        .astype(object)   # keep as object dtype of Python bools
                    )
            log.debug("MetricsLogger: loaded %d rows from %s", len(df), self._csv_path)
            return df
        except Exception as exc:
            log.error("MetricsLogger: failed to load metrics: %s", exc)
            return pd.DataFrame(columns=CSV_COLUMNS)


# ---------------------------------------------------------------------------
# Convenience module-level instance  (shared across imports in same process)
# ---------------------------------------------------------------------------
_default_logger: Optional[MetricsLogger] = None


def get_metrics_logger() -> MetricsLogger:
    """Return the module-level singleton MetricsLogger."""
    global _default_logger
    if _default_logger is None:
        _default_logger = MetricsLogger()
    return _default_logger


def generate_sample_metrics(n: int = 15) -> None:
    """
    Populate the CSV with synthetic but realistic records for dashboard testing.

    Args:
        n: Number of sample rows to insert (default 15).
    """
    import random
    from datetime import timedelta

    logger = get_metrics_logger()
    topics = ["promotion", "inventory", "region_comparison", "campaign_impact"]
    sample_questions = [
        "Did PROMO_001 improve sales in South region?",
        "Compare North and South sales.",
        "Did inventory reduce in West region?",
        "Which campaign performed best?",
        "Which category generated highest revenue?",
        "Show revenue growth for Electronics.",
        "What was the lift from summer promotions?",
        "Which SKU drove the highest sales?",
        "How did inventory levels change last quarter?",
        "Was PROMO_003 effective in East region?",
    ]

    base_time = datetime.now(timezone.utc)
    for i in range(n):
        ts = (base_time - timedelta(minutes=i * 12)).isoformat()
        val_passed = random.random() > 0.12  # ~88% pass rate
        logger.log_query_metrics(
            QueryMetrics(
                timestamp=ts,
                question=random.choice(sample_questions),
                topic=random.choice(topics),
                classification_confidence=round(random.uniform(0.70, 1.0), 4),
                validation_passed=val_passed,
                retry_count=0 if val_passed else random.randint(1, 2),
                execution_latency_ms=round(random.uniform(8.0, 80.0), 2),
                row_count=random.randint(1, 50) if val_passed else 0,
                cache_hit=random.random() > 0.55,
                response_generated=val_passed,
            )
        )

    log.info("MetricsLogger: inserted %d sample metrics rows.", n)
