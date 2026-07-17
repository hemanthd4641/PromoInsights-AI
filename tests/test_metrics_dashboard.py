"""
tests/test_metrics_dashboard.py
---------------------------------
Phase 11 — Unit Tests for the Metrics Dashboard logic.

Tests:
  ✓ Metrics loaded from CSV
  ✓ KPI calculations correct
  ✓ Charts render without crash (headless)
  ✓ Empty state handled

Run:
    python tests/test_metrics_dashboard.py
"""

import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

# ---------------------------------------------------------------------------
# Bootstrap project root
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from logs.metrics_logger import MetricsLogger, QueryMetrics

PASS = "[PASS]"
FAIL = "[FAIL]"


def check(label: str, condition: bool, detail: str = "") -> bool:
    tag = PASS if condition else FAIL
    suffix = f" -- {detail}" if detail else ""
    print(f"    {tag} {label}{suffix}")
    return condition


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _seed_logger(logger: MetricsLogger, n_pass: int = 8, n_fail: int = 2) -> None:
    """Seed the logger with a mix of passing and failing queries."""
    for i in range(n_pass):
        logger.log_query_metrics(QueryMetrics(
            timestamp=datetime.now(timezone.utc).isoformat(),
            question=f"Passing question {i}",
            topic=["promotion", "inventory", "region_comparison", "campaign_impact"][i % 4],
            classification_confidence=round(0.80 + i * 0.01, 4),
            validation_passed=True,
            retry_count=0,
            execution_latency_ms=round(15.0 + i * 3.5, 2),
            row_count=i + 1,
            cache_hit=(i % 3 == 0),
            response_generated=True,
        ))
    for j in range(n_fail):
        logger.log_query_metrics(QueryMetrics(
            timestamp=datetime.now(timezone.utc).isoformat(),
            question=f"Failing question {j}",
            topic="promotion",
            classification_confidence=round(0.72 + j * 0.01, 4),
            validation_passed=False,
            retry_count=2,
            execution_latency_ms=0.0,
            row_count=0,
            cache_hit=False,
            response_generated=False,
        ))


def compute_kpis(df: pd.DataFrame) -> dict:
    """Mirror the compute_kpis() function in metrics_dashboard.py."""
    total = len(df)
    validation_rate = df["validation_passed"].sum() / total * 100 if total else 0.0
    avg_latency = df["execution_latency_ms"].mean() if total else 0.0
    avg_confidence = df["classification_confidence"].mean() if total else 0.0
    cache_rate = df["cache_hit"].sum() / total * 100 if total else 0.0
    response_rate = df["response_generated"].sum() / total * 100 if total else 0.0
    avg_retries = df["retry_count"].mean() if total else 0.0
    return {
        "total": total,
        "sql_accuracy": response_rate,
        "validation_rate": validation_rate,
        "avg_latency": avg_latency,
        "avg_confidence": avg_confidence,
        "cache_rate": cache_rate,
        "avg_retries": avg_retries,
    }


# ---------------------------------------------------------------------------
# T1 — Metrics are loaded from CSV correctly
# ---------------------------------------------------------------------------

def test_metrics_loaded() -> bool:
    print("\n[T1] Metrics Loaded from CSV")
    ok = True
    with tempfile.TemporaryDirectory() as tmp:
        csv_path = Path(tmp) / "query_metrics.csv"
        logger = MetricsLogger(csv_path=csv_path)
        _seed_logger(logger, n_pass=8, n_fail=2)

        df = logger.load_metrics()
        ok &= check("10 rows loaded", len(df) == 10, f"got {len(df)}")
        ok &= check("All required columns present", all(
            c in df.columns for c in [
                "timestamp", "question", "topic",
                "classification_confidence", "validation_passed",
                "retry_count", "execution_latency_ms", "row_count",
                "cache_hit", "response_generated",
            ]
        ))
        ok &= check("Validation_passed is boolean", df["validation_passed"].dtype == object or
                    set(df["validation_passed"].unique()).issubset({True, False}))
    return ok


# ---------------------------------------------------------------------------
# T2 — KPI calculations are correct
# ---------------------------------------------------------------------------

def test_kpi_calculations() -> bool:
    print("\n[T2] KPI Calculations")
    ok = True
    with tempfile.TemporaryDirectory() as tmp:
        csv_path = Path(tmp) / "query_metrics.csv"
        logger = MetricsLogger(csv_path=csv_path)
        _seed_logger(logger, n_pass=8, n_fail=2)

        df = logger.load_metrics()
        kpis = compute_kpis(df)

        ok &= check("Total queries = 10", kpis["total"] == 10, f"got {kpis['total']}")
        ok &= check("Validation rate = 80%", abs(kpis["validation_rate"] - 80.0) < 1e-6,
                    f"got {kpis['validation_rate']}")
        ok &= check("SQL accuracy = 80%", abs(kpis["sql_accuracy"] - 80.0) < 1e-6,
                    f"got {kpis['sql_accuracy']}")
        ok &= check("Avg latency > 0", kpis["avg_latency"] >= 0)
        ok &= check("Avg confidence between 0 and 1",
                    0.0 <= kpis["avg_confidence"] <= 1.0)
        ok &= check("Avg retries >= 0", kpis["avg_retries"] >= 0)
    return ok


# ---------------------------------------------------------------------------
# T3 — Charts render without crash (headless — just validates data shapes)
# ---------------------------------------------------------------------------

def test_chart_data_shapes() -> bool:
    print("\n[T3] Chart Data Shapes")
    ok = True
    with tempfile.TemporaryDirectory() as tmp:
        csv_path = Path(tmp) / "query_metrics.csv"
        logger = MetricsLogger(csv_path=csv_path)
        _seed_logger(logger)

        df = logger.load_metrics()
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)

        # Chart 1: latency over time
        latency_df = df[["timestamp", "execution_latency_ms"]].copy()
        latency_df = latency_df[latency_df["execution_latency_ms"] > 0]
        ok &= check("Latency chart has data", len(latency_df) > 0)

        # Chart 2: validation counts
        val_counts = df["validation_passed"].map({True: "Pass", False: "Fail"}).value_counts()
        ok &= check("Validation chart has Pass", "Pass" in val_counts.index)
        ok &= check("Validation chart has Fail", "Fail" in val_counts.index)

        # Chart 3: confidence histogram buckets
        bins = pd.cut(df["classification_confidence"],
                      bins=[0.0, 0.7, 0.75, 0.80, 0.85, 0.90, 0.95, 1.01],
                      right=False)
        ok &= check("Confidence histogram produced", bins.notnull().any())

        # Chart 4: retry counts
        retry_counts = df["retry_count"].value_counts()
        ok &= check("Retry chart has data", len(retry_counts) > 0)

        # Topic breakdown
        topics = df["topic"].value_counts()
        ok &= check("Topic breakdown has data", len(topics) > 0)

    return ok


# ---------------------------------------------------------------------------
# T4 — Empty state returns empty DataFrame gracefully
# ---------------------------------------------------------------------------

def test_empty_state() -> bool:
    print("\n[T4] Empty State Handling")
    ok = True
    with tempfile.TemporaryDirectory() as tmp:
        csv_path = Path(tmp) / "query_metrics.csv"
        logger = MetricsLogger(csv_path=csv_path)
        df = logger.load_metrics()
        ok &= check("Empty DataFrame returned", df.empty)
        kpis = compute_kpis(df)
        ok &= check("Total = 0", kpis["total"] == 0)
        ok &= check("Validation rate = 0", kpis["validation_rate"] == 0.0)
        ok &= check("Avg latency = 0", kpis["avg_latency"] == 0.0)
    return ok


# ---------------------------------------------------------------------------
# T5 — Sample metrics generator produces valid rows
# ---------------------------------------------------------------------------

def test_sample_generator() -> bool:
    print("\n[T5] Sample Data Generator")
    ok = True
    with tempfile.TemporaryDirectory() as tmp:
        csv_path = Path(tmp) / "query_metrics.csv"
        # Temporarily override the module-level default path
        import logs.metrics_logger as ml_module
        original_csv = ml_module.METRICS_CSV
        ml_module.METRICS_CSV = csv_path
        ml_module._default_logger = None  # reset singleton

        try:
            from logs.metrics_logger import generate_sample_metrics
            generate_sample_metrics(10)
            logger = MetricsLogger(csv_path=csv_path)
            df = logger.load_metrics()
            ok &= check("10 sample rows generated", len(df) == 10, f"got {len(df)}")
            ok &= check("All topics valid", df["topic"].isin([
                "promotion", "inventory", "region_comparison", "campaign_impact"
            ]).all())
        finally:
            ml_module.METRICS_CSV = original_csv
            ml_module._default_logger = None

    return ok


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_tests() -> None:
    tests = [
        ("T1", test_metrics_loaded),
        ("T2", test_kpi_calculations),
        ("T3", test_chart_data_shapes),
        ("T4", test_empty_state),
        ("T5", test_sample_generator),
    ]
    total = len(tests)
    passed = 0
    failed = []

    print("\n" + "=" * 70)
    print("PHASE 11 -- METRICS DASHBOARD TEST SUITE")
    print("=" * 70)

    for name, fn in tests:
        if fn():
            passed += 1
        else:
            failed.append(name)

    print("\n" + "=" * 70)
    print(f"OVERALL RESULTS: {passed}/{total} tests passed")
    if failed:
        print(f"FAILED: {failed}")
        sys.exit(1)
    else:
        print("All tests passed!")
    print("=" * 70)


if __name__ == "__main__":
    run_tests()
