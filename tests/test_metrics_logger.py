"""
tests/test_metrics_logger.py
-----------------------------
Phase 11 — Unit Tests for MetricsLogger.

Tests:
  ✓ CSV created on first call
  ✓ Rows can be appended
  ✓ Data can be loaded back correctly

Run:
    python tests/test_metrics_logger.py
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

from logs.metrics_logger import MetricsLogger, QueryMetrics

PASS = "[PASS]"
FAIL = "[FAIL]"


def check(label: str, condition: bool, detail: str = "") -> bool:
    tag = PASS if condition else FAIL
    suffix = f" -- {detail}" if detail else ""
    print(f"    {tag} {label}{suffix}")
    return condition


def _make_metric(
    topic: str = "promotion",
    confidence: float = 0.85,
    validation_passed: bool = True,
    retries: int = 0,
    latency: float = 22.5,
    rows: int = 4,
    cache: bool = False,
    responded: bool = True,
) -> QueryMetrics:
    return QueryMetrics(
        timestamp=datetime.now(timezone.utc).isoformat(),
        question="Did PROMO_001 improve sales in South region?",
        topic=topic,
        classification_confidence=confidence,
        validation_passed=validation_passed,
        retry_count=retries,
        execution_latency_ms=latency,
        row_count=rows,
        cache_hit=cache,
        response_generated=responded,
    )


# ---------------------------------------------------------------------------
# T1 — CSV is auto-created when MetricsLogger initialises
# ---------------------------------------------------------------------------

def test_csv_created() -> bool:
    print("\n[T1] CSV Auto-Creation")
    ok = True
    with tempfile.TemporaryDirectory() as tmp:
        csv_path = Path(tmp) / "metrics" / "query_metrics.csv"
        logger = MetricsLogger(csv_path=csv_path)
        ok &= check("CSV file created", csv_path.exists())
        ok &= check("CSV file non-empty (has header)", csv_path.stat().st_size > 0)
        # Re-initialising must not crash or duplicate the header
        logger.initialize_metrics_store()
        ok &= check("Re-init is idempotent (no crash)", True)
    return ok


# ---------------------------------------------------------------------------
# T2 — Rows are correctly appended
# ---------------------------------------------------------------------------

def test_rows_appended() -> bool:
    print("\n[T2] Row Append")
    ok = True
    with tempfile.TemporaryDirectory() as tmp:
        csv_path = Path(tmp) / "query_metrics.csv"
        logger = MetricsLogger(csv_path=csv_path)

        logger.log_query_metrics(_make_metric(topic="promotion"))
        logger.log_query_metrics(_make_metric(topic="inventory", confidence=0.91))
        logger.log_query_metrics(_make_metric(topic="campaign_impact", validation_passed=False, retries=2))

        df = logger.load_metrics()
        ok &= check("3 rows loaded", len(df) == 3, f"got {len(df)}")
        ok &= check("Topic column present", "topic" in df.columns)
        ok &= check("Row 1 topic = promotion", df.iloc[0]["topic"] == "promotion")
        ok &= check("Row 2 topic = inventory", df.iloc[1]["topic"] == "inventory")
        ok &= check("Row 3 validation_passed = False", not df.iloc[2]["validation_passed"])
        ok &= check("Row 3 retry_count = 2", df.iloc[2]["retry_count"] == 2)
    return ok


# ---------------------------------------------------------------------------
# T3 — load_metrics on empty store returns empty DataFrame
# ---------------------------------------------------------------------------

def test_load_empty() -> bool:
    print("\n[T3] Empty Store Loads Gracefully")
    ok = True
    with tempfile.TemporaryDirectory() as tmp:
        csv_path = Path(tmp) / "query_metrics.csv"
        logger = MetricsLogger(csv_path=csv_path)
        df = logger.load_metrics()
        ok &= check("Returns DataFrame", hasattr(df, "columns"))
        ok &= check("Empty DataFrame (header only)", len(df) == 0)
    return ok


# ---------------------------------------------------------------------------
# T4 — Boolean columns survive CSV round-trip correctly
# ---------------------------------------------------------------------------

def test_boolean_roundtrip() -> bool:
    print("\n[T4] Boolean Column Round-Trip")
    ok = True
    with tempfile.TemporaryDirectory() as tmp:
        csv_path = Path(tmp) / "query_metrics.csv"
        logger = MetricsLogger(csv_path=csv_path)
        logger.log_query_metrics(_make_metric(validation_passed=True, cache=True, responded=True))
        logger.log_query_metrics(_make_metric(validation_passed=False, cache=False, responded=False))
        df = logger.load_metrics()
        ok &= check("Row 0 validation_passed=True", df.iloc[0]["validation_passed"] is True)
        ok &= check("Row 0 cache_hit=True", df.iloc[0]["cache_hit"] is True)
        ok &= check("Row 1 validation_passed=False", df.iloc[1]["validation_passed"] is False)
    return ok


# ---------------------------------------------------------------------------
# T5 — QueryMetrics Pydantic model validates bounds
# ---------------------------------------------------------------------------

def test_pydantic_validation() -> bool:
    print("\n[T5] QueryMetrics Pydantic Validation")
    ok = True
    try:
        _ = QueryMetrics(
            question="test",
            topic="promotion",
            classification_confidence=1.5,   # invalid: > 1.0
            validation_passed=True,
            retry_count=0,
            execution_latency_ms=10.0,
            row_count=5,
            cache_hit=False,
            response_generated=True,
        )
        ok &= check("Should have raised ValidationError", False)
    except Exception:
        ok &= check("Raised ValidationError for confidence > 1.0", True)

    try:
        m = _make_metric()
        ok &= check("Valid model constructs without error", m is not None)
        ok &= check("Timestamp auto-populated", bool(m.timestamp))
    except Exception as exc:
        ok &= check("Valid model failed", False, str(exc))

    return ok


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_tests() -> None:
    tests = [
        ("T1", test_csv_created),
        ("T2", test_rows_appended),
        ("T3", test_load_empty),
        ("T4", test_boolean_roundtrip),
        ("T5", test_pydantic_validation),
    ]
    total = len(tests)
    passed = 0
    failed = []

    print("\n" + "=" * 70)
    print("PHASE 11 -- METRICS LOGGER TEST SUITE")
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
