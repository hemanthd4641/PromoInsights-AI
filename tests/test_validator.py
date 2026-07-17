"""
tests/test_validator.py
------------------------
Phase 6 -- Unit Tests for the SQL Validation Agent.

Covers:
  Positive tests:
    T1 -- Valid SELECT with LIMIT
    T2 -- Valid aggregation query

  Negative tests:
    T3 -- Broken SQL syntax (typo in SELECT)
    T4 -- Missing column reference
    T5 -- Missing table reference
    T6 -- Row count below ROW_COUNT_MIN (empty result)
    T7 -- Row count above ROW_COUNT_MAX (unbounded query)

  Retry logic tests:
    T8 -- Retry count decrements correctly across failures
    T9 -- Exhausted retries produce should_regenerate=False

Run:
    python tests/test_validator.py
"""

import json
import logging
import sys
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Bootstrap project root
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from agents.validator import (
    RegenerationSignal,
    SQLValidator,
    ValidationResult,
)
from config import MAX_RETRIES, ROW_COUNT_MAX, ROW_COUNT_MIN

# ---------------------------------------------------------------------------
# Logging (WARNING level to keep test output clean)
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

PASS = "[PASS]"
FAIL = "[FAIL]"


# ---------------------------------------------------------------------------
# Assertion helpers
# ---------------------------------------------------------------------------

def check(label: str, condition: bool, detail: str = "") -> bool:
    tag = PASS if condition else FAIL
    suffix = f" -- {detail}" if detail else ""
    print(f"    {tag} {label}{suffix}")
    return condition


# ---------------------------------------------------------------------------
# Test Cases
# ---------------------------------------------------------------------------

def run_tests() -> None:
    validator = SQLValidator()
    total = 0
    passed = 0
    failed_ids = []

    print("\n" + "=" * 70)
    print("PHASE 6 -- SQL VALIDATOR TEST SUITE")
    print(f"  ROW_COUNT_MIN={ROW_COUNT_MIN} | ROW_COUNT_MAX={ROW_COUNT_MAX} | MAX_RETRIES={MAX_RETRIES}")
    print("=" * 70)

    # ======================================================================
    # POSITIVE TESTS
    # ======================================================================
    print("\n--- POSITIVE TESTS ---")

    # T1 -- Valid SELECT with LIMIT
    total += 1
    print(f"\n[T1] Valid SELECT LIMIT 10")
    sql = "SELECT * FROM vw_weekly_sales LIMIT 10"
    ok = True
    try:
        result: ValidationResult = validator.validate(sql, retries_used=0)
        signal: RegenerationSignal = validator.create_regeneration_signal(result)

        print(f"  Result : {result.model_dump_json()}")
        print(f"  Signal : {signal.model_dump_json()}")

        ok &= check("is_valid=True",             result.is_valid)
        ok &= check("retry_required=False",      not result.retry_required)
        ok &= check("failure_reason is None",    result.failure_reason is None)
        ok &= check("row_count in [1, 10]",      result.row_count is not None and 1 <= result.row_count <= 10,
                    str(result.row_count))
        ok &= check("should_regenerate=False",   not signal.should_regenerate)
        ok &= check("isinstance ValidationResult", isinstance(result, ValidationResult))
        ok &= check("isinstance RegenerationSignal", isinstance(signal, RegenerationSignal))
        ok &= check("Valid JSON (result)",       bool(json.loads(result.model_dump_json())))
        ok &= check("Valid JSON (signal)",       bool(json.loads(signal.model_dump_json())))
    except Exception as exc:
        ok = False
        print(f"    {FAIL} Unexpected exception: {exc}")

    if ok:
        passed += 1
        print("  --> PASSED")
    else:
        failed_ids.append("T1")
        print("  --> FAILED")

    # T2 -- Valid aggregation
    total += 1
    print(f"\n[T2] Valid aggregation (GROUP BY region)")
    sql = "SELECT region, SUM(revenue) AS total_revenue FROM vw_weekly_sales GROUP BY region"
    ok = True
    try:
        result = validator.validate(sql, retries_used=0)
        signal = validator.create_regeneration_signal(result)

        print(f"  Result : {result.model_dump_json()}")
        print(f"  Signal : {signal.model_dump_json()}")

        ok &= check("is_valid=True",           result.is_valid)
        ok &= check("retry_required=False",    not result.retry_required)
        ok &= check("row_count >= 1",          result.row_count is not None and result.row_count >= 1,
                    str(result.row_count))
        ok &= check("should_regenerate=False", not signal.should_regenerate)
    except Exception as exc:
        ok = False
        print(f"    {FAIL} Unexpected exception: {exc}")

    if ok:
        passed += 1
        print("  --> PASSED")
    else:
        failed_ids.append("T2")
        print("  --> FAILED")

    # ======================================================================
    # NEGATIVE TESTS
    # ======================================================================
    print("\n--- NEGATIVE TESTS ---")

    # T3 -- Broken syntax
    total += 1
    print(f"\n[T3] Broken SQL syntax (SELEC typo)")
    sql = "SELEC * FROM vw_weekly_sales"
    ok = True
    try:
        result = validator.validate(sql, retries_used=0)
        signal = validator.create_regeneration_signal(result)

        print(f"  Result : {result.model_dump_json()}")
        print(f"  Signal : {signal.model_dump_json()}")

        ok &= check("is_valid=False",               not result.is_valid)
        ok &= check("validation_stage=syntax",      result.validation_stage == "syntax",
                    result.validation_stage)
        ok &= check("failure_reason set",           bool(result.failure_reason))
        ok &= check("retry_required=True",          result.retry_required)
        ok &= check("should_regenerate=True",       signal.should_regenerate)
        ok &= check("retries_remaining=MAX_RETRIES",
                    signal.retries_remaining == MAX_RETRIES,
                    str(signal.retries_remaining))
    except Exception as exc:
        ok = False
        print(f"    {FAIL} Unexpected exception: {exc}")

    if ok:
        passed += 1
        print("  --> PASSED")
    else:
        failed_ids.append("T3")
        print("  --> FAILED")

    # T4 -- Missing column
    total += 1
    print(f"\n[T4] Missing column (revenuex)")
    sql = "SELECT revenuex FROM vw_weekly_sales"
    ok = True
    try:
        result = validator.validate(sql, retries_used=0)
        signal = validator.create_regeneration_signal(result)

        print(f"  Result : {result.model_dump_json()}")
        print(f"  Signal : {signal.model_dump_json()}")

        ok &= check("is_valid=False",           not result.is_valid)
        ok &= check("failure_reason set",       bool(result.failure_reason))
        ok &= check("retry_required=True",      result.retry_required)
        ok &= check("should_regenerate=True",   signal.should_regenerate)
    except Exception as exc:
        ok = False
        print(f"    {FAIL} Unexpected exception: {exc}")

    if ok:
        passed += 1
        print("  --> PASSED")
    else:
        failed_ids.append("T4")
        print("  --> FAILED")

    # T5 -- Missing table
    total += 1
    print(f"\n[T5] Missing table (unknown_table)")
    sql = "SELECT * FROM unknown_table"
    ok = True
    try:
        result = validator.validate(sql, retries_used=0)
        signal = validator.create_regeneration_signal(result)

        print(f"  Result : {result.model_dump_json()}")
        print(f"  Signal : {signal.model_dump_json()}")

        ok &= check("is_valid=False",           not result.is_valid)
        ok &= check("failure_reason set",       bool(result.failure_reason))
        ok &= check("retry_required=True",      result.retry_required)
        ok &= check("should_regenerate=True",   signal.should_regenerate)
    except Exception as exc:
        ok = False
        print(f"    {FAIL} Unexpected exception: {exc}")

    if ok:
        passed += 1
        print("  --> PASSED")
    else:
        failed_ids.append("T5")
        print("  --> FAILED")

    # T6 -- Row count below ROW_COUNT_MIN
    total += 1
    print(f"\n[T6] Row count below ROW_COUNT_MIN ({ROW_COUNT_MIN}) -- using WHERE 1=0")
    sql = "SELECT * FROM vw_weekly_sales WHERE 1=0"
    ok = True
    try:
        result = validator.validate(sql, retries_used=0)
        signal = validator.create_regeneration_signal(result)

        print(f"  Result : {result.model_dump_json()}")
        print(f"  Signal : {signal.model_dump_json()}")

        ok &= check("is_valid=False",               not result.is_valid)
        ok &= check("validation_stage=row_count",   result.validation_stage == "row_count",
                    result.validation_stage)
        ok &= check("row_count=0",                  result.row_count == 0,
                    str(result.row_count))
        ok &= check("retry_required=True",          result.retry_required)
        ok &= check("should_regenerate=True",       signal.should_regenerate)
    except Exception as exc:
        ok = False
        print(f"    {FAIL} Unexpected exception: {exc}")

    if ok:
        passed += 1
        print("  --> PASSED")
    else:
        failed_ids.append("T6")
        print("  --> FAILED")

    # T7 -- Row count above ROW_COUNT_MAX
    total += 1
    print(f"\n[T7] Row count above ROW_COUNT_MAX ({ROW_COUNT_MAX}) -- unbounded query")
    # Generate a cross-join to guarantee > ROW_COUNT_MAX rows
    sql = (
        "SELECT a.sale_id, b.inventory_id "
        "FROM vw_weekly_sales a, vw_weekly_inventory b"
    )
    ok = True
    try:
        result = validator.validate(sql, retries_used=0)
        signal = validator.create_regeneration_signal(result)

        print(f"  Result : {result.model_dump_json()}")
        print(f"  Signal : {signal.model_dump_json()}")

        ok &= check("is_valid=False",               not result.is_valid)
        ok &= check("validation_stage=row_count",   result.validation_stage == "row_count",
                    result.validation_stage)
        ok &= check("row_count > ROW_COUNT_MAX",
                    result.row_count is not None and result.row_count > ROW_COUNT_MAX,
                    str(result.row_count))
        ok &= check("retry_required=True",          result.retry_required)
        ok &= check("should_regenerate=True",       signal.should_regenerate)
    except Exception as exc:
        ok = False
        print(f"    {FAIL} Unexpected exception: {exc}")

    if ok:
        passed += 1
        print("  --> PASSED")
    else:
        failed_ids.append("T7")
        print("  --> FAILED")

    # ======================================================================
    # RETRY LOGIC TESTS
    # ======================================================================
    print("\n--- RETRY LOGIC TESTS ---")

    # T8 -- Retry count decrements correctly
    total += 1
    print(f"\n[T8] Retry countdown -- retries_used=0 -> retries_remaining={MAX_RETRIES}")
    sql = "SELEC * FROM vw_weekly_sales"   # broken SQL
    ok = True
    try:
        for attempt in range(MAX_RETRIES + 1):
            result = validator.validate(sql, retries_used=attempt)
            signal = validator.create_regeneration_signal(result)
            expected_remaining = max(0, MAX_RETRIES - attempt)
            matches = signal.retries_remaining == expected_remaining
            ok &= check(
                f"Attempt {attempt}: retries_remaining={expected_remaining}",
                matches,
                f"got={signal.retries_remaining}",
            )
    except Exception as exc:
        ok = False
        print(f"    {FAIL} Unexpected exception: {exc}")

    if ok:
        passed += 1
        print("  --> PASSED")
    else:
        failed_ids.append("T8")
        print("  --> FAILED")

    # T9 -- Exhausted retries produce should_regenerate=False
    total += 1
    print(f"\n[T9] Exhausted retries -- retries_used={MAX_RETRIES} -> should_regenerate=False")
    sql = "SELEC * FROM vw_weekly_sales"   # broken SQL
    ok = True
    try:
        result = validator.validate(sql, retries_used=MAX_RETRIES)
        signal = validator.create_regeneration_signal(result)

        print(f"  Result : {result.model_dump_json()}")
        print(f"  Signal : {signal.model_dump_json()}")

        ok &= check("is_valid=False",                  not result.is_valid)
        ok &= check("should_regenerate=False",         not signal.should_regenerate,
                    f"(retries_remaining={signal.retries_remaining})")
        ok &= check("retries_remaining=0",             signal.retries_remaining == 0,
                    str(signal.retries_remaining))
    except Exception as exc:
        ok = False
        print(f"    {FAIL} Unexpected exception: {exc}")

    if ok:
        passed += 1
        print("  --> PASSED")
    else:
        failed_ids.append("T9")
        print("  --> FAILED")

    # ======================================================================
    # Summary
    # ======================================================================
    print("\n" + "=" * 70)
    print(f"OVERALL RESULTS: {passed}/{total} tests passed")
    if failed_ids:
        print(f"FAILED: {failed_ids}")
    else:
        print("All tests passed!")
    print("=" * 70)

    if failed_ids:
        sys.exit(1)


if __name__ == "__main__":
    run_tests()
