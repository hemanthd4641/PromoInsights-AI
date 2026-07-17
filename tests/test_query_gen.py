"""
tests/test_query_gen.py
------------------------
Phase 5 -- Unit Tests for the Query Generation Agent.

Covers:
  - 5 positive SQL generation tests (correct views used)
  - 3 negative whitelist validation tests (raw tables / dangerous SQL blocked)
  - DuckDB syntax validation for every generated query

Run:
    python tests/test_query_gen.py
"""

import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict

# ---------------------------------------------------------------------------
# Bootstrap project root
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Patch Groq LLM to run in mock mode
from tests.mock_llm import patch_groq
patch_groq()

from agents.intent_classifier import Intent
from agents.query_grounding import GroundedIntent, QueryGroundingAgent
from agents.query_gen import (
    QueryGenerationAgent,
    SQLGenerationResult,
    SQLWhitelistViolationError,
    extract_referenced_views,
    validate_whitelist,
    validate_syntax,
)

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
# Pre-built intents (avoids Groq calls for intent classification)
# ---------------------------------------------------------------------------

POSITIVE_TESTS = [
    {
        "id": 1,
        "question": "Did PROMO_001 improve sales in South region?",
        "intent": Intent(topic="promotion", region="South", confidence=0.96),
        "expected_views": ["vw_weekly_sales"],
        "description": "Promotion effectiveness -- should use vw_weekly_sales",
    },
    {
        "id": 2,
        "question": "Did inventory reduce in West region?",
        "intent": Intent(topic="inventory", region="West", confidence=0.92),
        "expected_views": ["vw_weekly_inventory"],
        "description": "Inventory reduction -- should use vw_weekly_inventory",
    },
    {
        "id": 3,
        "question": "Compare North and South sales.",
        "intent": Intent(topic="region_comparison", confidence=0.90),
        "expected_views": ["vw_weekly_sales"],
        "description": "Region comparison -- should use vw_weekly_sales",
    },
    {
        "id": 4,
        "question": "Which campaign performed best?",
        "intent": Intent(topic="campaign_impact", confidence=0.88),
        "expected_views": ["vw_weekly_sales"],
        "description": "Campaign impact -- should use vw_weekly_sales (and optionally vw_promo_calendar)",
    },
    {
        "id": 5,
        "question": "Which category generated highest revenue?",
        "intent": Intent(topic="campaign_impact", confidence=0.87),
        "expected_views": ["vw_weekly_sales"],
        "description": "Revenue ranking -- should use vw_weekly_sales",
    },
]

# Negative tests operate directly on the safety layer (no LLM call needed)
NEGATIVE_TESTS = [
    {
        "id": "N1",
        "sql": "SELECT * FROM sales_raw",
        "description": "Raw table reference -- must fail",
        "expected_violation": "sales_raw",
    },
    {
        "id": "N2",
        "sql": "DROP TABLE vw_weekly_sales",
        "description": "DROP statement -- must fail",
        "expected_violation": "drop",
    },
    {
        "id": "N3",
        "sql": "SELECT * FROM unknown_table WHERE region = 'North'",
        "description": "Unknown table -- must fail",
        "expected_violation": "unknown_table",
    },
]


# ---------------------------------------------------------------------------
# Assertion helpers
# ---------------------------------------------------------------------------

def check_true(label: str, condition: bool, detail: str = "") -> bool:
    tag = PASS if condition else FAIL
    suffix = f" -- {detail}" if detail else ""
    print(f"    {tag} {label}{suffix}")
    return condition


def check_any_view_match(
    label: str, used_views: list, expected: list
) -> bool:
    """At least one expected view must appear in used_views."""
    matched = any(v in used_views for v in expected)
    if matched:
        print(f"    {PASS} {label}: {used_views} contains one of {expected}")
    else:
        print(f"    {FAIL} {label}: {used_views} does not contain any of {expected}")
    return matched


# ---------------------------------------------------------------------------
# Test Runners
# ---------------------------------------------------------------------------

def run_positive_tests(grounder: QueryGroundingAgent, gen: QueryGenerationAgent) -> tuple:
    total = len(POSITIVE_TESTS)
    passed = 0
    failed_ids = []

    print("\n" + "=" * 70)
    print("POSITIVE TESTS -- SQL Generation")
    print("=" * 70)

    for tc in POSITIVE_TESTS:
        test_id = tc["id"]
        question = tc["question"]
        intent: Intent = tc["intent"]

        print(f"\n[Test {test_id}] {question}")
        print(f"  ({tc['description']})")

        checks_ok = True

        # Ground the intent
        try:
            grounded: GroundedIntent = grounder.ground(question, intent)
        except Exception as exc:
            print(f"    {FAIL} Grounding failed: {exc}")
            failed_ids.append(test_id)
            continue

        # Generate SQL
        try:
            result: SQLGenerationResult = gen.generate_sql(question, grounded)
        except SQLWhitelistViolationError as exc:
            print(f"    {FAIL} Whitelist violation (unexpected): {exc}")
            failed_ids.append(test_id)
            continue
        except Exception as exc:
            print(f"    {FAIL} Exception: {exc}")
            failed_ids.append(test_id)
            continue

        # Print SQL
        print(f"  SQL: {result.sql[:120].strip()}...")
        print(f"  Used views: {result.used_views}")

        # 1. Is SQLGenerationResult instance?
        if not check_true("Is SQLGenerationResult", isinstance(result, SQLGenerationResult)):
            checks_ok = False

        # 2. SQL not empty
        if not check_true("SQL not empty", bool(result.sql and result.sql.strip())):
            checks_ok = False

        # 3. Whitelist compliant
        if not check_true("is_whitelist_compliant", result.is_whitelist_compliant):
            checks_ok = False

        # 4. At least one expected view used
        if not check_any_view_match("Expected views", result.used_views, tc["expected_views"]):
            checks_ok = False

        # 5. Valid JSON round-trip
        try:
            json.loads(result.model_dump_json())
            check_true("Valid JSON", True)
        except Exception as e:
            check_true("Valid JSON", False, str(e))
            checks_ok = False

        # 6. DuckDB syntax validation
        syntax_ok = validate_syntax(result.sql)
        if not check_true("DuckDB syntax valid", syntax_ok):
            checks_ok = False

        if checks_ok:
            passed += 1
            print(f"  --> PASSED")
        else:
            failed_ids.append(test_id)
            print(f"  --> FAILED")

    return passed, total, failed_ids


def run_negative_tests() -> tuple:
    total = len(NEGATIVE_TESTS)
    passed = 0
    failed_ids = []

    print("\n" + "=" * 70)
    print("NEGATIVE TESTS -- Whitelist Validation")
    print("=" * 70)

    for tc in NEGATIVE_TESTS:
        test_id = tc["id"]
        sql = tc["sql"]

        print(f"\n[Test {test_id}] {tc['description']}")
        print(f"  SQL: {sql}")

        checks_ok = True
        is_compliant, violations = validate_whitelist(sql)

        # Expect validation to FAIL (is_compliant must be False)
        if not check_true(
            "Validation correctly rejected",
            not is_compliant,
            f"violations={violations}",
        ):
            checks_ok = False

        # Expected violation string should appear in violations
        expected_v = tc["expected_violation"].lower()
        found = any(expected_v in v.lower() for v in violations)
        if not check_true(
            f"Expected violation '{expected_v}' detected",
            found,
            f"got={violations}",
        ):
            checks_ok = False

        if checks_ok:
            passed += 1
            print(f"  --> PASSED (correctly rejected)")
        else:
            failed_ids.append(test_id)
            print(f"  --> FAILED")

    return passed, total, failed_ids


def run_utility_tests() -> tuple:
    """Quick unit tests for extract_referenced_views."""
    total = 4
    passed = 0
    failed_ids = []

    print("\n" + "=" * 70)
    print("UTILITY TESTS -- extract_referenced_views()")
    print("=" * 70)

    cases = [
        ("SELECT * FROM vw_weekly_sales", ["vw_weekly_sales"], "U1"),
        ("SELECT * FROM vw_weekly_sales s JOIN vw_promo_calendar p ON s.promo_id = p.promo_id",
         ["vw_weekly_sales", "vw_promo_calendar"], "U2"),
        ("SELECT * FROM sales_raw", ["sales_raw"], "U3"),
        ("SELECT 1", [], "U4"),
    ]

    for sql, expected_set, uid in cases:
        result = extract_referenced_views(sql)
        result_set = set(result)
        expected = set(expected_set)
        ok = expected.issubset(result_set)
        tag = PASS if ok else FAIL
        print(f"\n[{uid}] SQL: {sql[:60]}")
        print(f"    {tag} extracted={result}, expected_subset={expected_set}")
        if ok:
            passed += 1
        else:
            failed_ids.append(uid)

    return passed, total, failed_ids


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    grounder = QueryGroundingAgent()
    gen = QueryGenerationAgent()

    p1, t1, f1 = run_positive_tests(grounder, gen)
    p2, t2, f2 = run_negative_tests()
    p3, t3, f3 = run_utility_tests()

    total_passed = p1 + p2 + p3
    total_tests = t1 + t2 + t3
    all_failed = f1 + f2 + f3

    print("\n" + "=" * 70)
    print(f"OVERALL RESULTS: {total_passed}/{total_tests} tests passed")
    if all_failed:
        print(f"FAILED: {all_failed}")
    else:
        print("All tests passed!")
    print("=" * 70)

    if all_failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
