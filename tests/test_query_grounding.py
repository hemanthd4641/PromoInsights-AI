"""
tests/test_query_grounding.py
------------------------------
Phase 4 — Unit Tests for the Query Grounding Agent.

Tests that the grounding agent resolves the correct metric_definition,
baseline_formula, and comparison_window for 4 representative questions.
Also validates the GroundedIntent schema, few-shot attachment, and
deterministic consistency on repeated calls.

Run:
    python tests/test_query_grounding.py
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

from agents.intent_classifier import Intent, IntentClassifier
from agents.query_grounding import GroundedIntent, QueryGroundingAgent

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.WARNING,          # Suppress INFO noise during tests
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

PASS = "[PASS]"
FAIL = "[FAIL]"

# ---------------------------------------------------------------------------
# Test Cases
# keyword_checks: list of (field, substring_to_find)
# These are substring checks — any matching substring in the field = pass.
# ---------------------------------------------------------------------------

TEST_CASES = [
    {
        "id": 1,
        "question": "Did PROMO_001 improve sales in South region?",
        "pre_classified_intent": Intent(
            topic="promotion",
            region="South",
            sku=None,
            category=None,
            time_window=None,
            confidence=0.96,
        ),
        "keyword_checks": [
            ("metric_definition", "effectiveness"),
            ("baseline_formula", "baseline"),
            ("comparison_window", "baseline"),
        ],
        "description": "Promotion question -- effectiveness metric",
    },
    {
        "id": 2,
        "question": "Did inventory reduction happen in West region?",
        "pre_classified_intent": Intent(
            topic="inventory",
            region="West",
            sku=None,
            category=None,
            time_window=None,
            confidence=0.92,
        ),
        "keyword_checks": [
            ("metric_definition", "reduction"),
            ("comparison_window", "delta"),
        ],
        "description": "Inventory reduction -- negative delta metric",
    },
    {
        "id": 3,
        "question": "Which campaign performed best?",
        "pre_classified_intent": Intent(
            topic="campaign_impact",
            region=None,
            sku=None,
            category=None,
            time_window=None,
            confidence=0.88,
        ),
        "keyword_checks": [
            ("metric_definition", "campaign"),
        ],
        "description": "Best campaign -- campaign_impact metric",
    },
    {
        "id": 4,
        "question": "Show revenue growth for Electronics category.",
        "pre_classified_intent": Intent(
            topic="campaign_impact",
            region=None,
            sku=None,
            category="Electronics",
            time_window=None,
            confidence=0.85,
        ),
        "keyword_checks": [
            ("metric_definition", "growth"),
        ],
        "description": "Revenue growth -- revenue_growth metric",
    },
]


# ---------------------------------------------------------------------------
# Assertion helpers
# ---------------------------------------------------------------------------

def check_contains(label: str, actual: Any, expected_substr: str) -> bool:
    """Check that expected_substr appears in str(actual) (case-insensitive)."""
    if actual is None:
        print(f"    {FAIL} {label}: got None, expected to contain {expected_substr!r}")
        return False
    if expected_substr.lower() in str(actual).lower():
        print(f"    {PASS} {label}: {str(actual)[:80]!r} contains {expected_substr!r}")
        return True
    print(f"    {FAIL} {label}: {str(actual)[:80]!r} does NOT contain {expected_substr!r}")
    return False


def check_true(label: str, condition: bool, detail: str = "") -> bool:
    if condition:
        print(f"    {PASS} {label}{(' — ' + detail) if detail else ''}")
    else:
        print(f"    {FAIL} {label}{(' — ' + detail) if detail else ''}")
    return condition


# ---------------------------------------------------------------------------
# Main Test Runner
# ---------------------------------------------------------------------------

def run_tests() -> None:
    agent = QueryGroundingAgent()

    total = len(TEST_CASES)
    passed = 0
    failed_ids = []

    print("\n" + "=" * 70)
    print("PHASE 4 — QUERY GROUNDING AGENT TEST SUITE")
    print("=" * 70)

    for tc in TEST_CASES:
        test_id = tc["id"]
        question = tc["question"]
        intent: Intent = tc["pre_classified_intent"]

        print(f"\n[Test {test_id}] {question}")
        print(f"  ({tc['description']})")

        # --- Ground the intent -------------------------------------------
        try:
            grounded: GroundedIntent = agent.ground(question, intent)
        except Exception as exc:
            print(f"    {FAIL} Exception raised: {exc}")
            failed_ids.append(test_id)
            continue

        # --- Print output --------------------------------------------------
        output = grounded.model_dump()
        # Show compact output (hide few_shot details for brevity)
        compact = {k: v for k, v in output.items() if k != "few_shot_examples"}
        compact["few_shot_examples_count"] = len(output.get("few_shot_examples", []))
        print(f"  Output: {json.dumps(compact)}")

        checks_ok = True

        # 1. Is a GroundedIntent instance?
        if not check_true("Is GroundedIntent instance", isinstance(grounded, GroundedIntent)):
            checks_ok = False

        # 2. Valid JSON round-trip
        try:
            parsed: Dict = json.loads(grounded.model_dump_json())
            assert "metric_definition" in parsed
            assert "topic" in parsed
            check_true("Valid JSON structure", True)
        except Exception as e:
            check_true("Valid JSON structure", False, str(e))
            checks_ok = False

        # 3. metric_definition not empty
        if not check_true(
            "metric_definition not empty",
            bool(grounded.metric_definition and grounded.metric_definition.strip()),
            grounded.metric_definition,
        ):
            checks_ok = False

        # 4. Confidence in range
        if not check_true(
            "Confidence in range",
            0.0 <= grounded.confidence <= 1.0,
            str(grounded.confidence),
        ):
            checks_ok = False

        # 5. few_shot_examples list (0–3 items)
        n_ex = len(grounded.few_shot_examples)
        if not check_true(
            "few_shot_examples count",
            0 <= n_ex <= 3,
            f"{n_ex} examples",
        ):
            checks_ok = False

        # 6. Keyword-specific field checks
        for field, expected_substr in tc["keyword_checks"]:
            val = getattr(grounded, field, None)
            if not check_contains(field, val, expected_substr):
                checks_ok = False

        # 7. Determinism: run a second time — results must be identical
        try:
            grounded2: GroundedIntent = agent.ground(question, intent)
            same = (
                grounded.metric_definition == grounded2.metric_definition
                and grounded.baseline_formula == grounded2.baseline_formula
                and grounded.comparison_window == grounded2.comparison_window
            )
            if not check_true("Deterministic (same result on 2nd call)", same):
                checks_ok = False
        except Exception as exc:
            check_true("Deterministic (2nd call)", False, str(exc))
            checks_ok = False

        # --- Result -------------------------------------------------------
        if checks_ok:
            passed += 1
            print(f"  --> PASSED")
        else:
            failed_ids.append(test_id)
            print(f"  --> FAILED")

    # --- Summary ----------------------------------------------------------
    print("\n" + "=" * 70)
    print(f"RESULTS: {passed}/{total} tests passed")
    if failed_ids:
        print(f"FAILED tests: {failed_ids}")
    else:
        print("All tests passed!")
    print("=" * 70)

    if failed_ids:
        sys.exit(1)


if __name__ == "__main__":
    run_tests()
