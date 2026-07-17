"""
tests/test_intent_classifier.py
--------------------------------
Phase 3 — Unit Tests for the Intent Classifier Agent.

Tests all 6 required sample questions against expected topic and
entity extractions. Validates Pydantic model constraints.

Run:
    python tests/test_intent_classifier.py
"""

import json
import logging
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Bootstrap project root
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Patch Groq LLM to run in mock mode
from tests.mock_llm import patch_groq
patch_groq()

from agents.intent_classifier import Intent, IntentClassifier

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Test Cases
# ---------------------------------------------------------------------------

TEST_CASES = [
    {
        "id": 1,
        "question": "Did PROMO_001 improve sales in South region?",
        "expected_topic": "promotion",
        "expected_region": "South",
        "expected_category": None,
    },
    {
        "id": 2,
        "question": "Which region performed best during PROMO_002?",
        "expected_topic": "region_comparison",
        "expected_region": None,
        "expected_category": None,
    },
    {
        "id": 3,
        "question": "Compare North and South sales during PROMO_003.",
        "expected_topic": "region_comparison",
        "expected_region": None,   # Multiple regions — accept null at top level
        "expected_category": None,
    },
    {
        "id": 4,
        "question": "Which category reacted best to the summer campaign?",
        "expected_topic": "campaign_impact",
        "expected_region": None,
        "expected_category": None,
    },
    {
        "id": 5,
        "question": "Did inventory reduce for Electronics products in West region?",
        "expected_topic": "inventory",
        "expected_region": "West",
        "expected_category": "Electronics",
    },
    {
        "id": 6,
        "question": "Which SKU generated the highest revenue last quarter?",
        "expected_topic": "campaign_impact",
        "expected_region": None,
        "expected_category": None,
    },
]

# ---------------------------------------------------------------------------
# Assertion helpers
# ---------------------------------------------------------------------------

PASS = "[PASS]"
FAIL = "[FAIL]"


def check(label: str, actual, expected, strict: bool = True) -> bool:
    """
    Assert actual == expected (or skip if strict=False and expected is None).
    Returns True if the check passed.
    """
    if not strict and expected is None:
        # Only validate that actual has the right type, don't require a value
        print(f"    {PASS} {label}: {actual!r} (not required)")
        return True

    if actual == expected:
        print(f"    {PASS} {label}: {actual!r}")
        return True
    else:
        print(f"    {FAIL} {label}: expected={expected!r}, got={actual!r}")
        return False


def run_tests() -> None:
    """Run all test cases and report results."""
    clf = IntentClassifier()

    total = len(TEST_CASES)
    passed = 0
    failed_ids = []

    print("\n" + "=" * 70)
    print("PHASE 3 — INTENT CLASSIFIER TEST SUITE")
    print("=" * 70)

    for tc in TEST_CASES:
        test_id = tc["id"]
        question = tc["question"]
        print(f"\n[Test {test_id}] {question}")

        # --- Classify --------------------------------------------------
        try:
            intent: Intent = clf.classify(question)
        except Exception as exc:
            print(f"    {FAIL} Exception raised: {exc}")
            failed_ids.append(test_id)
            continue

        # --- Print raw output ------------------------------------------
        print(f"  Output: {intent.model_dump_json()}")

        # --- Structural checks (always required) -----------------------
        checks_passed = True

        # 1. Valid Intent object
        if not isinstance(intent, Intent):
            print(f"    {FAIL} Output is not an Intent instance")
            checks_passed = False

        # 2. Valid JSON round-trip
        try:
            parsed = json.loads(intent.model_dump_json())
            assert "topic" in parsed and "confidence" in parsed
            print(f"    {PASS} Valid JSON structure")
        except Exception as e:
            print(f"    {FAIL} JSON validation failed: {e}")
            checks_passed = False

        # 3. Confidence in range
        if 0.0 <= intent.confidence <= 1.0:
            print(f"    {PASS} Confidence in range: {intent.confidence}")
        else:
            print(f"    {FAIL} Confidence out of range: {intent.confidence}")
            checks_passed = False

        # 4. Topic check (strict)
        if not check("topic", intent.topic, tc["expected_topic"], strict=True):
            checks_passed = False

        # 5. Region check (strict only when expected is not None)
        if tc["expected_region"] is not None:
            if not check("region", intent.region, tc["expected_region"], strict=True):
                checks_passed = False
        else:
            print(f"    {PASS} region: {intent.region!r} (not required)")

        # 6. Category check (strict only when expected is not None)
        if tc["expected_category"] is not None:
            if not check("category", intent.category, tc["expected_category"], strict=True):
                checks_passed = False
        else:
            print(f"    {PASS} category: {intent.category!r} (not required)")

        # --- Result ----------------------------------------------------
        if checks_passed:
            passed += 1
            print(f"  --> PASSED")
        else:
            failed_ids.append(test_id)
            print(f"  --> FAILED")

    # --- Summary -------------------------------------------------------
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
