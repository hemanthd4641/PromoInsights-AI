"""
tests/test_pipeline.py
-----------------------
Phase 12 — End-to-End Pipeline Integration Tests.

Validates the complete Promotion Analytics pipeline:

  Question
  → Intent Classification
  → Query Grounding
  → SQL Generation
  → Validation
  → Execution
  → Response Synthesis
  → Final Structured Response

Uses PromotionAnalyticsOrchestrator.handle() as the single test entry-point.
All 10 questions are run sequentially to avoid Groq rate-limit errors.

Run:
    python tests/test_pipeline.py
    pytest tests/test_pipeline.py -v
"""

import sys
import time
from pathlib import Path
from typing import NamedTuple

# ---------------------------------------------------------------------------
# Bootstrap project root
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Patch Groq LLM to run in mock mode
from tests.mock_llm import patch_groq
patch_groq()

from agents.orchestrator import PromotionAnalyticsOrchestrator
from agents.synthesizer import SynthesizedResponse

# ---------------------------------------------------------------------------
# Shared orchestrator (single instance to reuse warm caches / model init)
# ---------------------------------------------------------------------------
ORCHESTRATOR: PromotionAnalyticsOrchestrator | None = None

PASS = "[PASS]"
FAIL = "[FAIL]"


def get_orchestrator() -> PromotionAnalyticsOrchestrator:
    global ORCHESTRATOR
    if ORCHESTRATOR is None:
        ORCHESTRATOR = PromotionAnalyticsOrchestrator()
    return ORCHESTRATOR


# ---------------------------------------------------------------------------
# Test result model
# ---------------------------------------------------------------------------

class TestResult(NamedTuple):
    name: str
    passed: bool
    latency_ms: float
    validation_passed: bool
    response_generated: bool


# ---------------------------------------------------------------------------
# Assertion helpers
# ---------------------------------------------------------------------------

def check(label: str, condition: bool, detail: str = "") -> bool:
    tag = PASS if condition else FAIL
    suffix = f" -- {detail}" if detail else ""
    print(f"    {tag} {label}{suffix}")
    return condition


def run_question(
    name: str,
    question: str,
    expect_fallback: bool = False,
    session_id: str = "test_pipeline",
) -> TestResult:
    """
    Submit a question to the orchestrator and assert expectations.

    Args:
        name:           Human-readable test name.
        question:       The business question to send.
        expect_fallback: True if we expect a clarification/fallback response.
        session_id:     Session tag for the orchestrator.

    Returns:
        TestResult with pass/fail status and metrics.
    """
    print(f"\n{'-' * 60}")
    print(f"  {name}")
    print(f"  Q: {question!r}")

    t0 = time.perf_counter()
    ok = True
    validation_passed = False
    response_generated = False

    try:
        resp: SynthesizedResponse = get_orchestrator().handle(
            question=question,
            session_id=session_id,
        )
        latency_ms = (time.perf_counter() - t0) * 1000

        ok &= check("Returns SynthesizedResponse", isinstance(resp, SynthesizedResponse))
        ok &= check("answer_text not empty", bool(resp.answer_text.strip()))
        ok &= check("explanation not empty", bool(resp.explanation.strip()))
        ok &= check("coverage_flag present", resp.coverage_flag is not None)

        if expect_fallback:
            # Ambiguous questions should return clarification, not crash
            ok &= check(
                "Clarification/fallback returned",
                "clarification" in resp.answer_text.lower()
                or "reliable" in resp.answer_text.lower()
                or "ambiguous" in resp.explanation.lower()
                or "unexpected" in resp.explanation.lower()
                or len(resp.table) == 0,
            )
            response_generated = False  # by design
            validation_passed = False
        else:
            # Valid business questions should return real data
            ok &= check("SQL non-empty", bool(resp.sql_shown.strip()))
            ok &= check("Table returned (list)", isinstance(resp.table, list))
            validation_passed = bool(resp.sql_shown.strip())
            response_generated = bool(resp.answer_text.strip()) and bool(resp.sql_shown.strip())

    except Exception as exc:
        latency_ms = (time.perf_counter() - t0) * 1000
        ok = False
        check("No exception thrown", False, str(exc))

    status = PASS if ok else FAIL
    print(f"  {status} Overall | Latency: {latency_ms:.0f} ms")
    return TestResult(
        name=name,
        passed=ok,
        latency_ms=latency_ms,
        validation_passed=validation_passed,
        response_generated=response_generated,
    )


# ---------------------------------------------------------------------------
# Test Cases
# ---------------------------------------------------------------------------

def run_all_tests() -> list[TestResult]:
    results: list[TestResult] = []

    print("\n" + "=" * 70)
    print("PHASE 12 - END-TO-END PIPELINE TEST SUITE")
    print("=" * 70)
    print("Initialising orchestrator (warm-up may take 5-10 s)...")
    get_orchestrator()

    # ------------------------------------------------------------------
    # PROMOTION TESTS
    # ------------------------------------------------------------------

    results.append(run_question(
        name="[P1] Promotion - PROMO_001 in South",
        question="Did PROMO_001 improve sales in South region?",
        session_id="e2e_promo",
    ))

    # Brief pause between LLM calls to respect Groq rate limits
    time.sleep(3)

    results.append(run_question(
        name="[P2] Promotion - PROMO_002 vs baseline",
        question="How effective was PROMO_002 compared to baseline?",
        session_id="e2e_promo",
    ))

    time.sleep(3)

    # ------------------------------------------------------------------
    # INVENTORY TESTS
    # ------------------------------------------------------------------

    results.append(run_question(
        name="[I1] Inventory - reduction in West",
        question="Did inventory reduce in West region?",
        session_id="e2e_inventory",
    ))

    time.sleep(3)

    results.append(run_question(
        name="[I2] Inventory - highest stock levels",
        question="Which products have the highest stock levels?",
        session_id="e2e_inventory",
    ))

    time.sleep(3)

    # ------------------------------------------------------------------
    # REGION COMPARISON TESTS
    # ------------------------------------------------------------------

    results.append(run_question(
        name="[R1] Region - North vs South sales",
        question="Compare North and South sales.",
        session_id="e2e_region",
    ))

    time.sleep(3)

    results.append(run_question(
        name="[R2] Region - highest revenue region",
        question="Which region generated the highest revenue?",
        session_id="e2e_region",
    ))

    time.sleep(3)

    # ------------------------------------------------------------------
    # CAMPAIGN IMPACT TESTS
    # ------------------------------------------------------------------

    results.append(run_question(
        name="[C1] Campaign - best performing campaign",
        question="Which campaign performed best?",
        session_id="e2e_campaign",
    ))

    time.sleep(3)

    results.append(run_question(
        name="[C2] Campaign - highest revenue category",
        question="Which category generated highest revenue?",
        session_id="e2e_campaign",
    ))

    time.sleep(3)

    # ------------------------------------------------------------------
    # AMBIGUOUS / BAD INPUT TESTS
    # ------------------------------------------------------------------

    results.append(run_question(
        name="[A1] Ambiguous - tell me something interesting",
        question="Tell me something interesting.",
        expect_fallback=True,
        session_id="e2e_ambiguous",
    ))

    time.sleep(3)

    results.append(run_question(
        name="[A2] Ambiguous - analyze everything",
        question="Analyze everything.",
        expect_fallback=True,
        session_id="e2e_ambiguous",
    ))

    return results


# ---------------------------------------------------------------------------
# Performance Summary
# ---------------------------------------------------------------------------

def print_summary(results: list[TestResult]) -> None:
    total = len(results)
    passed = sum(r.passed for r in results)
    failed = [r.name for r in results if not r.passed]

    valid_latencies = [r.latency_ms for r in results if r.validation_passed]
    avg_latency = sum(valid_latencies) / len(valid_latencies) if valid_latencies else 0.0
    val_success = sum(r.validation_passed for r in results)
    resp_success = sum(r.response_generated for r in results)

    print("\n" + "=" * 70)
    print("PIPELINE TEST SUMMARY")
    print("=" * 70)

    # Detailed table
    print(f"\n{'Test':<42} {'Status':<8} {'Latency':>10} {'SQL':>5} {'Resp':>5}")
    print("-" * 72)
    for r in results:
        status = PASS if r.passed else FAIL
        lat = f"{r.latency_ms:.0f} ms" if r.latency_ms > 0 else "-"
        sql = "Y" if r.validation_passed else "N"
        resp = "Y" if r.response_generated else "N"
        print(f"  {r.name:<40} {status:<8} {lat:>10} {sql:>5} {resp:>5}")

    print("\n" + "-" * 70)
    print(f"  Overall:           {passed}/{total} tests passed")
    print(f"  Avg Latency:       {avg_latency:.1f} ms (valid queries only)")
    print(f"  SQL Validation:    {val_success}/{total}")
    print(f"  Response Success:  {resp_success}/{total}")

    if failed:
        print(f"\n  FAILED TESTS: {failed}")
    else:
        print("\n  [PASS] All tests passed!")

    print("=" * 70)

    # Exit with non-zero if any test failed
    if failed:
        sys.exit(1)


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    results = run_all_tests()
    print_summary(results)

