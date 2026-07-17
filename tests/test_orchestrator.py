"""
tests/test_orchestrator.py
---------------------------
Phase 9 -- Unit Tests for the Orchestrator.

Run:
    python tests/test_orchestrator.py
"""

import logging
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# Bootstrap project root
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from agents.orchestrator import PromotionAnalyticsOrchestrator
from agents.synthesizer import SynthesizedResponse

logging.basicConfig(level=logging.WARNING)

PASS = "[PASS]"
FAIL = "[FAIL]"

def check(label: str, condition: bool, detail: str = "") -> bool:
    tag = PASS if condition else FAIL
    suffix = f" -- {detail}" if detail else ""
    print(f"    {tag} {label}{suffix}")
    return condition

def run_tests():
    total = 0
    passed = 0
    failed_ids = []

    print("\n" + "=" * 70)
    print("PHASE 9 -- ORCHESTRATOR TEST SUITE")
    print("=" * 70)

    # We will mock the heavy LLM/DB calls where possible to keep unit tests fast and deterministic.
    # But for T1-T4 we want the actual pipeline to execute, so we rely on the DB and models.
    # Note: in a true CI environment, we would fully mock LLMs. Here we run them as end-to-end integration tests.
    
    orchestrator = PromotionAnalyticsOrchestrator()

    # ------------------------------------------------------------------
    # T1: Promotion Question
    # ------------------------------------------------------------------
    total += 1
    print("\n[T1] Promotion Question (End-to-End)")
    ok = True
    try:
        res = orchestrator.handle("Did PROMO_001 improve sales in South region?", session_id="t1")
        ok &= check("Returns SynthesizedResponse", isinstance(res, SynthesizedResponse))
        ok &= check("Pipeline succeeded (no fallback)", "clarification" not in res.explanation.lower() and "validation failed" not in res.explanation.lower())
    except Exception as exc:
        ok = False
        print(f"    {FAIL} Exception: {exc}")
    if ok: passed += 1
    else: failed_ids.append("T1")

    # ------------------------------------------------------------------
    # T2: Inventory Question
    # ------------------------------------------------------------------
    total += 1
    print("\n[T2] Inventory Question (End-to-End)")
    ok = True
    try:
        res = orchestrator.handle("Did inventory reduce for Electronics in West region?", session_id="t2")
        ok &= check("Returns SynthesizedResponse", isinstance(res, SynthesizedResponse))
        ok &= check("Pipeline succeeded", "clarification" not in res.explanation.lower())
    except Exception as exc:
        ok = False
        print(f"    {FAIL} Exception: {exc}")
    if ok: passed += 1
    else: failed_ids.append("T2")

    # ------------------------------------------------------------------
    # T3: Region Comparison
    # ------------------------------------------------------------------
    total += 1
    print("\n[T3] Region Comparison (End-to-End)")
    ok = True
    try:
        res = orchestrator.handle("Compare North and South sales", session_id="t3")
        ok &= check("Returns SynthesizedResponse", isinstance(res, SynthesizedResponse))
        ok &= check("Pipeline succeeded", "clarification" not in res.explanation.lower())
    except Exception as exc:
        ok = False
        print(f"    {FAIL} Exception: {exc}")
    if ok: passed += 1
    else: failed_ids.append("T3")

    # ------------------------------------------------------------------
    # T4: Campaign Impact
    # ------------------------------------------------------------------
    total += 1
    print("\n[T4] Campaign Impact (End-to-End)")
    ok = True
    try:
        res = orchestrator.handle("Which SKU generated the most revenue?", session_id="t4")
        ok &= check("Returns SynthesizedResponse", isinstance(res, SynthesizedResponse))
        ok &= check("Pipeline succeeded", "clarification" not in res.explanation.lower())
    except Exception as exc:
        ok = False
        print(f"    {FAIL} Exception: {exc}")
    if ok: passed += 1
    else: failed_ids.append("T4")

    # ------------------------------------------------------------------
    # T5: Low Confidence Test
    # ------------------------------------------------------------------
    total += 1
    print("\n[T5] Low Confidence Test")
    ok = True
    try:
        res = orchestrator.handle("Tell me something interesting", session_id="t5")
        ok &= check("Clarification requested", "clarification" in res.answer_text.lower())
        ok &= check("Pipeline stopped (explanation set)", "ambiguous" in res.explanation.lower())
    except Exception as exc:
        ok = False
        print(f"    {FAIL} Exception: {exc}")
    if ok: passed += 1
    else: failed_ids.append("T5")

    # ------------------------------------------------------------------
    # T6: Validation Failure Test
    # ------------------------------------------------------------------
    total += 1
    print("\n[T6] Validation Failure Loop")
    ok = True
    try:
        # We can force a validation failure by using a mock on the validator
        from agents.validator import ValidationResult
        with patch.object(orchestrator.validator, 'validate') as mock_validate:
            # Return a valid Pydantic object so create_regeneration_signal doesn't crash on MagicMocks
            def side_effect(sql, retries_used=0):
                return ValidationResult(
                    is_valid=False,
                    validation_stage="syntax",
                    failure_reason="Mock syntax error",
                    row_count=None,
                    retry_required=True,
                    retries_used=retries_used,
                    sql=sql
                )
            mock_validate.side_effect = side_effect
            
            res = orchestrator.handle("Compare North and South sales", session_id="t6")
            
            ok &= check("Returns Fallback", "reliable analysis" in res.answer_text.lower() or "clarification" in res.answer_text.lower())
            ok &= check("Explanation specifies validation failure", "validation failed" in res.explanation.lower())
            ok &= check("Retries occurred", mock_validate.call_count > 1)
    except Exception as exc:
        ok = False
        print(f"    {FAIL} Exception: {exc}")
    if ok: passed += 1
    else: failed_ids.append("T6")

    # ------------------------------------------------------------------
    # T7: Session Memory Test (Multi-turn)
    # ------------------------------------------------------------------
    total += 1
    print("\n[T7] Session Memory Multi-Turn")
    ok = True
    try:
        orchestrator.handle("How did South perform?", session_id="t7")
        # Ensure context stored
        session = orchestrator.session_memory.get_session("t7")
        ok &= check("Context stored", session.get("last_intent") is not None)
        
        # Second query (uses context)
        res = orchestrator.handle("What about North?", session_id="t7")
        ok &= check("Pipeline didn't crash", isinstance(res, SynthesizedResponse))
    except Exception as exc:
        ok = False
        print(f"    {FAIL} Exception: {exc}")
    if ok: passed += 1
    else: failed_ids.append("T7")

    print("\n" + "=" * 70)
    print(f"OVERALL RESULTS: {passed}/{total} tests passed")
    if failed_ids:
        print(f"FAILED: {failed_ids}")
        sys.exit(1)
    else:
        print("All tests passed!")
    print("=" * 70)


if __name__ == "__main__":
    run_tests()
