"""
tests/test_synthesizer.py
--------------------------
Phase 8 -- Unit Tests for Response Synthesis Agent.

Run:
    python tests/test_synthesizer.py
"""

import logging
import sys
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Bootstrap project root
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from agents.executor import ExecutionMetadata
from agents.query_grounding import GroundedIntent
from agents.synthesizer import ResponseSynthesizer, SynthesizedResponse

logging.basicConfig(level=logging.WARNING)

PASS = "[PASS]"
FAIL = "[FAIL]"

def check(label: str, condition: bool, detail: str = "") -> bool:
    tag = PASS if condition else FAIL
    suffix = f" -- {detail}" if detail else ""
    print(f"    {tag} {label}{suffix}")
    return condition

def run_tests():
    synth = ResponseSynthesizer()
    total = 0
    passed = 0
    failed_ids = []

    print("\n" + "=" * 70)
    print("PHASE 8 -- SYNTHESIS AGENT TEST SUITE")
    print("=" * 70)

    # ------------------------------------------------------------------
    # T1: Promotion Effectiveness
    # ------------------------------------------------------------------
    total += 1
    print("\n[T1] Promotion Effectiveness")
    ok = True
    try:
        df = pd.DataFrame({
            "week": [1, 2],
            "revenue": [100, 120],
            "delta": [float('nan'), 20.0],
            "pct_change": [float('nan'), 20.0],
        })
        intent = GroundedIntent(topic="promotion", confidence=1.0, metric_definition="effectiveness")
        meta = ExecutionMetadata()
        sql = "SELECT * FROM t"

        res = synth.synthesize(df, intent, meta, sql)
        ok &= check("Returns SynthesizedResponse", isinstance(res, SynthesizedResponse))
        ok &= check("Answer generated", bool(res.answer_text))
        ok &= check("Delta extracted", res.delta == 20.0)
        ok &= check("Pct_change extracted", res.pct_change == 20.0)
    except Exception as exc:
        ok = False
        print(f"    {FAIL} Exception: {exc}")
    if ok: passed += 1
    else: failed_ids.append("T1")

    # ------------------------------------------------------------------
    # T2: Inventory Reduction
    # ------------------------------------------------------------------
    total += 1
    print("\n[T2] Inventory Reduction")
    ok = True
    try:
        df = pd.DataFrame({"week": [1], "stock": [50]})
        intent = GroundedIntent(topic="inventory", confidence=1.0, metric_definition="reduction")
        meta = ExecutionMetadata()
        res = synth.synthesize(df, intent, meta, "")
        ok &= check("Explanation generated", "inventory" in res.explanation.lower() and "reduction" in res.explanation.lower())
    except Exception as exc:
        ok = False
        print(f"    {FAIL} Exception: {exc}")
    if ok: passed += 1
    else: failed_ids.append("T2")

    # ------------------------------------------------------------------
    # T3: Region Comparison
    # ------------------------------------------------------------------
    total += 1
    print("\n[T3] Region Comparison")
    ok = True
    try:
        df = pd.DataFrame({"region": ["North", "South"], "rev": [10, 20]})
        intent = GroundedIntent(topic="region_comparison", confidence=1.0, metric_definition="regional_performance")
        meta = ExecutionMetadata()
        res = synth.synthesize(df, intent, meta, "")
        ok &= check("Comparison summary", "comparison" in res.answer_text.lower())
    except Exception as exc:
        ok = False
        print(f"    {FAIL} Exception: {exc}")
    if ok: passed += 1
    else: failed_ids.append("T3")

    # ------------------------------------------------------------------
    # T4: Campaign Impact
    # ------------------------------------------------------------------
    total += 1
    print("\n[T4] Campaign Impact")
    ok = True
    try:
        df = pd.DataFrame({"promo_id": ["A"], "rev": [10]})
        intent = GroundedIntent(topic="campaign_impact", confidence=1.0, metric_definition="campaign_impact")
        meta = ExecutionMetadata()
        res = synth.synthesize(df, intent, meta, "")
        ok &= check("Top performer text", "impact" in res.answer_text.lower())
    except Exception as exc:
        ok = False
        print(f"    {FAIL} Exception: {exc}")
    if ok: passed += 1
    else: failed_ids.append("T4")

    # ------------------------------------------------------------------
    # T5: Partial Weeks Coverage
    # ------------------------------------------------------------------
    total += 1
    print("\n[T5] Partial Weeks Coverage (Weeks 1-4)")
    ok = True
    try:
        df = pd.DataFrame({"week": [1, 2, 3, 4], "val": [1, 2, 3, 4]})
        intent = GroundedIntent(topic="promotion", confidence=1.0, metric_definition="test")
        meta = ExecutionMetadata()
        res = synth.synthesize(df, intent, meta, "")
        ok &= check("is_partial=True", res.coverage_flag.is_partial)
        ok &= check("missing_weeks detected (5..52)", 5 in res.coverage_flag.missing_weeks and 52 in res.coverage_flag.missing_weeks)
    except Exception as exc:
        ok = False
        print(f"    {FAIL} Exception: {exc}")
    if ok: passed += 1
    else: failed_ids.append("T5")

    # ------------------------------------------------------------------
    # T6: Partial Regions Coverage
    # ------------------------------------------------------------------
    total += 1
    print("\n[T6] Partial Regions Coverage (North, South)")
    ok = True
    try:
        df = pd.DataFrame({"region": ["North", "South"], "val": [1, 2]})
        intent = GroundedIntent(topic="promotion", confidence=1.0, metric_definition="test")
        meta = ExecutionMetadata()
        res = synth.synthesize(df, intent, meta, "")
        ok &= check("is_partial=True", res.coverage_flag.is_partial)
        ok &= check("East is missing", "East" in res.coverage_flag.missing_regions)
        ok &= check("West is missing", "West" in res.coverage_flag.missing_regions)
    except Exception as exc:
        ok = False
        print(f"    {FAIL} Exception: {exc}")
    if ok: passed += 1
    else: failed_ids.append("T6")

    # ------------------------------------------------------------------
    # T7: Empty Data Test
    # ------------------------------------------------------------------
    total += 1
    print("\n[T7] Empty DataFrame")
    ok = True
    try:
        df = pd.DataFrame()
        intent = GroundedIntent(topic="promotion", confidence=1.0, metric_definition="test")
        meta = ExecutionMetadata()
        res = synth.synthesize(df, intent, meta, "SELECT 1")
        ok &= check("No exception", True)
        ok &= check("Empty message", "No data available" in res.answer_text)
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
