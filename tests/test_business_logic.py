"""
tests/test_business_logic.py
-----------------------------
Phase 12 — Business Logic Validation Tests.

Verifies all 6 issue fixes:
  1. Promo queries return valid effectiveness results (not 0 weeks).
  2. Ranking queries show the top winner.
  3. Category highest revenue — no delta shown.
  4. Region comparison — side-by-side with named winner.
  5. Inventory West — Complete Coverage (not missing N/S/E).
  6. Unknown promotion — graceful no-data message.

Run:
    python tests/test_business_logic.py
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from tests.mock_llm import patch_groq
patch_groq()

from agents.orchestrator import PromotionAnalyticsOrchestrator

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
PASS = "[PASS]"
FAIL = "[FAIL]"

def check(label: str, condition: bool, detail: str = "") -> bool:
    status = PASS if condition else FAIL
    suffix = f"  ({detail})" if detail else ""
    print(f"    {status} {label}{suffix}")
    return condition


def run_test(orch, session_id, question):
    print(f"\n  Q: {question!r}")
    return orch.handle(question=question, session_id=session_id)


# ---------------------------------------------------------------------------
# Main Test Runner
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("BUSINESS LOGIC VALIDATION SUITE")
    print("=" * 70)

    orch = PromotionAnalyticsOrchestrator()
    results = {}

    # ── TEST 1: PROMO_001 effectiveness ─────────────────────────────────────
    print("\n" + "-" * 60)
    print("  [BL1] PROMO_001 — Promotion Effectiveness in South")
    resp = run_test(orch, "bl_1", "Did PROMO_001 improve sales in South region?")

    t1 = True
    t1 &= check("Returns SynthesizedResponse", resp is not None)
    t1 &= check("answer_text not empty", bool(resp.answer_text))
    t1 &= check(
        "No '0 weeks' in answer",
        "0 weeks" not in resp.answer_text and "0 weeks" not in resp.explanation,
        detail=resp.explanation[:80],
    )
    t1 &= check(
        "Answer mentions promotion or sales",
        any(w in resp.answer_text.lower() for w in ["promo", "sales", "revenue", "lift", "south"]),
        detail=resp.answer_text[:80],
    )
    t1 &= check("Table returned (list)", isinstance(resp.table, list))
    results["BL1"] = t1
    print(f"  {'[PASS]' if t1 else '[FAIL]'} Overall")

    # ── TEST 2: Best campaign (winner identified) ────────────────────────────
    print("\n" + "-" * 60)
    print("  [BL2] Best Campaign — Winner Identified")
    resp = run_test(orch, "bl_2", "Which campaign performed best?")

    t2 = True
    t2 &= check("Returns SynthesizedResponse", resp is not None)
    t2 &= check("answer_text not empty", bool(resp.answer_text))
    t2 &= check(
        "Answer identifies a winner (mentions 'highest' or 'revenue' or campaign name)",
        any(w in resp.answer_text.lower() for w in ["highest", "revenue", "promo", "campaign", "top"]),
        detail=resp.answer_text[:80],
    )
    t2 &= check("Table returned (list)", isinstance(resp.table, list))
    t2 &= check(
        "No '0 weeks' in explanation",
        "0 weeks" not in resp.explanation,
        detail=resp.explanation[:80],
    )
    results["BL2"] = t2
    print(f"  {'[PASS]' if t2 else '[FAIL]'} Overall")

    # ── TEST 3: Highest revenue category — NO delta ──────────────────────────
    print("\n" + "-" * 60)
    print("  [BL3] Category Highest Revenue — No Delta Shown")
    resp = run_test(orch, "bl_3", "Which category generated highest revenue?")

    t3 = True
    t3 &= check("Returns SynthesizedResponse", resp is not None)
    t3 &= check("answer_text not empty", bool(resp.answer_text))
    t3 &= check(
        "Answer identifies winning category",
        any(w in resp.answer_text.lower() for w in ["groceries", "home", "sports", "fashion", "electronics", "highest", "revenue", "category"]),
        detail=resp.answer_text[:80],
    )
    t3 &= check(
        "delta is None (no delta for ranking query)",
        resp.delta is None,
        detail=f"delta={resp.delta}",
    )
    t3 &= check(
        "pct_change is None (no pct_change for ranking query)",
        resp.pct_change is None,
        detail=f"pct_change={resp.pct_change}",
    )
    results["BL3"] = t3
    print(f"  {'[PASS]' if t3 else '[FAIL]'} Overall")

    # ── TEST 4: Region comparison — side-by-side ─────────────────────────────
    print("\n" + "-" * 60)
    print("  [BL4] Region Comparison — Side-by-Side North vs South")
    resp = run_test(orch, "bl_4", "Compare North and South sales.")

    t4 = True
    t4 &= check("Returns SynthesizedResponse", resp is not None)
    t4 &= check("answer_text not empty", bool(resp.answer_text))
    t4 &= check(
        "Answer mentions both North and South",
        "north" in resp.answer_text.lower() and "south" in resp.answer_text.lower(),
        detail=resp.answer_text[:100],
    )
    t4 &= check(
        "Answer mentions revenue figures (Rs sign or numeric)",
        any(c.isdigit() for c in resp.answer_text),
        detail=resp.answer_text[:80],
    )
    t4 &= check(
        "delta is None (no generic delta for region comparison)",
        resp.delta is None,
        detail=f"delta={resp.delta}",
    )
    t4 &= check("Table returned (list)", isinstance(resp.table, list))
    results["BL4"] = t4
    print(f"  {'[PASS]' if t4 else '[FAIL]'} Overall")

    # ── TEST 5: Inventory West — Complete Coverage ────────────────────────────
    print("\n" + "-" * 60)
    print("  [BL5] Inventory West — Coverage Should Be Complete")
    resp = run_test(orch, "bl_5", "Did inventory reduce in West region?")

    t5 = True
    t5 &= check("Returns SynthesizedResponse", resp is not None)
    t5 &= check("answer_text not empty", bool(resp.answer_text))
    t5 &= check(
        "Coverage is complete (not missing N/S/E when only West was requested)",
        not resp.coverage_flag.is_partial or resp.coverage_flag.missing_regions == [],
        detail=f"missing_regions={resp.coverage_flag.missing_regions}",
    )
    t5 &= check(
        "Missing regions does not include North, South, East",
        not any(r in resp.coverage_flag.missing_regions for r in ["North", "South", "East"]),
        detail=f"missing_regions={resp.coverage_flag.missing_regions}",
    )
    results["BL5"] = t5
    print(f"  {'[PASS]' if t5 else '[FAIL]'} Overall")

    # ── TEST 6: Unknown promotion — graceful message ──────────────────────────
    print("\n" + "-" * 60)
    print("  [BL6] Unknown Promotion — Graceful No-Data Message")
    resp = run_test(orch, "bl_6", "How did unknown promotion PROMO_XYZ_UNKNOWN affect sales?")

    t6 = True
    t6 &= check("Returns SynthesizedResponse", resp is not None)
    t6 &= check("answer_text not empty", bool(resp.answer_text))
    t6 &= check(
        "No crash — response is graceful",
        "error" not in resp.answer_text.lower() or "matching" in resp.answer_text.lower(),
    )
    t6 &= check(
        "Does NOT say '0 weeks of available data'",
        "0 weeks of available data" not in resp.explanation,
        detail=resp.explanation[:80],
    )
    results["BL6"] = t6
    print(f"  {'[PASS]' if t6 else '[FAIL]'} Overall")

    # ── Summary ───────────────────────────────────────────────────────────────
    passed = sum(1 for v in results.values() if v)
    total = len(results)

    print("\n" + "=" * 70)
    print("BUSINESS LOGIC TEST SUMMARY")
    print("=" * 70)
    for key, val in results.items():
        status = "[PASS]" if val else "[FAIL]"
        print(f"  {status}  {key}")
    print(f"\n  Overall:  {passed}/{total} tests passed")
    if passed == total:
        print("  [ALL PASSED]")
    else:
        print(f"  [{total - passed} FAILED]")
    print("=" * 70)


if __name__ == "__main__":
    import io, sys as _sys
    _sys.stdout = io.TextIOWrapper(_sys.stdout.buffer, encoding="utf-8", errors="replace")
    main()
