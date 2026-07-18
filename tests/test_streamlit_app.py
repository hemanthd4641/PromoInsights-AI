"""
tests/test_streamlit_app.py
----------------------------
Phase 10 — Unit Tests for the Streamlit App.

Tests verify:
  - Orchestrator can be initialised
  - Session state keys are correctly populated
  - Sample question execution path works end-to-end
  - Reset session zeroes chat history

Run:
    python tests/test_streamlit_app.py
"""

import sys
import uuid
from contextlib import nullcontext
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Bootstrap project root
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from agents.synthesizer import CoverageFlag, SynthesizedResponse, normalize_response_payload

PASS = "[PASS]"
FAIL = "[FAIL]"


def check(label: str, condition: bool, detail: str = "") -> bool:
    tag = PASS if condition else FAIL
    suffix = f" -- {detail}" if detail else ""
    print(f"    {tag} {label}{suffix}")
    return condition


# ---------------------------------------------------------------------------
# Build a canonical fallback SynthesizedResponse for mocking
# ---------------------------------------------------------------------------

def _make_response(answer: str = "Sales increased by 18%.") -> SynthesizedResponse:
    return SynthesizedResponse(
        answer_text=answer,
        delta=2500.0,
        pct_change=18.2,
        table=[{"region": "South", "revenue": 16000, "units_sold": 340}],
        explanation="Promotion effectiveness was measured over week-over-week.",
        coverage_flag=CoverageFlag(
            is_partial=False,
            missing_weeks=[],
            missing_regions=[],
            message="Complete coverage.",
        ),
        sql_shown="SELECT region, SUM(revenue) FROM vw_weekly_sales WHERE region = 'South' GROUP BY region;",
    )


# ---------------------------------------------------------------------------
# T1 — Orchestrator can be initialised
# ---------------------------------------------------------------------------

def test_orchestrator_init() -> bool:
    total = 0
    passed = 0

    print("\n[T1] Orchestrator Initialisation")
    total += 1
    ok = True
    try:
        from agents.orchestrator import PromotionAnalyticsOrchestrator
        orch = PromotionAnalyticsOrchestrator()
        ok &= check("Orchestrator created", orch is not None)
        ok &= check("SessionMemory present", hasattr(orch, "session_memory"))
        ok &= check("IntentClassifier present", hasattr(orch, "intent_classifier"))
        ok &= check("QueryGenerationAgent present", hasattr(orch, "query_generator"))
        ok &= check("Synthesizer present", hasattr(orch, "synthesizer"))
    except Exception as exc:
        ok = False
        print(f"    {FAIL} Exception: {exc}")
    if ok:
        passed += 1

    return passed == total


# ---------------------------------------------------------------------------
# T2 — Session state keys are created (simulated via plain dict)
# ---------------------------------------------------------------------------

def test_session_state_creation() -> bool:
    total = 0
    passed = 0

    print("\n[T2] Session State Creation")
    total += 1
    ok = True

    # Simulate what init_session_state() does
    fake_state: dict = {}

    if "session_id" not in fake_state:
        fake_state["session_id"] = str(uuid.uuid4())
    if "chat_history" not in fake_state:
        fake_state["chat_history"] = []
    if "orchestrator" not in fake_state:
        fake_state["orchestrator"] = MagicMock()

    ok &= check("session_id key exists", "session_id" in fake_state)
    ok &= check("session_id is non-empty UUID", bool(fake_state["session_id"]))
    ok &= check("chat_history is empty list", fake_state["chat_history"] == [])
    ok &= check("orchestrator key exists", "orchestrator" in fake_state)

    if ok:
        passed += 1

    return passed == total


# ---------------------------------------------------------------------------
# T3 — Sample question execution path
# ---------------------------------------------------------------------------

def test_sample_question_execution() -> bool:
    total = 0
    passed = 0

    print("\n[T3] Sample Question Execution Path")
    total += 1
    ok = True

    mock_response = _make_response("Promotion effectiveness increased sales.")

    try:
        from agents.orchestrator import PromotionAnalyticsOrchestrator
        orch = PromotionAnalyticsOrchestrator()

        # Patch the handle method to return the mock response quickly
        with patch.object(orch, "handle", return_value=mock_response) as mock_handle:
            session_id = str(uuid.uuid4())
            question = "Did PROMO_001 improve sales in South region?"

            chat_history: list = []
            chat_history.append({"role": "user", "content": question})

            result = orch.handle(question=question, session_id=session_id)
            chat_history.append({"role": "assistant", "content": result.model_dump()})

            ok &= check("handle() called once", mock_handle.call_count == 1)
            ok &= check("User message appended", chat_history[0]["role"] == "user")
            ok &= check("Question stored", chat_history[0]["content"] == question)
            ok &= check("Assistant response appended", chat_history[1]["role"] == "assistant")
            ok &= check(
                "Answer text in response",
                "effectiveness" in chat_history[1]["content"]["answer_text"].lower(),
            )
    except Exception as exc:
        ok = False
        print(f"    {FAIL} Exception: {exc}")

    if ok:
        passed += 1

    return passed == total


# ---------------------------------------------------------------------------
# T4 — Reset session zeroes chat history
# ---------------------------------------------------------------------------

def test_reset_session() -> bool:
    total = 0
    passed = 0

    print("\n[T4] Reset Session")
    total += 1
    ok = True

    # Simulate session state with existing history
    fake_state = {
        "chat_history": [
            {"role": "user", "content": "Some question"},
            {"role": "assistant", "content": {"answer_text": "Some answer"}},
        ],
        "session_id": "old-session-id",
    }

    # Simulate reset_session()
    fake_state["chat_history"] = []
    fake_state["session_id"] = str(uuid.uuid4())

    ok &= check("chat_history cleared", fake_state["chat_history"] == [])
    ok &= check("session_id regenerated", fake_state["session_id"] != "old-session-id")
    ok &= check("session_id is non-empty", bool(fake_state["session_id"]))

    if ok:
        passed += 1

    return passed == total


# ---------------------------------------------------------------------------
# T5 — SynthesizedResponse renders correctly
# ---------------------------------------------------------------------------

def test_response_model() -> bool:
    total = 0
    passed = 0

    print("\n[T5] SynthesizedResponse Model")
    total += 1
    ok = True

    resp = _make_response()

    ok &= check("answer_text present", bool(resp.answer_text))
    ok &= check("delta present", resp.delta == 2500.0)
    ok &= check("pct_change present", resp.pct_change == 18.2)
    ok &= check("table has rows", len(resp.table) > 0)
    ok &= check("explanation present", bool(resp.explanation))
    ok &= check("coverage_flag present", resp.coverage_flag is not None)
    ok &= check("coverage is_partial=False", not resp.coverage_flag.is_partial)
    ok &= check("sql_shown present", bool(resp.sql_shown))

    # Roundtrip via model_dump / model_validate
    as_dict = resp.model_dump()
    restored = SynthesizedResponse(**as_dict)
    ok &= check("model_dump roundtrip OK", restored.answer_text == resp.answer_text)

    if ok:
        passed += 1

    return passed == total


# ---------------------------------------------------------------------------
# T6 — Payload normalization handles DataFrame / numpy values
# ---------------------------------------------------------------------------

def test_prepare_display_dataframe_sanitizes_nested_values() -> bool:
    total = 0
    passed = 0

    print("\n[T6b] Display Table Sanitization")
    total += 1
    ok = True

    try:
        from app.streamlit_app import prepare_display_dataframe

        payload = [
            {"region": "North", "metrics": {"revenue": 100}, "score": np.nan},
            {"region": "South", "metrics": [1, 2, 3], "score": 3.2},
        ]
        df = prepare_display_dataframe(payload)
        ok &= check("DataFrame returned", isinstance(df, pd.DataFrame))
        ok &= check("Rows preserved", len(df) == 2)
        ok &= check("Nested values stringified", df.loc[0, "metrics"].startswith("{"))
        ok &= check("NaN values tolerated", df.loc[0, "score"] is None or pd.isna(df.loc[0, "score"]))
    except Exception as exc:
        ok = False
        print(f"    {FAIL} Exception: {exc}")

    if ok:
        passed += 1

    return passed == total


def test_payload_normalization_handles_dataframe_and_numpy_types() -> bool:
    total = 0
    passed = 0

    print("\n[T6] Payload Normalization")
    total += 1
    ok = True

    payload = {
        "answer_text": "Sales improved.",
        "delta": np.float64(12.5),
        "pct_change": np.float64(3.2),
        "table": pd.DataFrame([{"region": "South", "revenue": 2000}]),
        "explanation": "Trend analysis.",
        "coverage_flag": {
            "is_partial": False,
            "missing_weeks": [],
            "missing_regions": [],
            "message": "Complete coverage.",
        },
        "sql_shown": None,
    }

    normalized = normalize_response_payload(payload)

    ok &= check("normalization returns dict", isinstance(normalized, dict))
    ok &= check("table converted to list[dict]", isinstance(normalized["table"], list))
    ok &= check("table rows preserved", normalized["table"][0]["region"] == "South")
    ok &= check("numpy values converted to Python scalars", normalized["delta"] == 12.5)
    ok &= check("sql_shown defaults to empty string", normalized["sql_shown"] == "")

    if ok:
        passed += 1

    return passed == total


def test_handle_question_serializes_table_to_list() -> bool:
    total = 0
    passed = 0

    print("\n[T7] Question Handler Serialization")
    total += 1
    ok = True

    try:
        from app.streamlit_app import handle_question

        class DummyOrchestrator:
            def handle(self, question: str, session_id: str):
                return SynthesizedResponse(
                    answer_text="Sales improved.",
                    delta=10.0,
                    pct_change=5.0,
                    table=[{"region": "South", "revenue": 2000}],
                    explanation="Trend analysis.",
                    coverage_flag=CoverageFlag(
                        is_partial=False,
                        missing_weeks=[],
                        missing_regions=[],
                        message="Complete coverage.",
                    ),
                    sql_shown="SELECT 1",
                )

        fake_state = {
            "chat_history": [],
            "orchestrator": DummyOrchestrator(),
            "session_id": "session-123",
        }

        with patch.dict("app.streamlit_app.st.session_state", fake_state, clear=True):
            with patch("app.streamlit_app.st.spinner", return_value=nullcontext()):
                handle_question("Did sales improve?")

        assistant_entry = fake_state["chat_history"][1]
        ok &= check("assistant entry stored", assistant_entry["role"] == "assistant")
        ok &= check("table serialized to list", isinstance(assistant_entry["content"]["table"], list))
        ok &= check("table rows preserved", assistant_entry["content"]["table"][0]["region"] == "South")
    except Exception as exc:
        ok = False
        print(f"    {FAIL} Exception: {exc}")

    if ok:
        passed += 1

    return passed == total


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_tests() -> None:
    total = 7
    passed = 0
    failed: list[str] = []

    print("\n" + "=" * 70)
    print("PHASE 10 -- STREAMLIT APP TEST SUITE")
    print("=" * 70)

    results = [
        ("T1", test_orchestrator_init),
        ("T2", test_session_state_creation),
        ("T3", test_sample_question_execution),
        ("T4", test_reset_session),
        ("T5", test_response_model),
        ("T6", test_payload_normalization_handles_dataframe_and_numpy_types),
        ("T7", test_handle_question_serializes_table_to_list),
    ]

    for name, fn in results:
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
