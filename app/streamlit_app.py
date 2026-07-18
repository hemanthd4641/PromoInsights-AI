"""
app/streamlit_app.py
---------------------
Phase 10 — Streamlit UI for the Promotion Analytics AI Assistant.

Provides a ChatGPT-style interface for business users to ask natural-language
questions and receive structured, business-friendly answers powered by the
Phase 9 orchestrator.

Run:
    streamlit run app/streamlit_app.py
"""

import sys
import uuid
from pathlib import Path

import json
import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# Bootstrap project root so agents/ and config.py are importable
# ---------------------------------------------------------------------------
APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from agents.orchestrator import PromotionAnalyticsOrchestrator
from agents.synthesizer import SynthesizedResponse, coerce_to_synthesized_response
import importlib
import config as app_config

DEBUG_MODE = getattr(app_config, "DEBUG_MODE", True)
if not hasattr(app_config, "DEBUG_MODE"):
    app_config = importlib.reload(app_config)
    DEBUG_MODE = getattr(app_config, "DEBUG_MODE", True)

# ---------------------------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Promotion Analytics AI Assistant",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS — premium dark-mode aesthetic
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
        /* ── Global background ── */
        .stApp { background-color: #0f1117; }

        /* ── Title block ── */
        .main-title {
            font-size: 2.2rem;
            font-weight: 800;
            background: linear-gradient(135deg, #6366f1 0%, #a855f7 50%, #ec4899 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            margin-bottom: 0.1rem;
        }
        .main-subtitle {
            color: #9ca3af;
            font-size: 0.95rem;
            margin-bottom: 1.5rem;
        }

        /* ── Answer card ── */
        .answer-card {
            background: linear-gradient(135deg, #1e1b4b 0%, #1a1a2e 100%);
            border: 1px solid #4338ca;
            border-radius: 12px;
            padding: 1.4rem 1.6rem;
            margin-bottom: 1rem;
            box-shadow: 0 4px 24px rgba(99, 102, 241, 0.15);
        }
        .answer-text {
            color: #e0e7ff;
            font-size: 1.15rem;
            font-weight: 600;
            line-height: 1.6;
        }

        /* ── Coverage badge ── */
        .badge-complete {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            background: #14532d;
            color: #86efac;
            border: 1px solid #22c55e;
            border-radius: 20px;
            padding: 4px 12px;
            font-size: 0.8rem;
            font-weight: 600;
        }
        .badge-partial {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            background: #451a03;
            color: #fcd34d;
            border: 1px solid #f59e0b;
            border-radius: 20px;
            padding: 4px 12px;
            font-size: 0.8rem;
            font-weight: 600;
        }

        /* ── Chat messages ── */
        .stChatMessage {
            border-radius: 12px !important;
            margin-bottom: 0.5rem !important;
        }

        /* ── Sidebar ── */
        .css-1d391kg, [data-testid="stSidebar"] {
            background-color: #111827 !important;
        }

        /* ── Sample question buttons ── */
        div[data-testid="stButton"] > button {
            background: #1f2937;
            color: #d1d5db;
            border: 1px solid #374151;
            border-radius: 8px;
            text-align: left;
            width: 100%;
            padding: 0.5rem 0.75rem;
            font-size: 0.82rem;
            transition: all 0.2s ease;
        }
        div[data-testid="stButton"] > button:hover {
            background: #374151;
            border-color: #6366f1;
            color: #e0e7ff;
        }

        /* ── Divider ── */
        hr { border-color: #1f2937 !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Sample questions shown in sidebar and empty state
# ---------------------------------------------------------------------------
SAMPLE_QUESTIONS = [
    "Which campaign performed best?",
    "Which category generated highest revenue?",
    "Did PROMO_001 improve sales in South region?",
]

# ---------------------------------------------------------------------------
# Session State Initialisation
# ---------------------------------------------------------------------------

def init_session_state() -> None:
    """Initialise all Streamlit session-state keys on first load."""
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []          # list[dict{role, content}]

    if "orchestrator" not in st.session_state:
        with st.spinner("Loading AI engine..."):
            st.session_state.orchestrator = PromotionAnalyticsOrchestrator()


def reset_session() -> None:
    """Clear chat history and generate a fresh session ID."""
    st.session_state.chat_history = []
    st.session_state.session_id = str(uuid.uuid4())
    # Clear the orchestrator's internal session memory for the old session
    if "orchestrator" in st.session_state:
        orch: PromotionAnalyticsOrchestrator = st.session_state.orchestrator
        orch.session_memory.clear_session(st.session_state.session_id)
    st.success("✅ Session reset successfully.")


# ---------------------------------------------------------------------------
# Response Renderer
# ---------------------------------------------------------------------------

def prepare_display_dataframe(table_payload: object) -> pd.DataFrame:
    """Convert arbitrary table payloads into a DataFrame that Streamlit can display safely."""
    if isinstance(table_payload, pd.DataFrame):
        frame = table_payload.copy()
    elif isinstance(table_payload, list):
        frame = pd.DataFrame(table_payload)
    else:
        frame = pd.DataFrame()

    try:
        if isinstance(frame, pd.DataFrame):
            normalized = json.loads(frame.fillna("").astype(str).to_json(orient="records"))
            assert isinstance(normalized, list)
            frame = pd.DataFrame(normalized)
    except Exception:
        frame = pd.DataFrame()

    if frame.empty:
        return pd.DataFrame(columns=["message"])

    def _sanitize(value: object) -> object:
        if value is None or value is pd.NA:
            return None
        if isinstance(value, (pd.Timestamp,)):
            return value.to_pydatetime()
        if isinstance(value, (dict, list, tuple, set)):
            try:
                return json.dumps(value, default=str)
            except Exception:
                return str(value)
        if isinstance(value, float) and pd.isna(value):
            return None
        if hasattr(value, "tolist"):
            try:
                return _sanitize(value.tolist())
            except Exception:
                return str(value)
        return value

    for column in frame.columns:
        frame[column] = frame[column].map(_sanitize)

    return frame


def _get_debug_info(response: object) -> dict:
    """Return a safe debug-info payload for both current and legacy response objects."""
    try:
        raw_debug = getattr(response, "debug_info", None)
        if isinstance(raw_debug, dict):
            return raw_debug
    except Exception:
        raw_debug = None

    try:
        payload = response.model_dump()
        if isinstance(payload, dict):
            raw_debug = payload.get("debug_info", {})
            if isinstance(raw_debug, dict):
                return raw_debug
    except Exception:
        pass

    return {}


def render_response(response: SynthesizedResponse) -> None:
    """Render a SynthesizedResponse in a chat-friendly layout based on its response type."""

    # 1. Answer text —— prominent card
    st.markdown(
        f"""
        <div class="answer-card">
            <div class="answer-text">💡 {response.answer_text}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    response_type = getattr(response, "response_type", "analytics") or "analytics"

    if response_type in {"chat", "help", "follow_up"}:
        if response_type == "error":
            st.info(f"🛠️ {response.explanation}")
        if response_type == "follow_up" and response.suggestions:
            st.markdown("**💡 Suggested follow-ups**")
            for suggestion in response.suggestions:
                st.markdown(f"- {suggestion}")
        return

    if response_type == "error":
        st.error("⚠ I hit a snag while preparing that answer, but I can help you try again.")
        st.info(f"🛠️ {response.explanation}")
        return

    # 2. Metric cards — delta & pct_change
    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        st.metric(
            label="📈 Delta",
            value=f"{response.delta:+,.2f}" if response.delta is not None else "—",
        )
    with col2:
        st.metric(
            label="% Change",
            value=(
                f"{response.pct_change:+.2f}%"
                if response.pct_change is not None
                else "—"
            ),
        )
    with col3:
        # 6. Coverage badge
        flag = response.coverage_flag
        if flag.is_partial:
            st.markdown(
                '<span class="badge-partial">⚠ Partial Coverage</span>',
                unsafe_allow_html=True,
            )
            details = []
            if flag.missing_regions:
                details.append(f"Missing regions: {', '.join(flag.missing_regions)}")
            if flag.missing_weeks:
                wk_count = len(flag.missing_weeks)
                details.append(f"Missing {wk_count} week(s)")
            if details:
                st.caption("  •  ".join(details))
        else:
            st.markdown(
                '<span class="badge-complete">✓ Complete Coverage</span>',
                unsafe_allow_html=True,
            )

    # 3. Supporting table
    if response.table:
        st.markdown("**📋 Supporting Data**")
        table_payload = response.table
        try:
            if isinstance(table_payload, list) and table_payload:
                preview_rows = []
                for row in table_payload[:5]:
                    if isinstance(row, dict):
                        preview_rows.append({str(k): str(v) for k, v in row.items() if not isinstance(v, (dict, list, tuple, set))})
                    else:
                        preview_rows.append({"value": str(row)})
                if preview_rows:
                    st.text("Preview: " + json.dumps(preview_rows, default=str))
                else:
                    st.caption("No tabular data returned.")
            else:
                st.caption("No tabular data returned.")
        except Exception as exc:
            print("[UI DEBUG] render failure=", exc)
            st.caption("Unable to render supporting data safely.")
    else:
        st.caption("No tabular data returned.")

    # 4. Explanation
    st.info(f"📝 {response.explanation}")

    # 5. Suggested follow-ups
    if response.suggestions:
        st.markdown("**💡 Suggested follow-ups**")
        for suggestion in response.suggestions:
            st.markdown(f"- {suggestion}")

    # 6. SQL expander
    if response.sql_shown:
        with st.expander("🔍 Show SQL", expanded=False):
            st.code(response.sql_shown, language="sql")

    debug_info = _get_debug_info(response)
    if DEBUG_MODE and debug_info:
        with st.expander("🧪 Debug Trace", expanded=False):
            st.json({
                "route": debug_info.get("route"),
                "intent": debug_info.get("intent"),
                "grounded_metric": debug_info.get("grounded_metric"),
                "generated_sql": debug_info.get("generated_sql"),
                "dataframe_head": debug_info.get("dataframe_head"),
                "template_used": debug_info.get("template_used"),
                "response_object": debug_info.get("response_object"),
            })


# ---------------------------------------------------------------------------
# Core question handler
# ---------------------------------------------------------------------------

def handle_question(question: str) -> None:
    """Run the orchestrator and append results to chat history."""
    # Append user message
    st.session_state.chat_history.append({"role": "user", "content": question})

    orch: PromotionAnalyticsOrchestrator = st.session_state.orchestrator

    try:
        with st.spinner("🔍 Analyzing your request..."):
            response: SynthesizedResponse = orch.handle(
                question=question,
                session_id=st.session_state.session_id,
            )
        # Store the full response object (serialised as dict) for re-rendering
        response_payload = response.model_dump()
        if "table" in response_payload:
            table_payload = response_payload["table"]
            if isinstance(table_payload, list):
                table_df = pd.DataFrame(table_payload)
            else:
                table_df = pd.DataFrame()
            table_data = json.loads(table_df.fillna("").astype(str).to_json(orient="records"))
            assert isinstance(table_data, list)
            response_payload["table"] = table_data
        st.session_state.chat_history.append(
            {"role": "assistant", "content": response_payload}
        )
    except Exception as exc:  # noqa: BLE001
        st.error(f"⚠ Unable to process your request. Please try again.\n\n`{exc}`")
        st.session_state.chat_history.append(
            {
                "role": "assistant",
                "content": {
                    "answer_text": "Unable to process your request.",
                    "delta": None,
                    "pct_change": None,
                    "table": [],
                    "explanation": str(exc),
                    "coverage_flag": {
                        "is_partial": True,
                        "missing_weeks": [],
                        "missing_regions": [],
                        "message": "Pipeline error.",
                    },
                    "sql_shown": "",
                },
            }
        )


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

def render_sidebar() -> str | None:
    """Render sidebar and return a sample question if a button was clicked."""
    clicked_question: str | None = None

    with st.sidebar:
        st.markdown("## ⚡ Quick Actions")
        st.markdown("---")

        st.markdown("### 💬 Sample Questions")
        for q in SAMPLE_QUESTIONS:
            if st.button(q, key=f"sample_{hash(q)}"):
                clicked_question = q

        st.markdown("---")
        st.markdown("### 🔄 Session")
        if st.button("🗑 Reset Session", use_container_width=True):
            reset_session()



        st.markdown("---")
        st.markdown("### ℹ About")
        st.caption(
            "Promotion Analytics AI Assistant uses an agentic pipeline to "
            "answer business questions with SQL-backed results.\n\n"
            "**Phases:** Intent → Ground → Generate SQL → Validate → Execute → Synthesize"
        )

    return clicked_question


# ---------------------------------------------------------------------------
# Empty State
# ---------------------------------------------------------------------------

def render_empty_state() -> None:
    """Show onboarding card when no chat history exists."""
    st.markdown(
        """
        <div style="
            text-align: center;
            padding: 3rem 2rem;
            background: linear-gradient(135deg, #1e1b4b20 0%, #0f172a 100%);
            border: 1px dashed #374151;
            border-radius: 16px;
            margin-top: 2rem;
        ">
            <div style="font-size: 3rem; margin-bottom: 1rem;">📊</div>
            <h3 style="color: #e0e7ff; margin-bottom: 0.5rem;">
                Ask your first question
            </h3>
            <p style="color: #9ca3af; max-width: 480px; margin: 0 auto 1.5rem;">
                Type a business question below or click a <strong>Sample Question</strong>
                in the sidebar. The AI assistant will generate SQL, execute it, and
                return a business-friendly answer.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Show example pills inline
    st.markdown("")
    cols = st.columns(len(SAMPLE_QUESTIONS))
    for i, (col, q) in enumerate(zip(cols, SAMPLE_QUESTIONS)):
        with col:
            if st.button(q[:50] + ("…" if len(q) > 50 else ""), key=f"empty_{i}"):
                handle_question(q)
                st.rerun()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    init_session_state()

    # ── Header ──────────────────────────────────────────────────────────────
    st.markdown(
        '<div class="main-title">📊 Promotion Analytics AI Assistant</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="main-subtitle">'
        "Ask questions about <strong>Promotion Performance</strong>, "
        "<strong>Inventory Movement</strong>, "
        "<strong>Regional Comparisons</strong>, or "
        "<strong>Campaign Impact</strong>."
        "</div>",
        unsafe_allow_html=True,
    )
    st.divider()

    # ── Sidebar (returns a question if a sample button was clicked) ──────────
    sidebar_question = render_sidebar()

    # ── Replay chat history ──────────────────────────────────────────────────
    history = st.session_state.chat_history
    if not history:
        render_empty_state()
    else:
        for entry in history:
            if entry["role"] == "user":
                with st.chat_message("user", avatar="🧑‍💼"):
                    st.markdown(entry["content"])
            else:
                with st.chat_message("assistant", avatar="📊"):
                    try:
                        payload = entry["content"]
                        print(type(payload))
                        print(payload)
                        print(coerce_to_synthesized_response(payload).model_dump())
                        resp = coerce_to_synthesized_response(payload)
                        render_response(resp)
                    except Exception as exc:  # noqa: BLE001
                        fallback = coerce_to_synthesized_response({
                            "answer_text": "I couldn't render this previous response safely.",
                            "delta": None,
                            "pct_change": None,
                            "table": [],
                            "explanation": str(exc),
                            "coverage_flag": {
                                "is_partial": True,
                                "missing_weeks": [],
                                "missing_regions": [],
                                "message": "Fallback response.",
                            },
                            "sql_shown": "",
                        })
                        render_response(fallback)
                        st.error("Could not render previous response.")

    # ── Handle sidebar button click ──────────────────────────────────────────
    if sidebar_question:
        handle_question(sidebar_question)
        st.rerun()

    # ── Chat input ───────────────────────────────────────────────────────────
    if prompt := st.chat_input("Ask a question about your promotions..."):
        handle_question(prompt)
        st.rerun()


if __name__ == "__main__":
    main()
