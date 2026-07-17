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

import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# Bootstrap project root so agents/ and config.py are importable
# ---------------------------------------------------------------------------
APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from agents.orchestrator import PromotionAnalyticsOrchestrator
from agents.synthesizer import SynthesizedResponse

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
    "Did PROMO_001 improve sales in South region?",
    "Compare North and South sales.",
    "Did inventory reduce in West region?",
    "Which campaign performed best?",
    "Which category generated highest revenue?",
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

def render_response(response: SynthesizedResponse) -> None:
    """Render a SynthesizedResponse in structured cards inside the chat bubble."""

    # 1. Answer text —— prominent card
    st.markdown(
        f"""
        <div class="answer-card">
            <div class="answer-text">💡 {response.answer_text}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

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
        df = pd.DataFrame(response.table)
        st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        st.caption("No tabular data returned.")

    # 4. Explanation
    st.info(f"📝 {response.explanation}")

    # 5. SQL expander
    if response.sql_shown:
        with st.expander("🔍 Show SQL", expanded=False):
            st.code(response.sql_shown, language="sql")


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
        st.session_state.chat_history.append(
            {"role": "assistant", "content": response.model_dump()}
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
                        resp = SynthesizedResponse(**entry["content"])
                        render_response(resp)
                    except Exception:  # noqa: BLE001
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
