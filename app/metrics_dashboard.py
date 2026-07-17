"""
app/metrics_dashboard.py
-------------------------
Phase 11 — AI Analytics Monitoring Dashboard.

Reads logs/query_metrics.csv and displays real KPIs, trend charts,
distribution plots, and a recent-queries table using Streamlit.

Run:
    streamlit run app/metrics_dashboard.py
"""

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# Bootstrap project root
# ---------------------------------------------------------------------------
APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from logs.metrics_logger import MetricsLogger, generate_sample_metrics

# ---------------------------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="AI Analytics Monitoring Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
        .stApp { background-color: #0f1117; }

        .dash-title {
            font-size: 2rem;
            font-weight: 800;
            background: linear-gradient(135deg, #10b981 0%, #06b6d4 50%, #6366f1 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            margin-bottom: 0.15rem;
        }
        .dash-subtitle {
            color: #9ca3af;
            font-size: 0.9rem;
            margin-bottom: 1.5rem;
        }

        /* KPI card containers */
        div[data-testid="metric-container"] {
            background: #1f2937;
            border: 1px solid #374151;
            border-radius: 12px;
            padding: 1rem 1.2rem;
        }

        /* Section headers */
        .section-header {
            color: #e5e7eb;
            font-size: 1.05rem;
            font-weight: 700;
            margin: 1.5rem 0 0.5rem;
            border-left: 4px solid #6366f1;
            padding-left: 10px;
        }

        hr { border-color: #1f2937 !important; }

        .empty-state {
            text-align: center;
            padding: 3rem 2rem;
            background: #111827;
            border: 1px dashed #374151;
            border-radius: 16px;
            color: #9ca3af;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------

@st.cache_data(ttl=10)   # refresh every 10 s so live queries appear quickly
def load_data() -> pd.DataFrame:
    logger = MetricsLogger()
    df = logger.load_metrics()
    if not df.empty:
        # Ensure timestamp is datetime for charts
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
        df = df.sort_values("timestamp", ascending=True).reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# KPI Calculations
# ---------------------------------------------------------------------------

def compute_kpis(df: pd.DataFrame) -> dict:
    total = len(df)
    validation_rate = df["validation_passed"].sum() / total * 100 if total else 0.0
    avg_latency = df["execution_latency_ms"].mean() if total else 0.0
    avg_confidence = df["classification_confidence"].mean() if total else 0.0
    cache_rate = df["cache_hit"].sum() / total * 100 if total else 0.0
    response_rate = df["response_generated"].sum() / total * 100 if total else 0.0
    avg_retries = df["retry_count"].mean() if total else 0.0
    return {
        "total": total,
        "sql_accuracy": response_rate,      # response_generated ≈ SQL generation accuracy
        "validation_rate": validation_rate,
        "avg_latency": avg_latency,
        "avg_confidence": avg_confidence,
        "cache_rate": cache_rate,
        "avg_retries": avg_retries,
    }


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

def render_sidebar(df: pd.DataFrame) -> None:
    with st.sidebar:
        st.markdown("## 📈 Dashboard Controls")
        st.divider()

        # Refresh button
        if st.button("🔄 Refresh Metrics", use_container_width=True):
            st.cache_data.clear()
            st.rerun()

        st.divider()
        st.markdown("### 🧪 Development Tools")
        if st.button("➕ Generate Sample Data (15 rows)", use_container_width=True):
            with st.spinner("Generating sample metrics..."):
                generate_sample_metrics(15)
            st.cache_data.clear()
            st.success("✅ Sample data added.")
            st.rerun()

        st.divider()
        st.markdown("### 📊 Dataset Info")
        if not df.empty:
            st.metric("Total Records", len(df))
            st.caption(
                f"**Earliest:** {df['timestamp'].min().strftime('%Y-%m-%d %H:%M')}\n\n"
                f"**Latest:** {df['timestamp'].max().strftime('%Y-%m-%d %H:%M')}"
            )
        else:
            st.caption("No data yet.")

        st.divider()
        st.markdown("### 🔗 Quick Links")
        st.caption("Run the main app:\n`streamlit run app/streamlit_app.py`")


# ---------------------------------------------------------------------------
# Empty State
# ---------------------------------------------------------------------------

def render_empty_state() -> None:
    st.markdown(
        """
        <div class="empty-state">
            <div style="font-size:3rem">📭</div>
            <h3 style="color:#e5e7eb; margin:1rem 0 0.4rem">No metrics collected yet</h3>
            <p>Run some questions through the main Streamlit app, or click<br>
            <strong>Generate Sample Data</strong> in the sidebar to populate test records.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Main Dashboard
# ---------------------------------------------------------------------------

def main() -> None:
    # Header
    st.markdown(
        '<div class="dash-title">📈 AI Analytics Monitoring Dashboard</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="dash-subtitle">Real-time metrics from the Promotion Analytics pipeline</div>',
        unsafe_allow_html=True,
    )
    st.divider()

    df = load_data()
    render_sidebar(df)

    if df.empty:
        render_empty_state()
        return

    kpis = compute_kpis(df)

    # ── KPI Cards ────────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">🎯 Key Performance Indicators</div>', unsafe_allow_html=True)
    k1, k2, k3, k4, k5 = st.columns(5)

    with k1:
        st.metric(
            "SQL Generation Accuracy",
            f"{kpis['sql_accuracy']:.1f}%",
            help="% of queries where a valid response was generated (response_generated=True).",
        )
    with k2:
        st.metric(
            "Validation Pass Rate",
            f"{kpis['validation_rate']:.1f}%",
            help="% of queries where SQL passed all validation checks.",
        )
    with k3:
        st.metric(
            "Avg Latency",
            f"{kpis['avg_latency']:.1f} ms",
            help="Mean DuckDB execution time across all queries.",
        )
    with k4:
        st.metric(
            "Avg Confidence",
            f"{kpis['avg_confidence']:.3f}",
            help="Mean intent classification confidence score.",
        )
    with k5:
        st.metric(
            "Total Queries",
            kpis["total"],
            help="Total number of pipeline executions logged.",
        )

    # Row 2 KPIs
    k6, k7, _, _, _ = st.columns(5)
    with k6:
        st.metric(
            "Cache Hit Rate",
            f"{kpis['cache_rate']:.1f}%",
            help="% of queries served from the rollup cache.",
        )
    with k7:
        st.metric(
            "Avg Retries",
            f"{kpis['avg_retries']:.2f}",
            help="Mean SQL regeneration retries per query.",
        )

    st.divider()

    # ── Chart 1: Response Latency Over Time ──────────────────────────────────
    st.markdown('<div class="section-header">📉 Response Latency Over Time</div>', unsafe_allow_html=True)
    latency_df = df[["timestamp", "execution_latency_ms"]].copy()
    latency_df = latency_df[latency_df["execution_latency_ms"] > 0]
    if not latency_df.empty:
        latency_df = latency_df.set_index("timestamp")
        st.line_chart(latency_df, color="#6366f1", use_container_width=True)
    else:
        st.caption("No latency data available (filtered queries had 0 ms latency).")

    st.divider()

    # ── Chart 2 + Chart 3 side-by-side ───────────────────────────────────────
    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown('<div class="section-header">✅ Validation Results</div>', unsafe_allow_html=True)
        val_counts = df["validation_passed"].map({True: "Pass", False: "Fail"}).value_counts().reset_index()
        val_counts.columns = ["Result", "Count"]
        st.bar_chart(val_counts.set_index("Result"), color="#10b981", use_container_width=True)

    with col_right:
        st.markdown('<div class="section-header">📊 Classification Confidence Distribution</div>', unsafe_allow_html=True)
        # Build histogram buckets manually for st.bar_chart compatibility
        hist_df = pd.cut(
            df["classification_confidence"],
            bins=[0.0, 0.7, 0.75, 0.80, 0.85, 0.90, 0.95, 1.01],
            labels=["<0.70", "0.70-0.75", "0.75-0.80", "0.80-0.85", "0.85-0.90", "0.90-0.95", "0.95-1.00"],
            right=False,
        ).value_counts().sort_index().reset_index()
        hist_df.columns = ["Confidence Range", "Count"]
        st.bar_chart(hist_df.set_index("Confidence Range"), color="#a855f7", use_container_width=True)

    st.divider()

    # ── Chart 4 + Chart 5 side-by-side ───────────────────────────────────────
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown('<div class="section-header">🔁 Retry Counts</div>', unsafe_allow_html=True)
        retry_counts = df["retry_count"].value_counts().sort_index().reset_index()
        retry_counts.columns = ["Retries", "Frequency"]
        retry_counts["Retries"] = retry_counts["Retries"].astype(str)
        st.bar_chart(retry_counts.set_index("Retries"), color="#f59e0b", use_container_width=True)

    with col_b:
        st.markdown('<div class="section-header">📋 Row Count Distribution</div>', unsafe_allow_html=True)
        row_df = df[df["row_count"] > 0][["timestamp", "row_count"]].set_index("timestamp")
        if not row_df.empty:
            st.bar_chart(row_df, color="#06b6d4", use_container_width=True)
        else:
            st.caption("No row count data for successful queries yet.")

    st.divider()

    # ── Topic Breakdown ───────────────────────────────────────────────────────
    st.markdown('<div class="section-header">🗂 Query Volume by Topic</div>', unsafe_allow_html=True)
    topic_counts = df["topic"].value_counts().reset_index()
    topic_counts.columns = ["Topic", "Count"]
    st.bar_chart(topic_counts.set_index("Topic"), use_container_width=True)

    st.divider()

    # ── Recent Queries Table ──────────────────────────────────────────────────
    st.markdown('<div class="section-header">🕐 Recent Queries (Last 20)</div>', unsafe_allow_html=True)
    recent = df.tail(20).sort_values("timestamp", ascending=False).copy()
    display_cols = [
        "timestamp", "question", "topic",
        "classification_confidence", "execution_latency_ms", "validation_passed",
        "retry_count", "row_count", "cache_hit",
    ]
    display_cols = [c for c in display_cols if c in recent.columns]
    rename_map = {
        "timestamp": "Time",
        "question": "Question",
        "topic": "Topic",
        "classification_confidence": "Confidence",
        "execution_latency_ms": "Latency (ms)",
        "validation_passed": "Valid",
        "retry_count": "Retries",
        "row_count": "Rows",
        "cache_hit": "Cache",
    }
    recent_display = recent[display_cols].rename(columns=rename_map)
    # Truncate long questions for table display
    if "Question" in recent_display.columns:
        recent_display["Question"] = recent_display["Question"].str[:60] + "…"
    st.dataframe(recent_display, use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
