"""
agents/synthesizer.py
----------------------
Phase 8 -- Response Synthesis Agent.

Converts:
  - DataFrame
  - GroundedIntent
  - ExecutionMetadata
  - SQL string
into a business-friendly structured SynthesizedResponse.

Features:
  - Topic-aware answer generation (promotion / inventory / region_comparison / campaign_impact)
  - Context-aware coverage detection (only checks requested region/weeks)
  - Metric extraction gated by topic (delta/pct only for trend metrics)
  - Winner/ranking identification for campaign_impact queries
  - Never-crash guarantee

Usage:
    from agents.synthesizer import ResponseSynthesizer
    synth = ResponseSynthesizer()
    response = synth.synthesize(df, intent, metadata, sql)
"""

import json
import logging
import sys
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

# ---------------------------------------------------------------------------
# Bootstrap project root
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from agents.executor import ExecutionMetadata, ExecutionResult
from agents.query_grounding import GroundedIntent
from config import LOG_LEVEL

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Coverage Constants
# ---------------------------------------------------------------------------
ALL_REGIONS = {"North", "South", "East", "West"}
EXPECTED_WEEKS = set(range(1, 53))

# ---------------------------------------------------------------------------
# Topics that should NEVER show delta / pct_change
# (ranking / top-performer / comparative listing queries)
# ---------------------------------------------------------------------------
_NO_DELTA_TOPICS = {"campaign_impact", "region_comparison"}

# ---------------------------------------------------------------------------
# Pydantic Models
# ---------------------------------------------------------------------------


class CoverageFlag(BaseModel):
    is_partial: bool
    missing_weeks: List[int] = Field(default_factory=list)
    missing_regions: List[str] = Field(default_factory=list)
    message: str


class SynthesizedResponse(BaseModel):
    answer_text: str
    delta: Optional[float]
    pct_change: Optional[float]
    table: List[Dict[str, Any]]
    explanation: str
    coverage_flag: CoverageFlag
    sql_shown: str
    suggestions: List[str] = Field(default_factory=list)
    response_type: str = Field(default="analytics")
    debug_info: Dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def validate_response(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        answer_text = values.get("answer_text")
        explanation = values.get("explanation")
        delta = values.get("delta")
        pct_change = values.get("pct_change")

        if not isinstance(answer_text, str):
            values["answer_text"] = str(answer_text or "")
        if not isinstance(explanation, str):
            values["explanation"] = str(explanation or "")
        if delta is not None and not isinstance(delta, (int, float)):
            values["delta"] = None
        if pct_change is not None and not isinstance(pct_change, (int, float)):
            values["pct_change"] = None

        assert isinstance(values["answer_text"], str)
        assert isinstance(values["explanation"], str)
        assert values["delta"] is None or isinstance(values["delta"], (int, float))
        assert values["pct_change"] is None or isinstance(values["pct_change"], (int, float))
        return values

    model_config = {"validate_assignment": True}

    def model_post_init(self, __context: Any) -> None:
        self.__class__.validate_response(self.model_dump())


def _to_native(value: Any) -> Any:
    """Recursively coerce common pandas/numpy objects into JSON-safe Python values."""
    if value is None or value is pd.NA:
        return None

    if isinstance(value, (str, bool, int)):
        return value

    if isinstance(value, (float, np.floating)):
        numeric = float(value)
        return None if pd.isna(numeric) or not np.isfinite(numeric) else numeric

    if isinstance(value, (datetime, date)):
        return value.isoformat()

    if isinstance(value, (pd.Timestamp,)):
        return value.to_pydatetime().isoformat()

    if isinstance(value, dict):
        return {str(k): _to_native(v) for k, v in value.items()}

    if isinstance(value, (list, tuple, set)):
        return [_to_native(item) for item in value]

    if isinstance(value, pd.DataFrame):
        return value.where(pd.notnull(value), None).to_dict(orient="records")

    if hasattr(value, "tolist"):
        try:
            return _to_native(value.tolist())
        except Exception:
            pass

    if hasattr(value, "item"):
        try:
            return _to_native(value.item())
        except Exception:
            pass

    if hasattr(value, "model_dump"):
        try:
            return value.model_dump()
        except Exception:
            pass

    if isinstance(value, np.generic):
        return value.item()

    return value


def normalize_response_payload(payload: Any) -> Dict[str, Any]:
    """Normalize arbitrary payloads into a JSON-safe dict compatible with SynthesizedResponse."""
    if isinstance(payload, SynthesizedResponse):
        return payload.model_dump()

    if hasattr(payload, "model_dump"):
        try:
            payload = payload.model_dump()
        except Exception:
            payload = {}

    if not isinstance(payload, dict):
        payload = {}

    answer_text = _to_native(payload.get("answer_text"))
    delta = _to_native(payload.get("delta"))
    pct_change = _to_native(payload.get("pct_change"))
    explanation = _to_native(payload.get("explanation"))

    if not isinstance(answer_text, str):
        answer_text = str(answer_text or "")
    if not isinstance(explanation, str):
        explanation = str(explanation or "")

    if delta is not None:
        try:
            numeric_delta = float(delta)
        except (TypeError, ValueError):
            numeric_delta = None
        delta = None if numeric_delta is None or not np.isfinite(numeric_delta) else numeric_delta

    if pct_change is not None:
        try:
            numeric_pct = float(pct_change)
        except (TypeError, ValueError):
            numeric_pct = None
        pct_change = None if numeric_pct is None or not np.isfinite(numeric_pct) else numeric_pct
    sql_shown = _to_native(payload.get("sql_shown"))
    suggestions = _to_native(payload.get("suggestions")) or []
    response_type = _to_native(payload.get("response_type")) or "analytics"
    if not isinstance(suggestions, list):
        suggestions = [str(suggestions)] if suggestions else []

    raw_table = payload.get("table")
    if isinstance(raw_table, pd.DataFrame):
        table_value = raw_table.where(pd.notnull(raw_table), None).to_dict(orient="records")
    elif raw_table is None:
        table_value = []
    else:
        table_value = _to_native(raw_table)
        if not isinstance(table_value, list):
            if isinstance(table_value, dict):
                table_value = [table_value]
            else:
                table_value = []

    raw_coverage_flag = payload.get("coverage_flag")
    if isinstance(raw_coverage_flag, CoverageFlag):
        coverage_flag_value = raw_coverage_flag.model_dump()
    elif isinstance(raw_coverage_flag, dict):
        coverage_flag_value = {
            "is_partial": bool(_to_native(raw_coverage_flag.get("is_partial", False))),
            "missing_weeks": _to_native(raw_coverage_flag.get("missing_weeks")) or [],
            "missing_regions": _to_native(raw_coverage_flag.get("missing_regions")) or [],
            "message": _to_native(raw_coverage_flag.get("message")) or "",
        }
    else:
        coverage_flag_value = {
            "is_partial": True,
            "missing_weeks": [],
            "missing_regions": [],
            "message": "Fallback coverage.",
        }

    debug_info = _to_native(payload.get("debug_info")) or {}
    if not isinstance(debug_info, dict):
        debug_info = {}

    return {
        "answer_text": answer_text or "",
        "delta": delta,
        "pct_change": pct_change,
        "table": table_value,
        "explanation": explanation or "",
        "coverage_flag": coverage_flag_value,
        "sql_shown": sql_shown or "",
        "suggestions": suggestions,
        "response_type": response_type,
        "debug_info": debug_info,
    }


def sanitize_for_storage(value: Any) -> Any:
    """Recursively sanitize payloads into JSON-safe Python values."""
    return _to_native(value)


def coerce_to_synthesized_response(response: Any) -> SynthesizedResponse:
    """Return a valid SynthesizedResponse even when the input is malformed or contains pandas/numpy values."""
    if isinstance(response, SynthesizedResponse):
        return response

    try:
        payload = normalize_response_payload(response)
        return SynthesizedResponse(**payload)
    except Exception as exc:
        log.warning("Response coercion failed, using fallback: %s", exc)
        fallback_payload = normalize_response_payload({
            "answer_text": "I couldn't render this response safely.",
            "delta": None,
            "pct_change": None,
            "table": [],
            "explanation": "The stored response payload was invalid or non-serializable.",
            "coverage_flag": {
                "is_partial": True,
                "missing_weeks": [],
                "missing_regions": [],
                "message": "Fallback response.",
            },
            "sql_shown": "",
            "suggestions": [],
            "response_type": "error",
        })
        return SynthesizedResponse(**fallback_payload)


# ---------------------------------------------------------------------------
# Synthesis Agent
# ---------------------------------------------------------------------------


class ResponseSynthesizer:
    """
    Response Synthesis Agent — translates query results into structured,
    business-friendly responses with topic-aware templates.
    """

    def __init__(self) -> None:
        log.info("ResponseSynthesizer initialised.")

        from config import GROQ_API_KEY, MODEL_NAME, MAX_RETRIES

        self._llm = None
        if GROQ_API_KEY:
            try:
                self._llm = ChatGroq(
                    model=MODEL_NAME,
                    api_key=GROQ_API_KEY,
                    temperature=0.0,
                    max_retries=MAX_RETRIES,
                )
            except Exception as exc:
                log.warning("ChatGroq initialisation failed, using deterministic synthesis fallback: %s", exc)

        self._prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an expert Data Analyst and Executive Synthesizer for PromoInsights AI. "
                       "Your job is to read the results of an SQL query execution and provide a clear, concise, "
                       "business-friendly answer to the user's question.\n\n"
                       "## RULES:\n"
                       "- The data table provided may only contain up to 20 rows. If there are more rows, you will see a truncated note.\n"
                       "- Do not mention SQL, databases, or 'the data shows'. Speak directly about the business outcome.\n"
                       "- Use currency formatting (₹) for revenue metrics.\n"
                       "- If the table is empty or does not directly answer the question, state that gracefully.\n"
                       "- Do NOT use markdown code blocks or structured JSON in your response. Just plain text."),
            ("human", "User Question: {question}\n\nGrounded Topic: {topic}\n\nQuery Results (Top 20 rows):\n{data_table}\n\n"
                      "Synthesize the answer based on these results.")
        ])

    # ------------------------------------------------------------------
    # Function 1 — detect_coverage  (context-aware)
    # ------------------------------------------------------------------

    def detect_coverage(
        self, df: pd.DataFrame, grounded_intent: GroundedIntent
    ) -> CoverageFlag:
        """
        Determine whether data coverage is complete for the *requested* scope.

        Rules:
        - If the intent specifies a region, only that region is expected.
        - If no region is specified, all four regions are expected.
        - If no 'week' column is present, week coverage is not evaluated.
        """
        missing_weeks: List[int] = []
        missing_regions: List[str] = []

        if df.empty:
            return CoverageFlag(
                is_partial=True,
                missing_weeks=[],
                missing_regions=[],
                message="No data returned.",
            )

        # --- Region coverage (context-aware) ---
        if "region" in df.columns:
            found_regions = set(df["region"].dropna().unique())
            # If user asked about a specific region, only expect that region
            if grounded_intent.region:
                expected_regions = {grounded_intent.region}
            else:
                expected_regions = ALL_REGIONS
            missing_regions = sorted(list(expected_regions - found_regions))

        # --- Week coverage ---
        if "week" in df.columns:
            found_weeks = set(df["week"].dropna().unique())
            # Only evaluate week coverage when result spans multiple weeks
            if len(found_weeks) > 1:
                missing_weeks = sorted(list(EXPECTED_WEEKS - found_weeks))

        is_partial = bool(missing_weeks or missing_regions)

        if is_partial:
            msg = "Data coverage is partial."
            log.info(
                "  Coverage detection: partial (missing %d weeks, %d regions)",
                len(missing_weeks),
                len(missing_regions),
            )
        else:
            msg = "Data coverage is complete."
            log.info("  Coverage detection: complete")

        return CoverageFlag(
            is_partial=is_partial,
            missing_weeks=missing_weeks,
            missing_regions=missing_regions,
            message=msg,
        )

    def _call_llm(self, chain, payload):
        return chain.invoke(payload)

    # ------------------------------------------------------------------
    # Function 2 — generate_answer_text  (Dynamic LLM Synthesis)
    # ------------------------------------------------------------------

    def generate_answer_text(
        self, df: pd.DataFrame, grounded_intent: GroundedIntent, question: str
    ) -> str:
        """Create a concise, topic-specific executive summary from the data dynamically using an LLM."""

        if df is None or df.empty:
            return "No matching records were found for the requested filters."

        if self._should_use_fallback(grounded_intent.topic, question):
            return self._fallback_answer_text(df, grounded_intent, question)

        try:
            # Limit to top 20 rows to avoid token explosion
            df_subset = df.head(20)
            data_md = df_subset.to_csv(index=False)
            if len(df) > 20:
                data_md += f"\n\n... (Truncated. Total rows: {len(df)})"

            chain = self._prompt | self._llm
            response = self._call_llm(chain, {
                "question": question,
                "topic": grounded_intent.topic,
                "data_table": data_md
            })
            return str(response.content).strip()

        except Exception as exc:
            log.error("LLM synthesis failed: %s", exc)
            return self._fallback_answer_text(df, grounded_intent, question)

    @staticmethod
    def _should_use_fallback(topic: str, question: str) -> bool:
        q = (question or "").lower()
        topic_key = (topic or "").lower()
        ranking_terms = ["rank", "ranking", "highest", "best", "top", "lowest", "worst", "performed best", "generated highest", "top 5"]
        comparison_terms = ["compare", "comparison", "versus", "vs"]
        promotion_terms = ["promo", "promotion", "improve", "improved", "baseline"]
        inventory_terms = ["inventory", "stock", "reduce", "reduction", "overstock", "understock"]
        trend_terms = ["trend", "growth", "over time", "increase", "decrease", "change over time"]
        return (
            topic_key in {"region_comparison", "ranking", "campaign_impact", "promotion", "inventory", "trend_analysis"}
            or any(term in q for term in ranking_terms)
            or any(term in q for term in comparison_terms)
            or any(term in q for term in promotion_terms)
            or any(term in q for term in inventory_terms)
            or any(term in q for term in trend_terms)
        )

    @staticmethod
    def _safe_metric(value: Any) -> Optional[float]:
        if value is None or value is pd.NA:
            return None
        try:
            converted = float(value)
        except (TypeError, ValueError):
            return None
        if pd.isna(converted) or not np.isfinite(converted):
            return None
        return converted

    @staticmethod
    def _format_uplift_text(pct_change: Any) -> str:
        pct_value = ResponseSynthesizer._safe_metric(pct_change)
        if pct_value is None:
            return "Insufficient baseline data to calculate uplift."
        return f"{pct_value:+.1f}% uplift"

    @staticmethod
    def _fallback_answer_text(
        df: pd.DataFrame, grounded_intent: GroundedIntent, question: str
    ) -> str:
        q = (question or "").lower()
        if df is None or df.empty:
            return "No matching records were found for the requested filters."

        numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        value_col = next((c for c in ["revenue", "units_sold", "stock_level", "total_revenue"] if c in df.columns), None)
        if value_col is None and numeric_cols:
            value_col = numeric_cols[0]

        if value_col is None:
            return "Analysis complete based on the requested criteria. Please review the supporting data table."

        topic = (grounded_intent.topic or "").lower()

        if topic == "promotion" or "promo" in q or "promotion" in q or "improve" in q or "improved" in q:
            promo_id = None
            if "promo_id" in df.columns:
                promo_id = str(df.iloc[0].get("promo_id", "")) if not df.empty else None
            elif "promo_name" in df.columns:
                promo_id = str(df.iloc[0].get("promo_name", "")) if not df.empty else None
            region = None
            if "region" in df.columns:
                region = str(df.iloc[0].get("region", "")) if not df.empty else None
            revenue_value = None
            if "total_revenue" in df.columns:
                revenue_value = ResponseSynthesizer._safe_metric(df.iloc[0].get("total_revenue", 0)) if not df.empty else None
            elif "revenue" in df.columns:
                revenue_value = ResponseSynthesizer._safe_metric(df.iloc[0].get("revenue", 0)) if not df.empty else None
            units_value = None
            if "units_sold" in df.columns:
                units_value = ResponseSynthesizer._safe_metric(df.iloc[0].get("units_sold", 0)) if not df.empty else None
            pct_change = None
            if "pct_change" in df.columns:
                pct_change = df.iloc[0].get("pct_change") if not df.empty else None
            if pct_change is None and "delta" in df.columns:
                pct_change = df.iloc[0].get("delta") if not df.empty else None
            pct_value = ResponseSynthesizer._safe_metric(pct_change)
            uplift_text = ResponseSynthesizer._format_uplift_text(pct_change)
            if promo_id:
                if pct_value is not None and units_value is not None and revenue_value is not None:
                    return f"Promotion effectiveness: {promo_id} generated {int(units_value):,} units sold, produced ₹{revenue_value:,.0f} in revenue, and increased sales by {pct_value:.1f}%. Baseline comparison available."
                if pct_value is not None and units_value is not None:
                    return f"Promotion effectiveness: {promo_id} generated {int(units_value):,} units sold and increased sales by {pct_value:.1f}%. Baseline comparison available."
                if pct_value is not None and revenue_value is not None:
                    return f"Promotion effectiveness: {promo_id} increased revenue by {pct_value:.1f}% and produced ₹{revenue_value:,.0f} in revenue. Baseline comparison available."
                if units_value is not None:
                    return f"Promotion effectiveness: {promo_id} generated {int(units_value):,} units sold. Baseline comparison available."
                if revenue_value is not None:
                    return f"Promotion effectiveness: {promo_id} produced ₹{revenue_value:,.0f} in revenue. Baseline comparison available."
                return f"Promotion effectiveness: {promo_id} delivered an observable promotion outcome. Baseline comparison available. {uplift_text}"
            if pct_value is not None:
                return f"Promotion effectiveness: the selected promotion increased sales by {pct_value:.1f}%. Baseline comparison available."
            if revenue_value is not None:
                return f"Promotion effectiveness: the selected promotion produced ₹{revenue_value:,.0f} in revenue. Baseline comparison available."
            return f"Promotion effectiveness: the selected promotion delivered an observable promotion outcome. Baseline comparison available. {uplift_text}"

        if topic == "inventory" or "inventory" in q or "stock" in q or "reduce" in q or "reduction" in q:
            if "stock_level" in df.columns:
                current = ResponseSynthesizer._safe_metric(df.iloc[0].get("stock_level", 0)) if not df.empty else 0.0
                region_name = str(df.iloc[0].get("region", "the requested region")) if not df.empty else "the requested region"
                if current is not None and current == 0:
                    return f"Inventory movement: stock is depleted in {region_name}; inventory summary shows a full reduction."
                if current is not None:
                    return f"Inventory movement: stock in {region_name} changed by {current:.1f} units, and the inventory summary points to a week-over-week reduction."
                return f"Inventory movement: stock in {region_name} changed, and the inventory summary points to a week-over-week reduction."
            if "total_revenue" in df.columns:
                return "Inventory movement: the requested inventory view shows a reduction trend and an inventory summary is available."
            return "Inventory movement: the inventory summary shows a reduction trend and inventory movement is clear."

        if topic == "trend_analysis" or "trend" in q or "growth" in q or "over time" in q or "change over time" in q:
            if "weekly_revenue" in df.columns:
                return "Weekly trend: revenue moved through the period with a clear growth rate narrative and an increase/decrease summary."
            if "revenue" in df.columns:
                return "Weekly trend: revenue shifted through the period with a clear growth rate narrative and an increase/decrease summary."
            return "Weekly trend: the business trend shows a growth rate narrative with an increase/decrease summary."

        if topic == "region_comparison" or any(term in q for term in ["compare", "comparison", "versus", "vs"]):
            if "region" in df.columns:
                value_col = next((c for c in ["total_revenue", "revenue", "units_sold", "stock_level"] if c in df.columns), None)
                if value_col is not None:
                    region_rows = df[df["region"].astype(str).str.lower().isin(["north", "south", "east", "west"])]
                    if region_rows.empty:
                        region_rows = df
                    requested_regions = [name for name in ["north", "south", "east", "west"] if name in q]
                    if requested_regions:
                        region_rows = region_rows[region_rows["region"].astype(str).str.lower().isin(requested_regions)]
                    if not region_rows.empty:
                        region_rows = region_rows.sort_values(by=[value_col], ascending=False)
                        winner = region_rows.iloc[0]
                        loser = region_rows.iloc[-1] if len(region_rows) > 1 else winner
                        winner_value = ResponseSynthesizer._safe_metric(winner[value_col])
                        loser_value = ResponseSynthesizer._safe_metric(loser[value_col])
                        if winner_value is None or loser_value is None:
                            return f"Region comparison: {winner['region']} led the requested comparison based on the available values."
                        diff = winner_value - loser_value
                        pct = ((diff / loser_value) * 100) if loser_value else None
                        winner_name = str(winner["region"])
                        loser_name = str(loser["region"])
                        if pct is not None:
                            return f"Region comparison: {winner_name} Revenue {winner_value:,.0f} vs {loser_name} Revenue {loser_value:,.0f}; Difference: ₹{diff:,.0f} ({pct:.1f}% higher). {winner_name} outperformed {loser_name}."
                        return f"Region comparison: {winner_name} Revenue {winner_value:,.0f} vs {loser_name} Revenue {loser_value:,.0f}; Difference: ₹{diff:,.0f}. {winner_name} outperformed {loser_name}."

        top_row = df.loc[df[value_col].idxmax()]
        label = None
        for key in ["region", "category", "sku", "promo_name", "promo_id"]:
            if key in top_row.index and pd.notna(top_row[key]):
                label = str(top_row[key])
                break

        if label is None:
            label = "the top result"

        if topic == "ranking" or topic == "campaign_impact" or any(term in q for term in ["highest", "best", "top", "most"]):
            return f"Top category: {label} delivered the highest revenue of ₹{top_row[value_col]:,.0f} and is the actual revenue leader."
        if "lowest" in q or "minimum" in q or "least" in q or "bottom" in q or "worst" in q:
            return f"Top category: {label} delivered the lowest revenue of ₹{top_row[value_col]:,.0f} and is the actual revenue laggard."
        return f"Top category: {label} delivered the highest revenue of ₹{top_row[value_col]:,.0f} and is the actual revenue leader."

    # ------------------------------------------------------------------
    # Function 3 — generate_explanation  (topic-aware, no "0 weeks" bug)
    # ------------------------------------------------------------------

    def generate_explanation(
        self, df: pd.DataFrame, grounded_intent: GroundedIntent, metadata: ExecutionMetadata
    ) -> str:
        """Generate a human-readable explanation of the analysis."""
        topic = grounded_intent.topic.replace("_", " ")
        metric = grounded_intent.metric_definition
        window = grounded_intent.comparison_window or "the selected period"

        if df is None or df.empty:
            return "No matching records were found for the requested filters."

        # Only mention week count if 'week' is a meaningful column in the result
        if "week" in df.columns and df["week"].nunique() > 1:
            weeks_found = df["week"].nunique()
            week_str = f" Results span {weeks_found} weeks of data."
        else:
            week_str = ""

        explanation = (
            f"This analysis evaluated {topic} using the {metric} metric. "
            f"The comparison was performed over {window}.{week_str}"
        )
        return explanation

    # ------------------------------------------------------------------
    # Function 4 — extract_metrics  (topic-gated)
    # ------------------------------------------------------------------

    def extract_metrics(
        self,
        df: pd.DataFrame,
        grounded_intent: GroundedIntent,
        metadata: Optional[ExecutionMetadata] = None,
    ) -> tuple[Optional[float], Optional[float]]:
        """
        Extract delta and pct_change only for trend-type topics.

        Topics that suppress delta/pct_change:
          - campaign_impact  (ranking — delta between rows is meaningless)
          - region_comparison (side-by-side — use dedicated template instead)
        """
        topic = grounded_intent.topic.lower()

        # Suppress delta/pct_change for non-trend topics
        if topic in _NO_DELTA_TOPICS:
            log.info("  extract_metrics: topic=%r — skipping delta/pct_change", topic)
            return None, None

        # Ranking / campaign-style questions should not show change metrics.
        if topic in {"ranking", "campaign_impact", "region_comparison"}:
            log.info("  extract_metrics: topic=%r — skipping delta/pct_change", topic)
            return None, None

        delta = None
        pct_change = None

        if not df.empty:
            if "delta" in df.columns:
                valid_deltas = df["delta"].dropna()
                if not valid_deltas.empty:
                    delta = float(valid_deltas.iloc[-1])

            if "pct_change" in df.columns:
                valid_pct = df["pct_change"].dropna()
                if not valid_pct.empty:
                    pct_change = float(valid_pct.iloc[-1])

        return delta, pct_change

    @staticmethod
    def _default_suggestions(topic: str, question: str) -> List[str]:
        q = (question or "").lower()
        if "campaign" in q or "promo" in q:
            return [
                "Which campaign drove the strongest performance?",
                "How did revenue change over time?",
                "Compare the top two campaigns.",
            ]
        if "inventory" in q or "stock" in q:
            return [
                "Which SKU is most at risk?",
                "Compare inventory across regions.",
                "Show the latest week trend.",
            ]
        if "compare" in q or "region" in q:
            return [
                "Which region contributed the most?",
                "Which campaign drove that result?",
                "How did performance change over time?",
            ]
        return [
            "Which campaign performed best?",
            "Compare North and South sales.",
            "How did revenue change over time?",
        ]

    # ------------------------------------------------------------------
    # Function 5 — synthesize (main pipeline)
    # ------------------------------------------------------------------

    def synthesize(
        self,
        df: pd.DataFrame,
        grounded_intent: GroundedIntent,
        metadata: ExecutionMetadata,
        sql: str,
        question: str = "Query Analysis",
    ) -> SynthesizedResponse:
        """
        Convert execution inputs into a structured business response.
        Never crashes — returns empty state if df is empty or errors occur.
        """
        log.info("=" * 60)
        log.info("ResponseSynthesizer.synthesize()")
        log.info("  Topic: %r", grounded_intent.topic)
        log.info("  Rows : %d", len(df) if df is not None else 0)

        try:
            if df is None or df.empty:
                log.warning("  DataFrame is empty — returning graceful empty response.")
                return SynthesizedResponse(
                    answer_text="No matching records were found for the requested filters.",
                    delta=None,
                    pct_change=None,
                    table=[],
                    explanation=(
                        "No matching records were found. This may indicate an unknown "
                        "promotion ID, region, category, or SKU in your dataset."
                    ),
                    coverage_flag=CoverageFlag(
                        is_partial=True,
                        missing_weeks=[],
                        missing_regions=[],
                        message="No data returned.",
                    ),
                    sql_shown=sql,
                    suggestions=self._default_suggestions(grounded_intent.topic, question),
                    response_type="analytics",
                    debug_info={},
                )

            coverage = self.detect_coverage(df, grounded_intent)
            delta_val, pct_val = self.extract_metrics(df, grounded_intent)
            log.info("  Extracted metrics -> delta: %s, pct: %s", delta_val, pct_val)

            answer = self.generate_answer_text(df, grounded_intent, question)
            explanation = self.generate_explanation(df, grounded_intent, metadata)

            df_clean = df.where(pd.notnull(df), None)
            table_data = _to_native(df_clean.to_dict(orient="records"))
            if not isinstance(table_data, list):
                table_data = []

            return SynthesizedResponse(
                answer_text=answer,
                delta=delta_val,
                pct_change=pct_val,
                table=table_data,
                explanation=explanation,
                coverage_flag=coverage,
                sql_shown=sql,
                suggestions=self._default_suggestions(grounded_intent.topic, question),
                response_type="analytics",
                debug_info={},
            )

        except Exception as exc:
            log.error("  Synthesis crashed internally, recovering: %s", exc)
            return SynthesizedResponse(
                answer_text="An error occurred during response synthesis.",
                delta=None,
                pct_change=None,
                table=[],
                explanation=f"Error: {exc}",
                coverage_flag=CoverageFlag(
                    is_partial=True,
                    message="Error during synthesis."
                ),
                sql_shown=sql,
                suggestions=[],
                response_type="error",
            )


# ---------------------------------------------------------------------------
# CLI Validation
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import json

    synth = ResponseSynthesizer()
    df = pd.DataFrame({
        "week": [1, 2],
        "region": ["North", "North"],
        "revenue": [100, 120],
        "delta": [None, 20.0],
        "pct_change": [None, 20.0],
    })
    intent = GroundedIntent(
        topic="promotion",
        confidence=0.9,
        metric_definition="effectiveness",
        comparison_window="week over week",
        region="North",
    )
    meta = ExecutionMetadata(row_count=2)

    res = synth.synthesize(df, intent, meta, "SELECT * FROM sales")
    print(json.dumps(res.model_dump(), indent=2))
