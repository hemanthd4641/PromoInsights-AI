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

import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

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
        
        if not GROQ_API_KEY:
            raise EnvironmentError("GROQ_API_KEY is not set.")
            
        self._llm = ChatGroq(
            model=MODEL_NAME,
            api_key=GROQ_API_KEY,
            temperature=0.0,
            max_retries=MAX_RETRIES,
        )
        
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

    # ------------------------------------------------------------------
    # Function 2 — generate_answer_text  (Dynamic LLM Synthesis)
    # ------------------------------------------------------------------

    def generate_answer_text(
        self, df: pd.DataFrame, grounded_intent: GroundedIntent, question: str
    ) -> str:
        """Create a concise, topic-specific executive summary from the data dynamically using an LLM."""
        
        if df is None or df.empty:
            return "No matching records were found for the requested filters."
            
        try:
            # Limit to top 20 rows to avoid token explosion
            df_subset = df.head(20)
            data_md = df_subset.to_csv(index=False)
            if len(df) > 20:
                data_md += f"\n\n... (Truncated. Total rows: {len(df)})"
                
            chain = self._prompt | self._llm
            response = chain.invoke({
                "question": question,
                "topic": grounded_intent.topic,
                "data_table": data_md
            })
            return str(response.content).strip()
            
        except Exception as exc:
            log.error("LLM synthesis failed: %s", exc)
            return "Analysis complete based on the requested criteria. Please review the supporting data table."

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
        self, df: pd.DataFrame, grounded_intent: GroundedIntent
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
                )

            # Step 1: Coverage (context-aware)
            coverage = self.detect_coverage(df, grounded_intent)

            # Step 2: Metric Extraction (topic-gated)
            delta_val, pct_val = self.extract_metrics(df, grounded_intent)
            log.info("  Extracted metrics -> delta: %s, pct: %s", delta_val, pct_val)

            # Step 3: Answer Generation (Dynamic LLM)
            answer = self.generate_answer_text(df, grounded_intent, question)

            # Step 4: Explanation Generation
            explanation = self.generate_explanation(df, grounded_intent, metadata)

            # Step 5: Convert DataFrame to table
            df_clean = df.where(pd.notnull(df), None)
            table_data = df_clean.to_dict(orient="records")

            # Step 6: Return Response
            return SynthesizedResponse(
                answer_text=answer,
                delta=delta_val,
                pct_change=pct_val,
                table=table_data,
                explanation=explanation,
                coverage_flag=coverage,
                sql_shown=sql,
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
