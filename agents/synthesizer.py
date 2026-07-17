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
  - Coverage detection (missing regions/weeks)
  - Answer text generation (executive summary)
  - Metric extraction (delta / pct_change)
  - Explanation generation
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
EXPECTED_REGIONS = {"North", "South", "East", "West"}
EXPECTED_WEEKS = set(range(1, 53))

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
    business-friendly responses.
    """

    def __init__(self) -> None:
        log.info("ResponseSynthesizer initialised.")

    # ------------------------------------------------------------------
    # Function 1 — detect_coverage
    # ------------------------------------------------------------------

    def detect_coverage(
        self, df: pd.DataFrame, grounded_intent: GroundedIntent
    ) -> CoverageFlag:
        """Determine whether data coverage is complete for weeks and regions."""
        missing_weeks: List[int] = []
        missing_regions: List[str] = []

        if not df.empty:
            if "week" in df.columns:
                found_weeks = set(df["week"].dropna().unique())
                missing_weeks = sorted(list(EXPECTED_WEEKS - found_weeks))

            if "region" in df.columns:
                found_regions = set(df["region"].dropna().unique())
                missing_regions = sorted(list(EXPECTED_REGIONS - found_regions))

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
    # Function 2 — generate_answer_text
    # ------------------------------------------------------------------

    def generate_answer_text(
        self, df: pd.DataFrame, grounded_intent: GroundedIntent
    ) -> str:
        """Create a concise executive summary based on the topic."""
        topic = grounded_intent.topic.lower()

        if topic == "promotion":
            return "Promotion effectiveness improved sales performance."
        elif topic == "inventory":
            return "Inventory levels have been adjusted based on the latest metrics."
        elif topic == "region_comparison":
            return "Regional comparison highlights varying performance across territories."
        elif topic == "campaign_impact":
            return "Campaign impact analysis identifies key drivers of revenue."
        else:
            return "Analysis complete based on the requested criteria."

    # ------------------------------------------------------------------
    # Function 3 — generate_explanation
    # ------------------------------------------------------------------

    def generate_explanation(
        self, df: pd.DataFrame, grounded_intent: GroundedIntent, metadata: ExecutionMetadata
    ) -> str:
        """Generate a human-readable explanation of the analysis."""
        topic = grounded_intent.topic.replace("_", " ")
        metric = grounded_intent.metric_definition
        window = grounded_intent.comparison_window or "the selected period"

        weeks_found = 0
        if not df.empty and "week" in df.columns:
            weeks_found = df["week"].nunique()

        explanation = (
            f"This analysis evaluated {topic} "
            f"using the {metric} metric. "
            f"The comparison was performed over {window}. "
            f"Results are based on {weeks_found} weeks of available data."
        )
        return explanation

    # ------------------------------------------------------------------
    # Function 4 — extract_metrics
    # ------------------------------------------------------------------

    def extract_metrics(self, df: pd.DataFrame) -> tuple[Optional[float], Optional[float]]:
        """
        Extract delta and pct_change from the DataFrame.
        Uses the last valid non-NaN value if multiple rows exist.
        """
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
                log.warning("  DataFrame is empty — returning empty response.")
                return SynthesizedResponse(
                    answer_text="No data available for the requested analysis.",
                    delta=None,
                    pct_change=None,
                    table=[],
                    explanation="No matching records were found.",
                    coverage_flag=CoverageFlag(
                        is_partial=True,
                        missing_weeks=list(EXPECTED_WEEKS),
                        missing_regions=list(EXPECTED_REGIONS),
                        message="Data coverage is partial (empty).",
                    ),
                    sql_shown=sql,
                )

            # Step 1: Coverage
            coverage = self.detect_coverage(df, grounded_intent)

            # Step 2: Metric Extraction
            delta_val, pct_val = self.extract_metrics(df)
            log.info("  Extracted metrics -> delta: %s, pct: %s", delta_val, pct_val)

            # Step 3: Answer Generation
            answer = self.generate_answer_text(df, grounded_intent)

            # Step 4: Explanation Generation
            explanation = self.generate_explanation(df, grounded_intent, metadata)

            # Step 5: Convert DataFrame to table
            # replace NaN with None for valid JSON serialization
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
        comparison_window="week over week"
    )
    meta = ExecutionMetadata(row_count=2)

    res = synth.synthesize(df, intent, meta, "SELECT * FROM sales")
    print(json.dumps(res.model_dump(), indent=2))
