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
    # Function 2 — generate_answer_text  (topic-aware)
    # ------------------------------------------------------------------

    def generate_answer_text(
        self, df: pd.DataFrame, grounded_intent: GroundedIntent
    ) -> str:
        """Create a concise, topic-specific executive summary from the data."""
        topic = grounded_intent.topic.lower()

        if df is None or df.empty:
            return "No matching records were found for the requested filters."

        # ── PROMOTION effectiveness ──────────────────────────────────────────
        if topic == "promotion":
            return self._answer_promotion(df, grounded_intent)

        # ── INVENTORY reduction ──────────────────────────────────────────────
        elif topic == "inventory":
            return self._answer_inventory(df, grounded_intent)

        # ── REGION COMPARISON ────────────────────────────────────────────────
        elif topic == "region_comparison":
            return self._answer_region_comparison(df, grounded_intent)

        # ── CAMPAIGN IMPACT / RANKING ────────────────────────────────────────
        elif topic == "campaign_impact":
            return self._answer_campaign_impact(df, grounded_intent)

        return "Analysis complete based on the requested criteria."

    # ------------------------------------------------------------------
    # Topic-specific answer builders
    # ------------------------------------------------------------------

    def _answer_promotion(self, df: pd.DataFrame, intent: GroundedIntent) -> str:
        """Generate promotion effectiveness answer."""
        region_str = f" in the {intent.region} region" if intent.region else ""

        # Look for a sales_lift column first (most specific)
        if "sales_lift" in df.columns:
            lift = df["sales_lift"].iloc[0]
            if lift is not None and pd.notna(lift):
                direction = "improved" if lift > 0 else "declined"
                return (
                    f"Promotion effectiveness{region_str}: sales {direction} by "
                    f"{abs(lift):.1f}% compared to the pre-promotion baseline."
                )

        # Look for promo vs baseline units
        if "promo_units_sold" in df.columns and "baseline_units_sold" in df.columns:
            promo = df["promo_units_sold"].iloc[0]
            baseline = df["baseline_units_sold"].iloc[0]
            if baseline and baseline != 0:
                lift = ((promo - baseline) / baseline) * 100
                direction = "improved" if lift > 0 else "declined"
                return (
                    f"Promotion sales{region_str} {direction} by {abs(lift):.1f}% "
                    f"vs the 4-week baseline ({int(baseline):,} → {int(promo):,} units)."
                )

        # Fallback: total revenue
        rev_col = next((c for c in ["total_revenue", "revenue"] if c in df.columns), None)
        if rev_col:
            total = df[rev_col].sum()
            return f"Promotion generated ₹{total:,.2f} in revenue{region_str}."

        return f"Promotion analysis complete{region_str}."

    def _answer_inventory(self, df: pd.DataFrame, intent: GroundedIntent) -> str:
        """Generate inventory analysis answer."""
        region_str = f" in the {intent.region} region" if intent.region else ""

        stock_col = next(
            (c for c in ["avg_stock", "stock_level", "total_stock", "avg_stock_level"]
             if c in df.columns), None
        )

        if stock_col and len(df) >= 2:
            first = df[stock_col].iloc[0]
            last = df[stock_col].iloc[-1]
            if pd.notna(first) and pd.notna(last) and first != 0:
                change = ((last - first) / first) * 100
                direction = "decreased" if change < 0 else "increased"
                return (
                    f"Inventory{region_str} {direction} by {abs(change):.1f}% "
                    f"from {first:,.0f} to {last:,.0f} units over the analysis period."
                )

        if stock_col and len(df) == 1:
            val = df[stock_col].iloc[0]
            return f"Current inventory level{region_str}: {val:,.0f} units."

        return f"Inventory analysis complete{region_str}. Review the supporting data table."

    def _answer_region_comparison(self, df: pd.DataFrame, intent: GroundedIntent) -> str:
        """Generate a side-by-side region comparison answer."""
        rev_col = next(
            (c for c in ["total_revenue", "revenue"] if c in df.columns), None
        )
        region_col = "region" if "region" in df.columns else None

        if region_col and rev_col and len(df) >= 2:
            # Sort by revenue descending
            df_sorted = df.sort_values(rev_col, ascending=False).reset_index(drop=True)
            top = df_sorted.iloc[0]
            bottom = df_sorted.iloc[1]

            top_name = top[region_col]
            top_rev = top[rev_col]
            bottom_name = bottom[region_col]
            bottom_rev = bottom[rev_col]

            diff = top_rev - bottom_rev
            if bottom_rev and bottom_rev != 0:
                pct_diff = (diff / bottom_rev) * 100
                pct_str = f" ({pct_diff:.1f}% more)"
            else:
                pct_str = ""

            lines = []
            for _, row in df_sorted.iterrows():
                lines.append(
                    f"{row[region_col]} Revenue: ₹{row[rev_col]:,.2f}"
                )
            comparison = " | ".join(lines)
            summary = (
                f"{top_name} generated more revenue than {bottom_name}{pct_str}. "
                f"Difference: ₹{diff:,.2f}."
            )
            return f"{comparison}. {summary}"

        if region_col and rev_col and len(df) == 1:
            row = df.iloc[0]
            return f"{row[region_col]} Revenue: ₹{row[rev_col]:,.2f}."

        return "Regional comparison data is available in the supporting table."

    def _answer_campaign_impact(self, df: pd.DataFrame, intent: GroundedIntent) -> str:
        """Identify and name the top performer for campaign/category/region rankings."""
        rev_col = next(
            (c for c in ["total_revenue", "revenue"] if c in df.columns), None
        )

        # Identify winner column
        winner_col = None
        winner_label = "item"
        for col, label in [
            ("promo_name", "campaign"),
            ("promo_id", "campaign"),
            ("category", "category"),
            ("sku", "SKU"),
            ("region", "region"),
        ]:
            if col in df.columns:
                winner_col = col
                winner_label = label
                break

        if winner_col and rev_col and not df.empty:
            df_sorted = df.sort_values(rev_col, ascending=False).reset_index(drop=True)
            winner_name = df_sorted[winner_col].iloc[0]
            winner_rev = df_sorted[rev_col].iloc[0]
            return (
                f"The highest-performing {winner_label} was {winner_name} "
                f"with total revenue of ₹{winner_rev:,.2f}."
            )

        if winner_col and not df.empty and rev_col is None:
            winner_name = df[winner_col].iloc[0]
            return f"The top-performing {winner_label} is {winner_name}."

        if not df.empty:
            return "Campaign impact analysis complete. Review the supporting data table."

        return "No campaign records were found for the requested criteria."

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

            # Step 3: Answer Generation (topic-aware)
            answer = self.generate_answer_text(df, grounded_intent)

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
