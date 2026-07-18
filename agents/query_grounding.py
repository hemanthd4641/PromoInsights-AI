"""
agents/query_grounding.py
--------------------------
Phase 4 — Query Grounding Agent.

Converts a raw Intent (from Phase 3) into a fully specified GroundedIntent
by resolving vague business terms (e.g. "improve sales", "growth", "best")
to precise metric definitions, baseline formulas, and comparison windows.

Grounding uses ChromaDB retrieval (Phase 2) for semantic matching, then
applies deterministic keyword rules to guarantee consistency on repeated calls.

Usage:
    from agents.intent_classifier import Intent
    from agents.query_grounding import QueryGroundingAgent

    agent = QueryGroundingAgent()
    grounded = agent.ground(question, intent)
    print(grounded.model_dump_json(indent=2))
"""

import logging
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator

# ---------------------------------------------------------------------------
# Bootstrap: project root on sys.path
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import LOG_LEVEL
from agents.intent_classifier import Intent
from rag.retriever import retrieve_grounding

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
# Maximum few-shot examples to attach
# ---------------------------------------------------------------------------
MAX_FEW_SHOT: int = 3

# ---------------------------------------------------------------------------
# Fallback when no rule matches
# ---------------------------------------------------------------------------
_FALLBACK = {
    "metric_definition": "generic business metric",
    "baseline_formula": None,
    "comparison_window": None,
}

# Topic-level fallback overrides (when keyword match fails)
_TOPIC_DEFAULTS: Dict[str, Dict[str, Any]] = {
    "promotion": {
        "metric_definition": "effectiveness",
        "baseline_formula": "(promo_sales - baseline_sales) / baseline_sales * 100",
        "comparison_window": "4-week pre-promotion baseline",
    },
    "inventory": {
        "metric_definition": "reduction",
        "baseline_formula": "(value_before - value_after) / value_before * 100",
        "comparison_window": "negative week-over-week delta",
    },
    "region_comparison": {
        "metric_definition": "regional_performance",
        "baseline_formula": "SUM(revenue) or SUM(units_sold) GROUP BY region ORDER BY metric DESC",
        "comparison_window": "selected promotion or time window",
    },
    "campaign_impact": {
        "metric_definition": "campaign_impact",
        "baseline_formula": "SUM(revenue during promo) - SUM(expected baseline revenue)",
        "comparison_window": "full campaign window (start_week to end_week)",
    },
}


# ---------------------------------------------------------------------------
# Pydantic Output Model
# ---------------------------------------------------------------------------


class GroundedIntent(BaseModel):
    """
    Fully grounded intent — ready for SQL generation.

    Fields
    ------
    topic               : Classified question type (from Intent).
    region              : Geographic region extracted from the question.
    sku                 : SKU identifier extracted from the question.
    category            : Product category extracted from the question.
    time_window         : Time reference extracted from the question.
    confidence          : Classification confidence score [0.0, 1.0].
    metric_definition   : Resolved business metric name (never empty).
    baseline_formula    : SQL/mathematical formula for the metric.
    comparison_window   : Time or data window the metric is measured over.
    few_shot_examples   : Relevant SQL examples from the RAG bank (max 3).
    """

    topic: str = Field(description="Classified question type.")
    region: Optional[str] = Field(default=None)
    sku: Optional[str] = Field(default=None)
    category: Optional[str] = Field(default=None)
    time_window: Optional[str] = Field(default=None)
    confidence: float = Field(ge=0.0, le=1.0)

    metric_definition: str = Field(
        description="Resolved metric name — must never be empty."
    )
    baseline_formula: Optional[str] = Field(
        default=None,
        description="Formula or SQL expression used to compute the baseline.",
    )
    comparison_window: Optional[str] = Field(
        default=None,
        description="Time window or data window over which the metric is evaluated.",
    )
    few_shot_examples: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Up to 3 relevant SQL few-shot examples from ChromaDB.",
    )

    @field_validator("metric_definition")
    @classmethod
    def metric_definition_not_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("metric_definition cannot be empty.")
        return v.strip()

    @field_validator("confidence")
    @classmethod
    def validate_confidence(cls, v: float) -> float:
        if not (0.0 <= v <= 1.0):
            raise ValueError(f"confidence must be between 0.0 and 1.0, got {v}")
        return round(v, 4)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
def _pick_best_definition(definitions: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """
    From a list of ChromaDB glossary hits, return the one with the highest
    similarity_score. Returns None if the list is empty.
    """
    if not definitions:
        return None
    return max(definitions, key=lambda d: d.get("similarity_score", 0.0))


def _build_few_shot_list(examples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Convert ChromaDB few-shot hits to a clean list capped at MAX_FEW_SHOT.
    Strips the similarity_score from the public output.
    """
    out = []
    for ex in examples[:MAX_FEW_SHOT]:
        out.append(
            {
                "question": ex.get("question", ""),
                "question_type": ex.get("question_type", ""),
                "sql": ex.get("sql", ""),
                "explanation": ex.get("explanation", ""),
            }
        )
    return out


# ---------------------------------------------------------------------------
# Query Grounding Agent
# ---------------------------------------------------------------------------


class QueryGroundingAgent:
    """
    Converts an Intent into a fully grounded GroundedIntent by:

    1. Applying deterministic keyword rules (ensures consistency).
    2. Falling back to ChromaDB semantic retrieval if no rule matches.
    3. Falling back to topic-level defaults if retrieval also fails.
    4. Attaching up to 3 relevant few-shot SQL examples from ChromaDB.
    """

    @staticmethod
    def _apply_keyword_grounding(question: str, intent: Intent) -> Dict[str, Any]:
        q = (question or "").lower()
        topic = (intent.topic or "").lower()
        metric_definition = None
        baseline_formula = None
        comparison_window = None

        if topic == "promotion" or any(term in q for term in ["promo", "promotion", "improve", "improvement", "lift", "impact", "after", "baseline"]):
            metric_definition = "promotion_effectiveness"
            if "revenue" in q:
                metric_definition = "promotion_revenue_effectiveness"
            baseline_formula = "promo_period_metric - baseline_metric"
            comparison_window = "promotion window compared with a pre-promotion baseline"
        elif topic == "inventory" or any(term in q for term in ["inventory", "stock", "reduction"]):
            metric_definition = "inventory_reduction"
            baseline_formula = "(stock_before - stock_after) / stock_before * 100"
            comparison_window = "week-over-week inventory movement"
        elif topic == "region_comparison" or any(term in q for term in ["compare", "comparison", "region", "which region"]):
            metric_definition = "regional_performance"
            baseline_formula = "SUM(revenue) GROUP BY region"
            comparison_window = "selected period across regions"
        elif topic == "ranking" or any(term in q for term in ["rank", "highest", "lowest", "best", "worst"]):
            metric_definition = "ranked_performance"
            baseline_formula = "ORDER BY metric DESC LIMIT 1"
            comparison_window = "overall period"
        elif topic == "trend_analysis" or any(term in q for term in ["trend", "growth", "over time"]):
            metric_definition = "trend_analysis"
            baseline_formula = "current_value - prior_value"
            comparison_window = "time series across the selected period"
        elif topic == "anomaly_detection" or any(term in q for term in ["anomaly", "outlier", "spike", "drop"]):
            metric_definition = "anomaly_detection"
            baseline_formula = "z_score or deviation from rolling average"
            comparison_window = "recent period"
        elif topic == "metric_lookup" or any(term in q for term in ["metric", "summary"]):
            metric_definition = "business_metric"
            baseline_formula = None
            comparison_window = "selected period"
        else:
            metric_definition = "generic business metric"
            baseline_formula = None
            comparison_window = None

        return {
            "metric_definition": metric_definition,
            "baseline_formula": baseline_formula,
            "comparison_window": comparison_window,
        }

    def ground(self, question: str, intent: Intent) -> GroundedIntent:
        """
        Ground an intent into a resolved metric definition.

        Args:
            question : Original natural-language question from the user.
            intent   : Typed Intent object produced by Phase 3 classifier.

        Returns:
            GroundedIntent with resolved metric, formula, window, and examples.
        """
        log.info("=" * 60)
        log.info("QueryGroundingAgent.ground()")
        log.info("  Question : %r", question)
        log.info("  Topic    : %r", intent.topic)

        # ------------------------------------------------------------------
        # Step 1 — Semantic retrieval via ChromaDB (Phase 2 integration)
        # ------------------------------------------------------------------
        log.info("  Calling retrieve_grounding() for semantic matching")
        try:
            retrieval = retrieve_grounding(
                query=question,
                top_k_definitions=3,
                top_k_examples=MAX_FEW_SHOT,
            )
            best_def = _pick_best_definition(retrieval.get("definitions", []))
        except Exception as exc:
            log.warning("  Retrieval failed: %s — using fallback", exc)
            best_def = None
            retrieval = {"definitions": [], "examples": []}

        keyword_grounding = self._apply_keyword_grounding(question, intent)

        if best_def and not keyword_grounding.get("comparison_window"):
            metric_definition = best_def.get("term", "generic business metric")
            baseline_formula = best_def.get("formula") or None
            comparison_window = None
            log.info("  [RAG match] metric=%r (score=%.4f)",
                     metric_definition,
                     best_def.get("similarity_score", 0.0))
        else:
            # ----------------------------------------------------------
            # Step 2 — Topic-level default fallback
            # ----------------------------------------------------------
            topic_default = _TOPIC_DEFAULTS.get(intent.topic, _FALLBACK)
            metric_definition = keyword_grounding.get("metric_definition") or topic_default["metric_definition"]
            baseline_formula = keyword_grounding.get("baseline_formula") or topic_default.get("baseline_formula")
            comparison_window = keyword_grounding.get("comparison_window") or topic_default.get("comparison_window")
            log.info("  [Keyword fallback] metric=%r", metric_definition)

        # ------------------------------------------------------------------
        # Step 3 — Always fetch few-shot examples from ChromaDB (Phase 2)
        # ------------------------------------------------------------------
        few_shot_examples: List[Dict[str, Any]] = []
        try:
            fs_retrieval = retrieve_grounding(
                query=question,
                top_k_definitions=1,
                top_k_examples=MAX_FEW_SHOT,
            )
            few_shot_examples = _build_few_shot_list(
                fs_retrieval.get("examples", [])
            )
        except Exception as exc:
            log.warning("  Few-shot retrieval failed: %s", exc)

        log.info("  metric_definition  : %r", metric_definition)
        log.info("  baseline_formula   : %r", baseline_formula)
        log.info("  comparison_window  : %r", comparison_window)
        log.info("  few_shot_examples  : %d attached", len(few_shot_examples))

        # ------------------------------------------------------------------
        # Step 4 — Assemble and validate GroundedIntent
        # ------------------------------------------------------------------
        return GroundedIntent(
            topic=intent.topic,
            region=intent.region,
            sku=intent.sku,
            category=intent.category,
            time_window=intent.time_window,
            confidence=intent.confidence,
            metric_definition=metric_definition,
            baseline_formula=baseline_formula,
            comparison_window=comparison_window,
            few_shot_examples=few_shot_examples,
        )


# ---------------------------------------------------------------------------
# Quick CLI validation
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import json
    from agents.intent_classifier import IntentClassifier

    SAMPLE_QUESTIONS = [
        "Did PROMO_001 improve sales in South region?",
        "Did inventory reduction happen in West region?",
        "Which campaign performed best?",
        "Show revenue growth for Electronics category.",
    ]

    clf = IntentClassifier()
    agent = QueryGroundingAgent()

    print("\n" + "=" * 70)
    print("QUERY GROUNDING AGENT — VALIDATION RUN")
    print("=" * 70)

    for i, q in enumerate(SAMPLE_QUESTIONS, 1):
        print(f"\n[Test {i}] {q}")
        intent = clf.classify(q)
        grounded = agent.ground(q, intent)
        print(json.dumps(grounded.model_dump(), indent=2))
        print("-" * 60)
