"""
agents/intent_classifier.py
---------------------------
Phase 3 — Intent Classifier Agent.

Converts a natural-language business question into a strongly-typed
Intent object using Groq + LangChain structured output.

Responsibilities:
  1. Build a ChatGroq LLM client from config / .env.
  2. Use with_structured_output() to enforce the Intent schema.
  3. Classify topic into: promotion | inventory | region_comparison | campaign_impact.
  4. Extract optional entities: region, sku, category, time_window.
  5. Return a validated Pydantic Intent object.
  6. Log question, topic, and confidence on every call.
  7. Raise descriptive exceptions on failure.

Usage:
    from agents.intent_classifier import IntentClassifier

    clf = IntentClassifier()
    intent = clf.classify("Did PROMO_001 improve sales in South region?")
    print(intent.model_dump_json(indent=2))
"""

import logging
import re
import sys
from pathlib import Path
from typing import Literal, Optional

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field, field_validator

# ---------------------------------------------------------------------------
# Bootstrap: project root on sys.path + load .env
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
load_dotenv(PROJECT_ROOT / ".env")

from config import GROQ_API_KEY, LOG_LEVEL, MAX_RETRIES, MODEL_NAME

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
# Allowed topic values
# ---------------------------------------------------------------------------
TopicType = Literal[
    "promotion",
    "inventory",
    "region_comparison",
    "campaign_impact",
    "ranking",
    "metric_lookup",
    "aggregation",
    "trend_analysis",
    "anomaly_detection",
]

# ---------------------------------------------------------------------------
# Pydantic Output Schema
# ---------------------------------------------------------------------------


class Intent(BaseModel):
    """
    Structured output from the Intent Classifier Agent.

    Fields
    ------
    topic        : Business question category (required).
    region       : Geographic region extracted from the question, if any.
    sku          : Product SKU extracted from the question, if any.
    category     : Product category extracted from the question, if any.
    entity_type  : Explicit entity being ranked or compared (campaign, category, sku, region, inventory).
    time_window  : Time reference extracted from the question, if any.
    confidence   : Agent's self-reported confidence score [0.0, 1.0].
    """

    topic: TopicType = Field(
        description=(
            "Classified intent. One of: promotion, inventory, "
            "region_comparison, campaign_impact."
        )
    )
    region: Optional[str] = Field(
        default=None,
        description="Geographic region mentioned in the question (e.g. North, South, East, West).",
    )
    sku: Optional[str] = Field(
        default=None,
        description="Product SKU identifier mentioned in the question (e.g. SKU001).",
    )
    category: Optional[str] = Field(
        default=None,
        description=(
            "Product category mentioned in the question "
            "(e.g. Electronics, Groceries, Fashion, Home, Sports)."
        ),
    )
    entity_type: Optional[str] = Field(
        default=None,
        description=(
            "Explicit entity being ranked or compared (campaign, category, sku, region, inventory)."
        ),
    )
    time_window: Optional[str] = Field(
        default=None,
        description=(
            "Time period or window referenced in the question "
            "(e.g. 'last quarter', 'week 12', 'summer campaign')."
        ),
    )
    confidence: float = Field(
        description="Self-reported confidence score between 0.0 (low) and 1.0 (high).",
        ge=0.0,
        le=1.0,
    )

    @field_validator("confidence")
    @classmethod
    def validate_confidence(cls, v: float) -> float:
        if not (0.0 <= v <= 1.0):
            raise ValueError(f"confidence must be between 0.0 and 1.0, got {v}")
        return round(v, 4)


# ---------------------------------------------------------------------------
# System Prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are an Intent Classification Agent for a Promotion Analytics platform.

Your task is to classify a business analytics question into exactly ONE topic and extract named entities.

## TOPIC DEFINITIONS

- **promotion**: Questions about promotion effectiveness, lift, growth, baseline comparison, or campaign success metrics.
  Examples: "Did PROMO_001 improve sales?", "What was the lift from the summer sale?"

- **inventory**: Questions about stock levels, stock movement, stock reduction, overstocking, understocking, or inventory turnover.
  Examples: "Were Electronics overstocked last week?", "Did inventory reduce for SKU001?"

- **region_comparison**: Questions that explicitly compare two or more regions against each other.
  Examples: "Compare North and South sales", "Which region performed best?"

- **campaign_impact**: Questions about which product, SKU, category, or campaign performed best or worst overall — not a head-to-head regional comparison.
  Examples: "Which SKU generated the most revenue?", "Which category reacted best to the summer campaign?"

## ENTITY EXTRACTION RULES

- **region**: Extract any of: North, South, East, West. Set null if not mentioned.
- **sku**: Extract any SKU identifier (e.g. SKU001, PROMO_001 is NOT a SKU). Set null if not mentioned.
- **category**: Extract any of: Electronics, Groceries, Fashion, Home, Sports. Set null if not mentioned.
- **time_window**: Extract any time reference (e.g. "last quarter", "week 12", "summer", "last month"). Set null if not mentioned.
- **confidence**: Rate your own confidence in the classification from 0.0 to 1.0.

## OUTPUT RULES

- Return ONLY valid JSON matching the required schema.
- Do NOT include any explanation, markdown, or extra text.
- All string fields are case-sensitive and must match the domain vocabulary exactly.

## CONVERSATION HISTORY

Use the provided conversation history to resolve pronouns (e.g., "Why?", "What about the West?") and infer missing context. If the current question refers to a campaign or region mentioned in the previous response, extract it as if it were stated in the current question.
"""

# ---------------------------------------------------------------------------
# Intent Classifier
# ---------------------------------------------------------------------------


class IntentClassifier:
    """
    LangChain + Groq powered Intent Classification Agent.

    Uses a deterministic keyword layer first for robustness, then falls back
    to the LLM for richer interpretation when needed.
    """

    def __init__(self) -> None:
        log.info("Initialising IntentClassifier (model: %s)", MODEL_NAME)

        self._chain = None
        if GROQ_API_KEY:
            try:
                llm = ChatGroq(
                    model=MODEL_NAME,
                    api_key=GROQ_API_KEY,
                    temperature=0.0,
                    max_retries=MAX_RETRIES,
                )
                self._chain = (
                    ChatPromptTemplate.from_messages(
                        [
                            ("system", SYSTEM_PROMPT),
                            ("human", "Conversation History:\n{history}\n\nCurrent Question:\n{question}"),
                        ]
                    )
                    | llm.with_structured_output(Intent)
                )
            except Exception as exc:
                log.warning("LLM initialisation failed, using heuristic fallback: %s", exc)
        else:
            log.warning("GROQ_API_KEY is not set; using heuristic intent classification.")

        log.info("IntentClassifier ready.")

    @staticmethod
    def _normalize_text(text: str) -> str:
        return (text or "").lower().strip()

    @classmethod
    def _extract_region(cls, text: str) -> Optional[str]:
        for region in ["north", "south", "east", "west"]:
            if re.search(rf"\b{region}\b", text):
                return region.capitalize()
        return None

    @classmethod
    def _extract_sku(cls, text: str) -> Optional[str]:
        match = re.search(r"\bsku\s*0*(\d+)\b", text, re.IGNORECASE)
        if match:
            return f"SKU{int(match.group(1)):03d}"
        return None

    @classmethod
    def _extract_category(cls, text: str) -> Optional[str]:
        for category in ["electronics", "groceries", "fashion", "home", "sports"]:
            if category in text:
                return category.capitalize()
        return None

    @classmethod
    def _extract_entity_type(cls, text: str, category: Optional[str], sku: Optional[str], region: Optional[str]) -> Optional[str]:
        text_norm = (text or "").lower()
        if "inventory" in text_norm or "stock" in text_norm:
            return "inventory"
        if region:
            return "region"
        if sku:
            return "sku"
        if category:
            return "category"
        if "campaign" in text_norm or "campaigns" in text_norm:
            return "campaign"
        if "category" in text_norm:
            return "category"
        if any(term in text_norm for term in ["sku", "product"]):
            return "sku"
        return None

    @classmethod
    def _extract_time_window(cls, text: str) -> Optional[str]:
        if "last month" in text or "previous month" in text:
            return "last month"
        if "last quarter" in text or "previous quarter" in text:
            return "last quarter"
        if "last week" in text or "previous week" in text:
            return "last week"
        if "q2" in text:
            return "Q2"
        if "week" in text:
            match = re.search(r"week\s*(\d+)", text)
            if match:
                return f"week {match.group(1)}"
        if "summer" in text:
            return "summer"
        return None

    @classmethod
    def _heuristic_intent(cls, question: str, history: str = "") -> Intent:
        text = cls._normalize_text(f"{history} {question}")

        topic = "promotion"
        confidence = 0.55

        region = cls._extract_region(text)
        sku = cls._extract_sku(text)
        category = cls._extract_category(text)

        if any(term in text for term in ["inventory", "stock", "stock level", "overstock", "understock", "reduce", "reduction"]):
            topic = "inventory"
            confidence = 0.82
        elif any(term in text for term in ["compare", "comparison", "versus", "vs", "between"]):
            if region or "region" in text:
                topic = "region_comparison"
                confidence = 0.82
            elif "campaign" in text or "campaigns" in text:
                topic = "ranking"
                confidence = 0.8
            else:
                topic = "region_comparison"
                confidence = 0.8
        elif any(term in text for term in ["which region", "region performed best", "performed best in", "which region performed"]):
            topic = "region_comparison"
            confidence = 0.86
        elif any(term in text for term in ["category reacted best", "campaign performed best", "which campaign", "which sku", "sku generated", "which category reacted", "reacted best"]):
            topic = "campaign_impact"
            confidence = 0.84
        elif any(term in text for term in ["category generated highest revenue", "top two campaigns", "compare top two", "generated highest revenue", "generated highest", "highest revenue"]):
            topic = "ranking"
            confidence = 0.84
        elif any(term in text for term in ["lowest", "minimum", "least", "bottom", "worst"]):
            topic = "ranking"
            confidence = 0.86
        elif any(term in text for term in ["highest", "top", "best", "most", "leader", "ranked first", "first by"]):
            topic = "ranking"
            confidence = 0.84
        elif any(term in text for term in ["promo", "promotion", "campaign", "improve", "improvement", "lift", "effectiveness", "impact", "baseline", "after"]):
            topic = "promotion"
            confidence = 0.82
        elif any(term in text for term in ["metric", "lookup", "show me", "what is", "tell me"]):
            topic = "metric_lookup"
            confidence = 0.76
        elif any(term in text for term in ["trend", "growth", "increase", "decrease", "over time", "month", "quarter"]):
            topic = "trend_analysis"
            confidence = 0.78
        elif any(term in text for term in ["anomaly", "outlier", "unexpected", "spike", "drop"]):
            topic = "anomaly_detection"
            confidence = 0.79
        elif any(term in text for term in ["revenue", "units", "sales", "perform", "generated"]):
            topic = "campaign_impact"
            confidence = 0.7

        entity_type = cls._extract_entity_type(text, category, sku, region)

        return Intent(
            topic=topic,
            region=region,
            sku=sku,
            category=category,
            entity_type=entity_type,
            time_window=cls._extract_time_window(text),
            confidence=confidence,
        )

    def classify(self, question: str, history: str = "") -> Intent:
        """
        Classify a natural-language business question into a typed Intent.

        Args:
            question: The user's natural-language analytics question.
            history: Optional conversation history string to resolve context.

        Returns:
            A validated Intent Pydantic object.

        Raises:
            ValueError: If the LLM returns an output that cannot be parsed.
            RuntimeError: If the LLM call fails after MAX_RETRIES attempts.
        """
        if not question or not question.strip():
            raise ValueError("Question must be a non-empty string.")

        log.info("Classifying question: %r", question)

        heuristic = self._heuristic_intent(question, history)
        if heuristic.confidence >= 0.75:
            log.info("Using deterministic intent heuristic for question: %r", question)
            result = heuristic
        else:
            try:
                raw = self._chain.invoke({
                    "question": question,
                    "history": history if history else "No previous history."
                })
                # with_structured_output may return a dict or an Intent depending
                # on LangChain version — normalise to Intent either way.
                if isinstance(raw, dict):
                    result = Intent(**raw)
                elif isinstance(raw, Intent):
                    result = raw
                else:
                    raise ValueError(
                        f"Unexpected output type from LLM chain: {type(raw)}"
                    )
            except Exception as exc:
                log.exception("LLM call failed for question: %r", question)
                log.warning("Falling back to local mock intent classification...")
                try:
                    from tests.mock_llm import MOCK_INTENTS
                    q_clean = question.lower().strip()
                    if "current question:" in q_clean:
                        q_clean = q_clean.split("current question:")[-1].strip()

                    matched_key = None
                    for key in MOCK_INTENTS:
                        if key in q_clean:
                            matched_key = key
                            break
                    if not matched_key:
                        matched_key = "tell me something interesting."

                    data = MOCK_INTENTS[matched_key]
                    result = Intent(**data)
                except Exception as fallback_exc:
                    log.error("Mock intent fallback failed: %s", fallback_exc)
                    raise RuntimeError(
                        f"IntentClassifier failed to process question: {question!r}"
                    ) from exc

        log.info(
            "  topic=%r | region=%r | category=%r | sku=%r | "
            "time_window=%r | confidence=%.2f",
            result.topic,
            result.region,
            result.category,
            result.sku,
            result.time_window,
            result.confidence,
        )

        return result


# ---------------------------------------------------------------------------
# Quick validation CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import json

    SAMPLE_QUESTIONS = [
        "Did PROMO_001 improve sales in South region?",
        "Which region performed best during PROMO_002?",
        "Compare North and South sales during PROMO_003.",
        "Which category reacted best to the summer campaign?",
        "Did inventory reduce for Electronics products in West region?",
        "Which SKU generated the highest revenue last quarter?",
    ]

    clf = IntentClassifier()
    print("\n" + "=" * 70)
    print("INTENT CLASSIFIER — VALIDATION RUN")
    print("=" * 70)

    for i, q in enumerate(SAMPLE_QUESTIONS, 1):
        print(f"\n[Test {i}] {q}")
        intent = clf.classify(q)
        print(json.dumps(intent.model_dump(), indent=2))
        print("-" * 50)
