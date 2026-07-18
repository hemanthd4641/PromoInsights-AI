"""
agents/orchestrator.py
-----------------------
Phase 9 -- Promotion Analytics Orchestrator.

Wires together all previous phases into a complete pipeline:
  1. Intent Classification
  2. Query Grounding
  3. SQL Generation
  4. Validation Loop (with retries)
  5. Execution
  6. Response Synthesis

Also maintains simple session context for multi-turn support.
Guarantees the pipeline never crashes; returns fallback responses on failure.

Usage:
    from agents.orchestrator import PromotionAnalyticsOrchestrator
    orchestrator = PromotionAnalyticsOrchestrator()
    response = orchestrator.handle("How did South perform?")
"""

import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

# ---------------------------------------------------------------------------
# Bootstrap project root
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from agents.executor import ExecutionAgent
from agents.intent_classifier import IntentClassifier
from agents.query_gen import QueryGenerationAgent
from agents.query_grounding import QueryGroundingAgent
from agents.synthesizer import (
    CoverageFlag,
    ResponseSynthesizer,
    SynthesizedResponse,
    coerce_to_synthesized_response,
    sanitize_for_storage,
)
from agents.validator import SQLValidator
import importlib
import config as app_config
from logs.metrics_logger import QueryMetrics, get_metrics_logger

DEBUG_MODE = getattr(app_config, "DEBUG_MODE", True)
LOG_LEVEL = getattr(app_config, "LOG_LEVEL", "INFO")
MAX_RETRIES = getattr(app_config, "MAX_RETRIES", 2)
if not hasattr(app_config, "DEBUG_MODE") or not hasattr(app_config, "LOG_LEVEL") or not hasattr(app_config, "MAX_RETRIES"):
    app_config = importlib.reload(app_config)
    DEBUG_MODE = getattr(app_config, "DEBUG_MODE", True)
    LOG_LEVEL = getattr(app_config, "LOG_LEVEL", "INFO")
    MAX_RETRIES = getattr(app_config, "MAX_RETRIES", 2)

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
# Session Memory
# ---------------------------------------------------------------------------


class SessionMemory:
    """
    Simple in-memory store to maintain carry-forward context across queries.
    """
    def __init__(self):
        self._store: Dict[str, Dict[str, Any]] = {}

    def create_session(self, session_id: str) -> None:
        if session_id not in self._store:
            self._store[session_id] = {
                "last_question": None,
                "last_intent": None,
                "last_sql": None,
                "history": []
            }

    def get_session(self, session_id: str) -> Dict[str, Any]:
        return self._store.get(session_id, {})

    def update_session(self, session_id: str, data: Dict[str, Any]) -> None:
        if session_id not in self._store:
            self.create_session(session_id)
        self._store[session_id].update(data)
        if "history" in self._store[session_id]:
            # Simple append for history
            self._store[session_id]["history"].append(data)

    def clear_session(self, session_id: str) -> None:
        if session_id in self._store:
            del self._store[session_id]


# ---------------------------------------------------------------------------
# Orchestrator Class
# ---------------------------------------------------------------------------


class PromotionAnalyticsOrchestrator:
    """
    Main Orchestrator Agent.
    Coordinates all sub-agents to process a user question end-to-end.
    """

    def __init__(self):
        log.info("Initialising PromotionAnalyticsOrchestrator...")
        self.session_memory = SessionMemory()
        self.intent_classifier = IntentClassifier()
        self.query_grounder = QueryGroundingAgent()
        self.query_generator = QueryGenerationAgent()
        self.validator = SQLValidator()
        self.executor = ExecutionAgent()
        self.synthesizer = ResponseSynthesizer()
        log.info("Orchestrator ready.")

    def create_fallback_response(self, reason: str) -> SynthesizedResponse:
        """Create a safe fallback response when the pipeline must halt."""
        return SynthesizedResponse(
            answer_text="I couldn't generate a reliable analysis for this request." if "validation" in reason.lower() else "I need clarification before answering this question.",
            delta=None,
            pct_change=None,
            table=[],
            explanation=reason,
            coverage_flag=CoverageFlag(
                is_partial=True,
                missing_weeks=[],
                missing_regions=[],
                message="Pipeline halted."
            ),
            sql_shown="",
            suggestions=[],
            response_type="error",
            debug_info={},
        )

    def _build_debug_info(self, route: str, question: str, intent: Any = None, grounded_intent: Any = None, sql: str | None = None, df: Any = None, response: SynthesizedResponse | None = None) -> Dict[str, Any]:
        if not DEBUG_MODE:
            return {}

        debug_payload: Dict[str, Any] = {
            "route": route,
            "intent": intent.model_dump() if getattr(intent, "model_dump", None) else None,
            "grounded_metric": grounded_intent.model_dump() if getattr(grounded_intent, "model_dump", None) else None,
            "generated_sql": sql or "",
            "dataframe_head": [],
            "template_used": self._infer_template_name(question, getattr(grounded_intent, "topic", None) or getattr(intent, "topic", None)),
            "response_object": {},
        }

        if df is not None:
            try:
                debug_payload["dataframe_head"] = df.head(10).to_dict(orient="records")
            except Exception:
                debug_payload["dataframe_head"] = []

        if response is not None:
            response_payload = response.model_dump()
            response_payload.pop("debug_info", None)
            debug_payload["response_object"] = response_payload

        return debug_payload

    @staticmethod
    def _infer_template_name(question: str, topic: str | None = None) -> str:
        q = (question or "").lower()
        topic_key = (topic or "").lower()
        if topic_key == "promotion" or any(term in q for term in ["promo", "promotion", "improve", "improved", "baseline"]):
            return "promotion_effectiveness"
        if topic_key == "inventory" or any(term in q for term in ["inventory", "stock", "reduce", "reduction"]):
            return "inventory_movement"
        if topic_key == "trend_analysis" or any(term in q for term in ["trend", "growth", "over time", "change over time"]):
            return "trend_summary"
        if topic_key == "region_comparison" or any(term in q for term in ["compare", "comparison", "versus", "vs"]):
            return "region_comparison"
        if topic_key == "ranking" or any(term in q for term in ["highest", "lowest", "best", "worst", "rank", "top", "bottom"]):
            return "ranking_summary"
        return "fallback_generic"

    @staticmethod
    def _is_greeting(text: str) -> bool:
        lowered = (text or "").strip().lower()
        if not lowered:
            return False
        if any(char.isdigit() for char in lowered):
            return False
        greetings = ["hi", "hello", "hey", "good morning", "good evening", "good afternoon"]
        if lowered in greetings:
            return True
        if lowered.startswith(tuple(greetings)):
            return True
        if lowered.endswith(tuple(greetings)):
            return True
        if any(lowered == g for g in greetings):
            return True
        return False

    @staticmethod
    def _is_help(text: str) -> bool:
        lowered = (text or "").lower()
        help_terms = ["what can you do", "help", "show examples", "examples", "how do you work"]
        return any(term in lowered for term in help_terms)

    @staticmethod
    def _is_follow_up(text: str, session_data: Dict[str, Any]) -> bool:
        lowered = (text or "").lower()
        if not session_data.get("last_question") and not session_data.get("last_response"):
            return False
        follow_up_terms = [
            "why",
            "what about",
            "explain",
            "tell me more",
            "how much better",
            "and what about",
            "compare with",
            "show details",
            "show me details",
            "second place",
            "runner-up",
            "which campaign drove",
            "which region",
        ]
        return any(term in lowered for term in follow_up_terms)

    @staticmethod
    def _is_general_analysis(text: str) -> bool:
        lowered = (text or "").lower()
        general_terms = ["what trends", "give me a summary", "summary of the data", "what stands out", "analyze the business", "business performance", "overall performance", "dataset-level", "general business"]
        return any(term in lowered for term in general_terms)

    @staticmethod
    def _contains_analytics_keywords(text: str) -> bool:
        lowered = (text or "").lower()
        analytics_terms = [
            "revenue",
            "sales",
            "promotion",
            "campaign",
            "inventory",
            "region",
            "growth",
            "performance",
            "compare",
            "highest",
            "lowest",
            "best",
            "worst",
            "trend",
            "top",
            "bottom",
            "category",
            "sku",
        ]
        return any(term in lowered for term in analytics_terms)

    def _route_conversation(self, question: str, session_data: Dict[str, Any]) -> str:
        if self._is_greeting(question):
            return "greeting"
        if self._is_help(question):
            return "help"
        if self._is_follow_up(question, session_data):
            return "follow_up"
        if self._is_general_analysis(question):
            return "general_chat"
        if self._contains_analytics_keywords(question):
            return "analytics_query"
        return "analytics_query"

    def _build_conversational_response(self, route: str, question: str, session_data: Dict[str, Any]) -> SynthesizedResponse:
        prior_question = session_data.get("last_question") or ""
        prior_response = session_data.get("last_response") or {}
        previous_answer = prior_response.get("answer_text", "") if isinstance(prior_response, dict) else ""
        prior_entities = session_data.get("last_entities") or {}
        prior_promo = prior_entities.get("promo_id") or prior_entities.get("campaign") or prior_entities.get("promo_name")

        if route == "greeting":
            answer_text = "Hello! I’m your analytics copilot and I can help you understand promotions, inventory, regional performance, and campaign impact. Ask me anything in plain English."
            suggestions = [
                "Which campaign performed best?",
                "Compare North and South sales.",
                "How did revenue change over time?",
            ]
            explanation = "Greeting route — no SQL required."
        elif route == "help":
            answer_text = "I can help you with promotion performance, inventory movement, regional comparisons, campaign impact, and quick business summaries. I’ll answer in a conversational way and keep the context of your conversation."
            suggestions = [
                "Did PROMO_001 improve sales in South region?",
                "Show me a summary of the data.",
                "Which category generated the highest revenue?",
            ]
            explanation = "Help route — capabilities were explained without running SQL."
        elif route == "follow_up":
            if previous_answer:
                prior_question_lower = (prior_question or "").lower()
                if "campaign" in prior_question_lower or "promo" in prior_question_lower:
                    if prior_promo:
                        answer_text = (
                            f"The previous result points to {prior_promo} as the strongest campaign outcome so far, and I’d frame it as a clear business signal: {previous_answer}. "
                            "That means the leading campaign is outperforming the rest of the field, and I can help compare it with the runner-up or break down the difference."
                        )
                    else:
                        answer_text = (
                            f"The previous result points to the strongest campaign outcome so far, and I’d frame it as a clear business signal: {previous_answer}. "
                            "That means the leading campaign is outperforming the rest of the field, and I can help compare it with the runner-up or break down the difference."
                        )
                elif "category" in prior_question_lower:
                    answer_text = (
                        f"The previous result points to the strongest category outcome so far, and I’d frame it as a clear business signal: {previous_answer}. "
                        "That suggests the leading category is carrying the most momentum, and I can help you explore the gap or compare it with the next best category."
                    )
                elif "compare" in prior_question_lower or "south" in prior_question_lower or "north" in prior_question_lower:
                    answer_text = (
                        f"The earlier comparison context is still relevant: {previous_answer}. I’ll keep the conversation anchored to that North/South comparison and explain which campaign or region drove the result."
                    )
                else:
                    answer_text = (
                        f"The previous result points to the strongest outcome so far, and I’d frame it as a clear business signal: {previous_answer}. "
                        "If you want, I can also compare the leading option with the runner-up."
                    )
            else:
                answer_text = "I’m following up on your earlier analysis. The strongest result from the previous step appears to be the leading performer, and I can break down why it stood out."
            suggestions = [
                "Compare the top two campaigns.",
                "Which region contributed most?",
                "How did performance change over time?",
            ]
            explanation = "Follow-up route — reused session context to keep the conversation coherent."
        else:
            if previous_answer:
                answer_text = f"Based on the recent business context, the story looks promising: {previous_answer} The broader pattern suggests a few clear opportunities worth exploring in the strongest and weakest segments."
            else:
                answer_text = "The business picture suggests a few standout trends worth exploring, especially around the leading categories, regions, and campaigns. I can help you dig into those patterns in a more focused way."
            suggestions = [
                "Which campaign drove the strongest performance?",
                "Which region contributed the most?",
                "How did revenue change over time?",
            ]
            explanation = "General business-analysis route — no single-metric SQL path was needed."

        response_type = "chat"
        if route == "help":
            response_type = "help"
        elif route == "follow_up":
            response_type = "follow_up"

        return SynthesizedResponse(
            answer_text=answer_text,
            delta=None,
            pct_change=None,
            table=[],
            explanation=explanation,
            coverage_flag=CoverageFlag(
                is_partial=False,
                missing_weeks=[],
                missing_regions=[],
                message="Conversational response.",
            ),
            sql_shown="",
            suggestions=suggestions,
            response_type=response_type,
        )

    def handle(self, question: str, session_id: str = "default") -> SynthesizedResponse:
        """
        Main entrypoint.
        Executes the 10-step pipeline securely and logs output.
        """
        log.info("=" * 80)
        log.info("Orchestrator.handle() | session_id=%r", session_id)
        log.info("  Question: %r", question)

        _pipeline_start = time.perf_counter()
        _metrics_logger = get_metrics_logger()

        try:
            # STEP 1: Load session
            self.session_memory.create_session(session_id)
            session_data = self.session_memory.get_session(session_id)

            # Construct full conversation history
            history_text = ""
            if session_data.get("history"):
                for turn in session_data["history"][-3:]: # Get last 3 turns
                    q = turn.get("last_question")
                    ans_dict = turn.get("last_response", {})
                    ans_text = ans_dict.get("answer_text", "")
                    if q:
                        history_text += f"User: {q}\n"
                    if ans_text:
                        history_text += f"System: {ans_text}\n"

            route = self._route_conversation(question, session_data)
            if route != "analytics_query":
                log.info("  Conversation route: %s", route)
                response = self._build_conversational_response(route, question, session_data)
                response.debug_info = self._build_debug_info(route=route, question=question, response=response)
                self.session_memory.update_session(session_id, {
                    "last_question": question,
                    "last_intent": None,
                    "last_sql": None,
                    "last_response": response.model_dump(),
                    "last_route": route,
                })
                _metrics_logger.log_query_metrics(QueryMetrics(
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    question=question,
                    topic=route,
                    classification_confidence=1.0,
                    validation_passed=True,
                    retry_count=0,
                    execution_latency_ms=0.0,
                    row_count=0,
                    cache_hit=False,
                    response_generated=True,
                ))
                return response

            # STEP 2: Intent Classification
            intent = self.intent_classifier.classify(question, history=history_text)

            # STEP 3: Confidence Check
            if intent.confidence < 0.30:
                log.warning("  Low confidence (%.2f < 0.30) — aborting pipeline.", intent.confidence)
                _metrics_logger.log_query_metrics(QueryMetrics(
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    question=question,
                    topic=intent.topic,
                    classification_confidence=intent.confidence,
                    validation_passed=False,
                    retry_count=0,
                    execution_latency_ms=0.0,
                    row_count=0,
                    cache_hit=False,
                    response_generated=False,
                ))
                return self.create_fallback_response("The request is ambiguous.")

            # STEP 4: Grounding
            grounded_intent = self.query_grounder.ground(question, intent)

            # STEP 5 & 6: Generation & Validation Loop
            sql_result = self.query_generator.generate_sql(question, grounded_intent)
            sql = sql_result.sql

            retries = 0
            is_valid = False
            last_error = ""

            while retries <= MAX_RETRIES:
                log.info("  Validation attempt %d/%d", retries, MAX_RETRIES)
                val_result = self.validator.validate(sql, retries_used=retries)
                signal = self.validator.create_regeneration_signal(val_result)

                if val_result.is_valid:
                    is_valid = True
                    break
                else:
                    last_error = val_result.failure_reason or "Unknown validation error"
                    log.warning("  Validation failed: %s", last_error)
                    if signal.should_regenerate:
                        log.info("  Regenerating SQL...")
                        q_with_feedback = (
                            f"{question}\n\nPREVIOUS ATTEMPT FAILED WITH ERROR: "
                            f"{last_error}. Please fix the SQL."
                        )
                        sql_result = self.query_generator.generate_sql(
                            question=q_with_feedback,
                            grounded_intent=grounded_intent,
                        )
                        sql = sql_result.sql
                        retries += 1
                    else:
                        break

            # FAILURE RULE check
            if not is_valid:
                log.error("  SQL validation failed repeatedly. Aborting pipeline.")
                _metrics_logger.log_query_metrics(QueryMetrics(
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    question=question,
                    topic=intent.topic,
                    classification_confidence=intent.confidence,
                    validation_passed=False,
                    retry_count=retries,
                    execution_latency_ms=0.0,
                    row_count=0,
                    cache_hit=False,
                    response_generated=False,
                ))
                return self.create_fallback_response("SQL validation failed repeatedly.")

            # STEP 7: Execution
            execution_result = self.executor.execute(sql)

            # STEP 8: Response Synthesis
            response = self.synthesizer.synthesize(
                df=execution_result.dataframe,
                grounded_intent=grounded_intent,
                metadata=execution_result.metadata,
                sql=sql,
                question=question,
            )
            response = coerce_to_synthesized_response(response)
            response.debug_info = self._build_debug_info(
                route="analytics_query",
                question=question,
                intent=intent,
                grounded_intent=grounded_intent,
                sql=sql,
                df=execution_result.dataframe,
                response=response,
            )

            # STEP 9: Store Session Context
            response_payload = sanitize_for_storage(response.model_dump())
            entities = {
                "promo_id": getattr(grounded_intent, "promo_id", None),
                "region": grounded_intent.region,
                "category": grounded_intent.category,
                "sku": grounded_intent.sku,
            }
            self.session_memory.update_session(session_id, {
                "last_question": question,
                "last_intent": intent,
                "last_sql": sql,
                "last_response": response_payload,
                "last_route": "analytics_query",
                "last_entities": entities,
            })

            # STEP 10: Log metrics and return
            meta = execution_result.metadata
            log.info("  Pipeline succeeded. Latency: %.1f ms", meta.execution_time_ms)
            _metrics_logger.log_query_metrics(QueryMetrics(
                timestamp=datetime.now(timezone.utc).isoformat(),
                question=question,
                topic=intent.topic,
                classification_confidence=intent.confidence,
                validation_passed=True,
                retry_count=retries,
                execution_latency_ms=meta.execution_time_ms,
                row_count=meta.row_count,
                cache_hit=meta.cache_hit,
                response_generated=True,
            ))
            return response

        except Exception as exc:
            log.error("  Pipeline crashed: %s", exc)
            _elapsed = (time.perf_counter() - _pipeline_start) * 1000
            try:
                _metrics_logger.log_query_metrics(QueryMetrics(
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    question=question,
                    topic="unknown",
                    classification_confidence=0.0,
                    validation_passed=False,
                    retry_count=0,
                    execution_latency_ms=round(_elapsed, 2),
                    row_count=0,
                    cache_hit=False,
                    response_generated=False,
                ))
            except Exception:
                pass  # Never let logging crash the pipeline
            return self.create_fallback_response(f"An unexpected error occurred: {exc}")


# ---------------------------------------------------------------------------
# CLI Validation
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import json
    
    orchestrator = PromotionAnalyticsOrchestrator()
    print("\n" + "=" * 70)
    print("ORCHESTRATOR — VALIDATION RUN")
    print("=" * 70)
    
    questions = [
        "Did PROMO_001 improve sales in South region?",
        "What about North?"  # tests session carry-forward
    ]
    
    for q in questions:
        print(f"\nUser: {q}")
        res = orchestrator.handle(q, session_id="test_session")
        print(json.dumps(res.model_dump(), indent=2))
        print("-" * 50)
