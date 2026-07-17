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
from pathlib import Path
from typing import Any, Dict

# ---------------------------------------------------------------------------
# Bootstrap project root
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from agents.executor import ExecutionAgent
from agents.intent_classifier import IntentClassifier
from agents.query_gen import QueryGenerationAgent
from agents.query_grounding import QueryGroundingAgent
from agents.synthesizer import CoverageFlag, ResponseSynthesizer, SynthesizedResponse
from agents.validator import SQLValidator
from config import LOG_LEVEL, MAX_RETRIES

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
            sql_shown=""
        )

    def handle(self, question: str, session_id: str = "default") -> SynthesizedResponse:
        """
        Main entrypoint.
        Executes the 10-step pipeline securely and logs output.
        """
        log.info("=" * 80)
        log.info("Orchestrator.handle() | session_id=%r", session_id)
        log.info("  Question: %r", question)

        try:
            # STEP 1: Load session
            self.session_memory.create_session(session_id)
            session_data = self.session_memory.get_session(session_id)
            
            # Simple carry-forward context injection
            context_prefix = ""
            if session_data.get("last_intent"):
                last_intent = session_data["last_intent"]
                context_prefix = (
                    f"Previous context: topic={last_intent.topic}, "
                    f"region={last_intent.region}, category={last_intent.category}. "
                )
            contextualised_question = f"{context_prefix}Current Question: {question}"

            # STEP 2: Intent Classification
            intent = self.intent_classifier.classify(contextualised_question)
            
            # STEP 3: Confidence Check
            if intent.confidence < 0.70:
                log.warning("  Low confidence (%.2f < 0.70) — aborting pipeline.", intent.confidence)
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
                        # Pass failure reason to generation agent as feedback via the question
                        q_with_feedback = f"{question}\n\nPREVIOUS ATTEMPT FAILED WITH ERROR: {last_error}. Please fix the SQL."
                        sql_result = self.query_generator.generate_sql(
                            question=q_with_feedback,
                            grounded_intent=grounded_intent
                        )
                        sql = sql_result.sql
                        retries += 1
                    else:
                        break

            # FAILURE RULE check
            if not is_valid:
                log.error("  SQL validation failed repeatedly. Aborting pipeline.")
                return self.create_fallback_response("SQL validation failed repeatedly.")

            # STEP 7: Execution
            execution_result = self.executor.execute(sql)

            # STEP 8: Response Synthesis
            response = self.synthesizer.synthesize(
                df=execution_result.dataframe,
                grounded_intent=grounded_intent,
                metadata=execution_result.metadata,
                sql=sql
            )

            # STEP 9: Store Session Context
            self.session_memory.update_session(session_id, {
                "last_question": question,
                "last_intent": intent,
                "last_sql": sql,
                "last_response": response.model_dump()
            })

            # STEP 10: Return Response
            log.info("  Pipeline succeeded. Latency: %.1f ms", execution_result.metadata.execution_time_ms)
            return response

        except Exception as exc:
            log.error("  Pipeline crashed: %s", exc)
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
