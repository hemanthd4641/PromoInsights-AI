import pandas as pd
import pytest

from agents.intent_classifier import Intent, IntentClassifier
from agents.query_gen import QueryGenerationAgent, SQLGenerationResult
from agents.query_grounding import GroundedIntent, QueryGroundingAgent
from agents.synthesizer import ResponseSynthesizer
from tests.mock_llm import patch_groq

patch_groq()


def test_intent_classifier_supports_extended_topics():
    classifier = IntentClassifier()
    intent = classifier.classify("Which SKU ranked first by revenue last month?")

    assert intent.topic == "ranking"
    assert intent.time_window in {"last month", "last 4 weeks"}
    assert intent.confidence >= 0.5


def test_grounding_resolves_promo_improvement_window():
    agent = QueryGroundingAgent()
    intent = Intent(topic="promotion", confidence=0.95, region=None, sku=None, category=None, time_window=None)

    grounded = agent.ground("Was there any improvement in revenue after PROMO_003?", intent)

    assert grounded.metric_definition.lower().startswith("promotion") or "revenue" in grounded.metric_definition.lower()
    assert grounded.comparison_window is not None
    assert "baseline" in grounded.comparison_window.lower() or "promo" in grounded.comparison_window.lower()


def test_generate_sql_fallback_avoids_placeholder_sql(monkeypatch):
    agent = QueryGenerationAgent()
    agent._call_llm = lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("boom"))

    grounded = GroundedIntent(
        topic="promotion",
        confidence=0.9,
        metric_definition="promotion_effectiveness",
        baseline_formula="promo_vs_baseline",
        comparison_window="promo window vs baseline",
        region="South",
    )

    result = agent.generate_sql("Did PROMO_001 improve sales in South region?", grounded)

    assert isinstance(result, SQLGenerationResult)
    assert result.is_whitelist_compliant is True
    assert "select 1" not in result.sql.lower()
    assert "vw_weekly_sales" in result.sql.lower()
    assert "promo001" in result.sql.lower() or "promo_001" in result.sql.lower()


def test_synthesizer_falls_back_to_data_summary(monkeypatch):
    import agents.synthesizer as synthesizer_module
    import config

    config.GROQ_API_KEY = "test-key"
    synthesizer_module.GROQ_API_KEY = "test-key"

    synth = ResponseSynthesizer()
    synth._call_llm = lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("boom"))

    df = pd.DataFrame(
        [
            {"region": "North", "revenue": 1200.0},
            {"region": "South", "revenue": 800.0},
        ]
    )
    grounded = GroundedIntent(
        topic="ranking",
        confidence=0.9,
        metric_definition="ranked_performance",
        baseline_formula=None,
        comparison_window="overall period",
    )

    answer = synth.generate_answer_text(df, grounded, "Which region ranked highest?")

    assert "highest" in answer.lower() or "north" in answer.lower()
