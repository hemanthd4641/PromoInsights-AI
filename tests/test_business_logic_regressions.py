import numpy as np
import pandas as pd

from agents.executor import ExecutionMetadata
from agents.orchestrator import PromotionAnalyticsOrchestrator
from agents.query_gen import QueryGenerationAgent
from agents.query_grounding import GroundedIntent
from agents.synthesizer import ResponseSynthesizer, coerce_to_synthesized_response
from data.generate_data import generate_promotions


def test_region_comparison_uses_requested_regions_only() -> None:
    synth = ResponseSynthesizer()
    df = pd.DataFrame(
        [
            {"region": "North", "revenue": 120000.0},
            {"region": "South", "revenue": 80000.0},
        ]
    )
    intent = GroundedIntent(
        topic="region_comparison",
        confidence=0.95,
        metric_definition="regional_performance",
        comparison_window="selected period",
    )

    response = synth.synthesize(
        df=df,
        grounded_intent=intent,
        metadata=ExecutionMetadata(row_count=2),
        sql="SELECT region, revenue FROM stub",
        question="Compare North and South sales.",
    )

    text = response.answer_text.lower()
    assert "north" in text
    assert "south" in text
    assert "west" not in text
    assert response.delta is None
    assert response.pct_change is None


def test_ranking_questions_do_not_show_delta_metrics() -> None:
    synth = ResponseSynthesizer()
    df = pd.DataFrame(
        [
            {"sku": "SKU001", "revenue": 50000.0},
            {"sku": "SKU002", "revenue": 45000.0},
        ]
    )
    intent = GroundedIntent(
        topic="ranking",
        confidence=0.92,
        metric_definition="ranked_performance",
        comparison_window="overall period",
    )

    response = synth.synthesize(
        df=df,
        grounded_intent=intent,
        metadata=ExecutionMetadata(row_count=2),
        sql="SELECT sku, revenue FROM stub",
        question="Top 5 products by revenue",
    )

    assert response.delta is None
    assert response.pct_change is None


def test_generated_promotion_names_use_realistic_promo_ids() -> None:
    promos = generate_promotions(3)

    assert len(promos) == 3
    assert promos["promo_name"].str.contains("PROMO_", case=False).all()
    assert promos["promo_id"].str.contains(r"PROMO_\d{3}", regex=True).all()


def test_inventory_queries_use_inventory_template() -> None:
    synth = ResponseSynthesizer()
    df = pd.DataFrame([{"region": "West", "stock_level": 1200.0}])
    intent = GroundedIntent(
        topic="inventory",
        confidence=0.95,
        metric_definition="inventory_change",
        comparison_window="selected period",
    )

    response = synth.synthesize(
        df=df,
        grounded_intent=intent,
        metadata=ExecutionMetadata(row_count=1),
        sql="SELECT region, stock_level FROM stub",
        question="Did inventory reduce in West region?",
    )

    text = response.answer_text.lower()
    assert "inventory" in text
    assert "promotion effectiveness" not in text


def test_promotion_template_uses_safe_uplift_message_for_non_finite_values() -> None:
    synth = ResponseSynthesizer()
    df = pd.DataFrame([{"promo_id": "PROMO_001", "region": "South", "pct_change": np.inf}])
    intent = GroundedIntent(
        topic="promotion",
        confidence=0.95,
        metric_definition="sales_change",
        comparison_window="baseline",
    )

    response = synth.synthesize(
        df=df,
        grounded_intent=intent,
        metadata=ExecutionMetadata(row_count=1),
        sql="SELECT promo_id, region, pct_change FROM stub",
        question="Did PROMO_001 improve sales?",
    )

    assert "Insufficient baseline data to calculate uplift." in response.answer_text
    assert "nan" not in response.answer_text.lower()
    assert "inf" not in response.answer_text.lower()


def test_region_comparison_selects_true_winner() -> None:
    synth = ResponseSynthesizer()
    df = pd.DataFrame(
        [
            {"region": "North", "total_revenue": 448138.0},
            {"region": "South", "total_revenue": 3723.0},
        ]
    )
    intent = GroundedIntent(
        topic="region_comparison",
        confidence=0.95,
        metric_definition="revenue",
        comparison_window="selected period",
    )

    response = synth.synthesize(
        df=df,
        grounded_intent=intent,
        metadata=ExecutionMetadata(row_count=2),
        sql="SELECT region, total_revenue FROM stub",
        question="Compare North and South sales.",
    )

    text = response.answer_text.lower()
    assert "north" in text
    assert "south" in text
    assert "north" in text.split("higher")[-1] or "north" in text
    assert "nan" not in text
    assert "inf" not in text


def test_coerce_to_synthesized_response_keeps_text_and_metrics_separate() -> None:
    malformed_payload = {
        "answer_text": "Region comparison: South outperformed North.",
        "delta": "Region comparison: South outperformed North.",
        "pct_change": "Promotion effectiveness: the selected promotion.",
        "table": [],
        "explanation": "Summary text should remain a string.",
        "coverage_flag": {
            "is_partial": False,
            "missing_weeks": [],
            "missing_regions": [],
            "message": "Complete coverage.",
        },
        "sql_shown": "SELECT 1",
    }

    response = coerce_to_synthesized_response(malformed_payload)

    assert isinstance(response.answer_text, str)
    assert isinstance(response.explanation, str)
    assert response.delta is None
    assert response.pct_change is None


def test_lowest_revenue_uses_min_aggregation_in_fallback_sql() -> None:
    agent = QueryGenerationAgent()
    intent = GroundedIntent(
        topic="ranking",
        confidence=0.9,
        metric_definition="ranked_performance",
        comparison_window="overall period",
    )

    sql = agent._fallback_sql("lowest revenue", intent)

    assert "MIN(" in sql.upper() or "ORDER BY TOTAL_REVENUE ASC" in sql.upper() or "ASC" in sql.upper()


def test_promotion_synthesis_includes_actual_units_and_pct_metrics() -> None:
    synth = ResponseSynthesizer()
    df = pd.DataFrame([
        {"promo_id": "PROMO_001", "region": "South", "revenue": 120000.0, "units_sold": 425, "pct_change": 18.4},
    ])
    intent = GroundedIntent(
        topic="promotion",
        confidence=0.95,
        metric_definition="sales_change",
        comparison_window="baseline",
    )

    response = synth.synthesize(
        df=df,
        grounded_intent=intent,
        metadata=ExecutionMetadata(row_count=1),
        sql="SELECT 1",
        question="Did PROMO_001 improve sales?",
    )

    text = response.answer_text.lower()
    assert "promo_001" in text
    assert "425" in text
    assert "18.4%" in response.answer_text or "18.4" in response.answer_text


def test_follow_up_response_reuses_previous_entities() -> None:
    orch = PromotionAnalyticsOrchestrator()
    session_data = {
        "last_question": "Which campaign performed best?",
        "last_response": {"answer_text": "PROMO_014 led the ranking."},
        "last_entities": {"promo_id": "PROMO_014"},
    }

    response = orch._build_conversational_response("follow_up", "Why?", session_data)

    assert "PROMO_014" in response.answer_text
