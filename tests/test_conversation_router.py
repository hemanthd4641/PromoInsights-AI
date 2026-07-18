import pandas as pd

from agents.orchestrator import PromotionAnalyticsOrchestrator
from agents.query_grounding import GroundedIntent
from agents.synthesizer import ExecutionMetadata, ResponseSynthesizer, SynthesizedResponse


def test_greeting_route_returns_friendly_response():
    orch = PromotionAnalyticsOrchestrator()
    response = orch.handle("hi there", session_id="router-greeting")

    assert isinstance(response, SynthesizedResponse)
    assert response.answer_text
    assert response.response_type == "chat"
    assert "hello" in response.answer_text.lower() or "hi" in response.answer_text.lower()
    assert not response.sql_shown


def test_help_route_returns_capabilities_without_sql():
    orch = PromotionAnalyticsOrchestrator()
    response = orch.handle("what can you do", session_id="router-help")

    assert isinstance(response, SynthesizedResponse)
    assert response.answer_text
    assert response.response_type == "help"
    assert any(term in response.answer_text.lower() for term in ["help", "promotion", "inventory", "regional", "campaign", "business"])
    assert not response.sql_shown


def test_follow_up_uses_previous_context():
    orch = PromotionAnalyticsOrchestrator()
    first = orch.handle("Which campaign performed best?", session_id="router-follow")
    second = orch.handle("Why?", session_id="router-follow")

    assert isinstance(first, SynthesizedResponse)
    assert isinstance(second, SynthesizedResponse)
    assert second.answer_text
    assert second.response_type == "follow_up"
    assert "campaign" in second.answer_text.lower() or "best" in second.answer_text.lower() or "perform" in second.answer_text.lower()


def test_general_analysis_returns_business_summary():
    orch = PromotionAnalyticsOrchestrator()
    response = orch.handle("What trends do you see?", session_id="router-general")

    assert isinstance(response, SynthesizedResponse)
    assert response.answer_text
    assert response.response_type == "chat"
    assert "trend" in response.answer_text.lower() or "business" in response.answer_text.lower() or "performance" in response.answer_text.lower()
    assert not response.sql_shown


def test_ranking_questions_stay_on_analytics_path():
    orch = PromotionAnalyticsOrchestrator()

    campaign_response = orch.handle("Which campaign performed best?", session_id="router-ranking-campaign")
    category_response = orch.handle("Which category generated highest revenue?", session_id="router-ranking-category")

    assert isinstance(campaign_response, SynthesizedResponse)
    assert campaign_response.response_type == "analytics"
    assert campaign_response.sql_shown

    assert isinstance(category_response, SynthesizedResponse)
    assert category_response.response_type == "analytics"
    assert category_response.sql_shown


def test_analytics_keywords_never_route_to_greeting():
    orch = PromotionAnalyticsOrchestrator()

    for question in [
        "Which campaign performed best?",
        "Which category generated highest revenue?",
        "Did PROMO_001 improve sales in South region?",
        "Compare North and South sales.",
        "Did inventory reduce in West region?",
    ]:
        route = orch._route_conversation(question, {})
        assert route == "analytics_query", question


def test_promotion_synthesis_uses_promotion_language():
    synth = ResponseSynthesizer()
    df = pd.DataFrame([
        {"promo_id": "PROMO_001", "region": "South", "revenue": 120000, "units_sold": 5200, "pct_change": 18.4},
    ])
    intent = GroundedIntent(topic="promotion", confidence=0.95, metric_definition="sales_change", comparison_window="four-week baseline", region="South")
    response = synth.synthesize(df, intent, ExecutionMetadata(row_count=1), "SELECT 1", question="Did PROMO_001 improve sales in South region?")

    assert "PROMO_001" in response.answer_text
    assert "18.4%" in response.answer_text or "percent" in response.answer_text.lower()
    assert "top result" not in response.answer_text.lower()
    assert "ranked highest" not in response.answer_text.lower()


def test_region_comparison_synthesis_uses_requested_regions_only():
    synth = ResponseSynthesizer()
    df = pd.DataFrame([
        {"region": "North", "total_revenue": 7690000},
        {"region": "South", "total_revenue": 7430000},
        {"region": "East", "total_revenue": 5000000},
    ])
    intent = GroundedIntent(topic="region_comparison", confidence=0.95, metric_definition="revenue", comparison_window="overall period", region=None)
    response = synth.synthesize(df, intent, ExecutionMetadata(row_count=3), "SELECT 1", question="Compare North and South sales.")

    assert "North Revenue" in response.answer_text
    assert "South Revenue" in response.answer_text
    assert "Difference" in response.answer_text
    assert "East" not in response.answer_text


def test_topic_specific_paths_for_promotion_inventory_trend_and_ranking():
    orch = PromotionAnalyticsOrchestrator()

    promo_response = orch.handle("Did PROMO_001 improve sales?", session_id="router-topic-promo")
    inventory_response = orch.handle("Did inventory reduce in West region?", session_id="router-topic-inventory")
    trend_response = orch.handle("How did revenue change over time?", session_id="router-topic-trend")
    ranking_response = orch.handle("Which category generated highest revenue?", session_id="router-topic-ranking")

    assert isinstance(promo_response, SynthesizedResponse)
    assert "promotion effectiveness" in promo_response.answer_text.lower()
    assert "baseline comparison" in promo_response.answer_text.lower()

    assert isinstance(inventory_response, SynthesizedResponse)
    assert "inventory movement" in inventory_response.answer_text.lower()
    assert "inventory summary" in inventory_response.answer_text.lower()

    assert isinstance(trend_response, SynthesizedResponse)
    assert "weekly trend" in trend_response.answer_text.lower()
    assert "growth rate" in trend_response.answer_text.lower()

    assert isinstance(ranking_response, SynthesizedResponse)
    assert "top category" in ranking_response.answer_text.lower()
    assert "actual revenue" in ranking_response.answer_text.lower()


def test_debug_trace_surfaces_internal_state_for_analytics_queries():
    orch = PromotionAnalyticsOrchestrator()
    response = orch.handle("Compare North and South sales.", session_id="router-debug")

    assert isinstance(response, SynthesizedResponse)
    assert response.debug_info
    debug = response.debug_info

    assert debug["route"] == "analytics_query"
    assert debug["intent"]["topic"]
    assert debug["grounded_metric"]
    assert debug["generated_sql"]
    assert debug["dataframe_head"]
    assert debug["template_used"]
    assert debug["response_object"]["answer_text"]


def test_follow_up_reuses_previous_comparison_context():
    orch = PromotionAnalyticsOrchestrator()
    first = orch.handle("Compare North and South sales", session_id="router-follow-context")
    second = orch.handle("Which campaign drove that result?", session_id="router-follow-context")

    assert isinstance(first, SynthesizedResponse)
    assert isinstance(second, SynthesizedResponse)
    assert "north" in second.answer_text.lower() or "south" in second.answer_text.lower()
    assert "campaign" in second.answer_text.lower() or "result" in second.answer_text.lower()
