import pandas as pd

from agents.executor import ExecutionMetadata
from agents.query_grounding import GroundedIntent
from agents.synthesizer import ResponseSynthesizer
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
