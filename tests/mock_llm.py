"""
tests/mock_llm.py
------------------
Mock implementation of ChatGroq LLM for fast, offline, and rate-limit-free unit tests.
Patches ChatGroq class to intercept calls and return canned Intents/SQL responses.
"""

import pydantic
from unittest.mock import patch
from langchain_groq import ChatGroq
from langchain_core.runnables import RunnableLambda
from langchain_core.messages import AIMessage
from agents.intent_classifier import Intent

# Canned Intent responses mapping (lowercase question -> Dict of intent fields)
MOCK_INTENTS = {
    "did promo_001 improve sales in south region?": {
        "topic": "promotion", "region": "South", "sku": None, "category": None, "time_window": None, "confidence": 0.96
    },
    "how effective was promo_002 compared to baseline?": {
        "topic": "promotion", "region": "South", "sku": None, "category": None, "time_window": None, "confidence": 0.95
    },
    "which region performed best during promo_002?": {
        "topic": "region_comparison", "region": None, "sku": None, "category": None, "time_window": None, "confidence": 0.92
    },
    "compare north and south sales during promo_003.": {
        "topic": "region_comparison", "region": None, "sku": None, "category": None, "time_window": None, "confidence": 0.91
    },
    "which category reacted best to the summer campaign?": {
        "topic": "campaign_impact", "region": None, "sku": None, "category": None, "time_window": None, "confidence": 0.89
    },
    "did inventory reduce for electronics products in west region?": {
        "topic": "inventory", "region": "West", "sku": None, "category": "Electronics", "time_window": None, "confidence": 0.94
    },
    "did inventory reduce for electronics in west region?": {
        "topic": "inventory", "region": "West", "sku": None, "category": "Electronics", "time_window": None, "confidence": 0.94
    },
    "which sku generated the highest revenue last quarter?": {
        "topic": "campaign_impact", "region": None, "sku": None, "category": None, "time_window": None, "confidence": 0.88
    },
    "did inventory reduce in west region?": {
        "topic": "inventory", "region": "West", "sku": None, "category": None, "time_window": None, "confidence": 0.92
    },
    "compare north and south sales.": {
        "topic": "region_comparison", "region": "North", "sku": None, "category": None, "time_window": None, "confidence": 0.90
    },
    "compare north and south sales": {
        "topic": "region_comparison", "region": "North", "sku": None, "category": None, "time_window": None, "confidence": 0.90
    },
    "which campaign performed best?": {
        "topic": "campaign_impact", "region": None, "sku": None, "category": None, "time_window": None, "confidence": 0.88
    },
    "which campaign performed best": {
        "topic": "campaign_impact", "region": None, "sku": None, "category": None, "time_window": None, "confidence": 0.88
    },
    "which category generated highest revenue?": {
        "topic": "campaign_impact", "region": None, "sku": None, "category": None, "time_window": None, "confidence": 0.87
    },
    "which category generated highest revenue": {
        "topic": "campaign_impact", "region": None, "sku": None, "category": None, "time_window": None, "confidence": 0.87
    },
    "which products have the highest stock levels?": {
        "topic": "inventory", "region": None, "sku": None, "category": None, "time_window": None, "confidence": 0.93
    },
    "which region generated the highest revenue?": {
        "topic": "region_comparison", "region": None, "sku": None, "category": None, "time_window": None, "confidence": 0.91
    },
    "tell me something interesting.": {
        "topic": "promotion", "region": None, "sku": None, "category": None, "time_window": None, "confidence": 0.40
    },
    "tell me something interesting": {
        "topic": "promotion", "region": None, "sku": None, "category": None, "time_window": None, "confidence": 0.40
    },
    "analyze everything.": {
        "topic": "promotion", "region": None, "sku": None, "category": None, "time_window": None, "confidence": 0.35
    },
    "analyze everything": {
        "topic": "promotion", "region": None, "sku": None, "category": None, "time_window": None, "confidence": 0.35
    },
    "how did south perform?": {
        "topic": "promotion", "region": "South", "sku": None, "category": None, "time_window": None, "confidence": 0.95
    },
    "how did south perform": {
        "topic": "promotion", "region": "South", "sku": None, "category": None, "time_window": None, "confidence": 0.95
    },
    "what about north?": {
        "topic": "promotion", "region": "North", "sku": None, "category": None, "time_window": None, "confidence": 0.95
    },
    "what about north": {
        "topic": "promotion", "region": "North", "sku": None, "category": None, "time_window": None, "confidence": 0.95
    }
}


# Canned SQL responses mapping (keywords inside user prompt -> SQL string)
# NOTE: DB uses PROMO001 (no underscore) — all SQL must match this format.
MOCK_SQL = {
    "promo_001": """
SELECT 
    SUM(CASE WHEN s.promo_id = 'PROMO001' THEN s.units_sold ELSE 0 END) AS promo_units_sold,
    SUM(CASE WHEN s.promo_id IS NULL THEN s.units_sold ELSE 0 END) AS baseline_units_sold,
    CASE 
        WHEN SUM(CASE WHEN s.promo_id IS NULL THEN s.units_sold ELSE 0 END) = 0 THEN NULL
        ELSE ((SUM(CASE WHEN s.promo_id = 'PROMO001' THEN s.units_sold ELSE 0 END) - 
               SUM(CASE WHEN s.promo_id IS NULL THEN s.units_sold ELSE 0 END)) / 
              SUM(CASE WHEN s.promo_id IS NULL THEN s.units_sold ELSE 0 END)) * 100
    END AS sales_lift
FROM 
    vw_weekly_sales s
WHERE 
    s.region = 'South';
""",
    "promo_002": """
SELECT 
    SUM(CASE WHEN s.promo_id = 'PROMO002' THEN s.revenue ELSE 0 END) AS promo_revenue,
    SUM(CASE WHEN s.promo_id IS NULL THEN s.revenue ELSE 0 END) AS baseline_revenue,
    CASE
        WHEN SUM(CASE WHEN s.promo_id IS NULL THEN s.revenue ELSE 0 END) = 0 THEN NULL
        ELSE (SUM(CASE WHEN s.promo_id = 'PROMO002' THEN s.revenue ELSE 0 END) - 
              SUM(CASE WHEN s.promo_id IS NULL THEN s.revenue ELSE 0 END)) / 
             SUM(CASE WHEN s.promo_id IS NULL THEN s.revenue ELSE 0 END) * 100
    END AS effectiveness
FROM 
    vw_weekly_sales s
WHERE 
    s.region = 'South';
""",
    "reduce for electronics": """
SELECT 
    i.week, 
    i.region, 
    i.sku, 
    i.category, 
    i.stock_level AS current_stock, 
    LAG(i.stock_level) OVER (PARTITION BY i.sku, i.category ORDER BY i.week) AS previous_stock,
    ((LAG(i.stock_level) OVER (PARTITION BY i.sku, i.category ORDER BY i.week)) - i.stock_level) / LAG(i.stock_level) OVER (PARTITION BY i.sku, i.category ORDER BY i.week) * 100 AS stock_reduction_pct
FROM 
    vw_weekly_inventory i
WHERE 
    i.region = 'West' AND i.category = 'Electronics'
ORDER BY 
    i.week, 
    i.sku, 
    i.category;
""",
    "reduce in west": """
SELECT 
    i.week, 
    i.region, 
    i.sku, 
    i.category, 
    i.stock_level AS current_stock, 
    LAG(i.stock_level) OVER (PARTITION BY i.sku, i.category ORDER BY i.week) AS previous_stock,
    ((LAG(i.stock_level) OVER (PARTITION BY i.sku, i.category ORDER BY i.week)) - i.stock_level) / LAG(i.stock_level) OVER (PARTITION BY i.sku, i.category ORDER BY i.week) * 100 AS stock_reduction_pct
FROM 
    vw_weekly_inventory i
WHERE 
    i.region = 'West'
ORDER BY 
    i.week, 
    i.sku, 
    i.category;
""",
    "highest stock levels": """
SELECT sku, category, region, MAX(stock_level) AS max_stock FROM vw_weekly_inventory GROUP BY sku, category, region ORDER BY max_stock DESC;
""",
    "compare north and south sales": """
SELECT region, SUM(revenue) AS total_revenue, SUM(units_sold) AS total_units FROM vw_weekly_sales WHERE region IN ('North', 'South') GROUP BY region;
""",
    "highest revenue region": """
SELECT region, SUM(revenue) AS total_revenue FROM vw_weekly_sales GROUP BY region ORDER BY total_revenue DESC LIMIT 1;
""",
    "campaign performed best": """
SELECT 
    p.promo_name, 
    SUM(s.revenue) AS total_revenue 
FROM 
    vw_weekly_sales s
JOIN 
    vw_promo_calendar p ON s.promo_id = p.promo_id
GROUP BY 
    p.promo_name
ORDER BY 
    total_revenue DESC 
LIMIT 1;
""",
    "south perform": """
SELECT s.region, SUM(s.revenue) AS total_revenue FROM vw_weekly_sales s WHERE s.region = 'South' GROUP BY s.region;
""",
    "what about north": """
SELECT s.region, SUM(s.revenue) AS total_revenue FROM vw_weekly_sales s WHERE s.region = 'North' GROUP BY s.region;
""",
    "category generated highest revenue": """
SELECT category, SUM(revenue) AS total_revenue FROM vw_weekly_sales GROUP BY category ORDER BY total_revenue DESC LIMIT 1;
""",
    "category reacted best": """
SELECT category, SUM(revenue) AS total_revenue FROM vw_weekly_sales WHERE promo_id = 'PROMO001' GROUP BY category ORDER BY total_revenue DESC LIMIT 1;
""",
    "sku generated the highest revenue": """
SELECT sku, SUM(revenue) AS total_revenue FROM vw_weekly_sales GROUP BY sku ORDER BY total_revenue DESC LIMIT 1;
""",
    "unknown promotion": """
SELECT SUM(revenue) AS total_revenue FROM vw_weekly_sales WHERE promo_id = 'PROMO_UNKNOWN_XYZ';
"""
}

# Keep original methods in case we need live calls or selective patching
_orig_init = ChatGroq.__init__
_orig_with_structured_output = ChatGroq.with_structured_output
_orig_invoke = ChatGroq.invoke


def mock_with_structured_output(self, schema, **kwargs):
    def _invoke(input_dict, **config):
        question_text = ""
        # Inspect input_dict type to extract the human question
        if hasattr(input_dict, "to_messages"):
            messages = input_dict.to_messages()
            if messages:
                question_text = str(messages[-1].content)
        elif isinstance(input_dict, list):
            if input_dict:
                msg = input_dict[-1]
                if hasattr(msg, "content"):
                    question_text = str(msg.content)
                else:
                    question_text = str(msg)
        elif isinstance(input_dict, dict):
            question_text = input_dict.get("question", "")
        else:
            question_text = str(input_dict)
        
        q = question_text.lower().strip()
        matched_key = None
        for key in MOCK_INTENTS:
            if key in q:
                matched_key = key
                break
        
        if not matched_key:
            # default fallback
            matched_key = "tell me something interesting."
            
        data = MOCK_INTENTS[matched_key]
        if issubclass(schema, pydantic.BaseModel):
            return schema(**data)
        return data

    return RunnableLambda(_invoke)


def mock_invoke(self, messages, stop=None, **kwargs):
    # Find user message content
    user_msg = ""
    for msg in messages:
        if hasattr(msg, "content"):
            user_msg += "\n" + str(msg.content)
        elif isinstance(msg, dict):
            user_msg += "\n" + str(msg.get("content", ""))
        elif isinstance(msg, tuple):
            user_msg += "\n" + str(msg[1])

    # Check if this is a synthesizer request
    is_synthesizer = False
    for msg in messages:
        content = ""
        if hasattr(msg, "content"):
            content = str(msg.content)
        elif isinstance(msg, dict):
            content = str(msg.get("content", ""))
        elif isinstance(msg, tuple):
            content = str(msg[1])
            
        if "Executive Synthesizer" in content:
            is_synthesizer = True
            break

    if is_synthesizer:
        # Provide a realistic mock summary based on the question
        if "Compare North and South" in user_msg:
            return AIMessage(content="North generated more revenue than South. Difference: ₹15,000.")
        if "PROMO_001" in user_msg:
            return AIMessage(content="Yes, PROMO_001 improved sales in the South region.")
        if "Which campaign performed best" in user_msg:
            return AIMessage(content="Campaign X generated the highest revenue.")
        if "highest revenue" in user_msg:
            return AIMessage(content="The highest revenue was generated by Electronics category.")
        return AIMessage(content="Analysis complete. Please review the supporting data table.")

    user_msg_lower = user_msg.lower().strip()
    
    # If the message contains ## user question, extract the exact question to avoid matches with schema
    if "## user question" in user_msg_lower:
        parts = user_msg_lower.split("## user question")
        question_part = parts[-1].strip()
        question_lines = [line.strip() for line in question_part.split("\n") if line.strip()]
        if question_lines:
            user_msg_lower = question_lines[0]

    matched_sql = None
    for key, sql_str in MOCK_SQL.items():
        if key in user_msg_lower:
            matched_sql = sql_str
            break

    if not matched_sql:
        # Default fallback
        matched_sql = "SELECT 1;"

    return AIMessage(content=matched_sql)


def patch_groq():
    """Apply the ChatGroq monkeypatch globally."""
    ChatGroq.with_structured_output = mock_with_structured_output
    ChatGroq.invoke = mock_invoke
    print("[MOCK] ChatGroq successfully patched with mock responses.")


def unpatch_groq():
    """Restore the original ChatGroq methods."""
    ChatGroq.with_structured_output = _orig_with_structured_output
    ChatGroq.invoke = _orig_invoke
    print("[MOCK] ChatGroq successfully restored to live API.")
