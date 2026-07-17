import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from tests.mock_llm import patch_groq
patch_groq()

from agents.orchestrator import PromotionAnalyticsOrchestrator

def main():
    print("Testing unseen queries...")
    orchestrator = PromotionAnalyticsOrchestrator()
    queries = [
        "Which promotion had the biggest impact on Electronics sales?",
        "Show me the worst performing region.",
        "Why?",
        "Compare inventory movement across all regions during Q2.",
        "List all campaigns that ran in June."
    ]

    session_id = "test-session-unseen"
    for idx, q in enumerate(queries, 1):
        print(f"\n[{idx}] Question: {q}")
        try:
            resp = orchestrator.handle(session_id, q)
            print(f"SQL:\n{resp.sql_shown}\n")
            print(f"Summary:\n{resp.answer_text}\n")
            print(f"Data:\n{resp.table}\n")
        except Exception as e:
            print(f"Error handling query: {e}")

if __name__ == "__main__":
    main()
