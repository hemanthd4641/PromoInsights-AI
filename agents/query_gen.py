"""
agents/query_gen.py
--------------------
Phase 5 — Query Generation Agent.

Converts a GroundedIntent into valid, safe DuckDB SQL that ONLY queries
approved semantic-layer views. A post-generation safety layer validates
every SQL statement before it is returned to the caller.

Architecture:
  1. Build a structured prompt from: question + GroundedIntent +
     schema catalog + few-shot examples.
  2. Call Groq via LangChain to generate raw SQL text.
  3. Strip any markdown formatting from the response.
  4. Run the SQL through the whitelist safety layer.
  5. Return a typed SQLGenerationResult.

Usage:
    from agents.query_gen import QueryGenerationAgent
    from agents.query_grounding import GroundedIntent

    agent = QueryGenerationAgent()
    result = agent.generate_sql(question, grounded_intent)
    print(result.sql)
"""

import logging
import re
import sys
import time
from pathlib import Path
from typing import List, Optional, Set, Tuple

import duckdb
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Bootstrap: project root on sys.path + load .env
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
load_dotenv(PROJECT_ROOT / ".env")

from config import DUCKDB_PATH, GROQ_API_KEY, LOG_LEVEL, MAX_RETRIES, MODEL_NAME
from agents.query_grounding import GroundedIntent
from db.schema_catalog import SCHEMA_CATALOG, catalog_as_text

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
# Whitelist & Blocklist
# ---------------------------------------------------------------------------
WHITELISTED_VIEWS: Set[str] = {
    "vw_weekly_sales",
    "vw_weekly_inventory",
    "vw_promo_calendar",
}

BLOCKED_KEYWORDS: List[str] = [
    "sales_raw",
    "inventory_raw",
    "promotions_raw",
    "drop",
    "delete",
    "update",
    "insert",
    "alter",
    "truncate",
    "create",
    "attach",
    "detach",
    "pragma",
]

# ---------------------------------------------------------------------------
# Custom Exception
# ---------------------------------------------------------------------------


class SQLWhitelistViolationError(Exception):
    """
    Raised when generated SQL references a non-whitelisted object
    or contains a dangerous keyword.
    """

    def __init__(self, offending_object: str, offending_sql: str) -> None:
        self.offending_object = offending_object
        self.offending_sql = offending_sql
        super().__init__(
            f"SQL whitelist violation: '{offending_object}' is not allowed.\n"
            f"SQL: {offending_sql[:200]}"
        )


# ---------------------------------------------------------------------------
# Pydantic Output Model
# ---------------------------------------------------------------------------


class SQLGenerationResult(BaseModel):
    """
    Typed result returned by QueryGenerationAgent.generate_sql().

    Fields
    ------
    sql                   : The generated DuckDB SQL statement.
    used_views            : Whitelisted views referenced in the SQL.
    is_whitelist_compliant: True only if all referenced objects are whitelisted.
    """

    sql: str = Field(description="Generated DuckDB SQL statement.")
    used_views: List[str] = Field(
        default_factory=list,
        description="Whitelisted views referenced by the SQL.",
    )
    is_whitelist_compliant: bool = Field(
        description="True if SQL only references whitelisted views.",
    )


# ---------------------------------------------------------------------------
# Safety Layer — standalone functions
# ---------------------------------------------------------------------------


def extract_referenced_views(sql: str) -> List[str]:
    """
    Extract all table/view identifiers referenced in a SQL statement.

    Uses regex to find tokens after FROM, JOIN (all variants), INTO,
    UPDATE, and TABLE keywords.

    Args:
        sql: SQL string to scan.

    Returns:
        Deduplicated, lower-cased list of referenced identifiers.
    """
    # Match word-like tokens after SQL clauses that introduce table names
    pattern = re.compile(
        r"\b(?:FROM|JOIN|INNER\s+JOIN|LEFT\s+JOIN|RIGHT\s+JOIN|"
        r"FULL\s+JOIN|CROSS\s+JOIN|UPDATE|INTO|TABLE)\s+([a-zA-Z_][a-zA-Z0-9_.]*)",
        re.IGNORECASE,
    )
    found = pattern.findall(sql)
    # Deduplicate and normalise to lowercase
    return list({name.lower().strip() for name in found})


def validate_whitelist(sql: str) -> Tuple[bool, List[str]]:
    """
    Validate that a SQL string only references whitelisted views and
    contains no blocked keywords (DROP, DELETE, raw tables, etc.).

    Args:
        sql: SQL string to validate.

    Returns:
        (is_compliant, list_of_violations)
        is_compliant is True only when list_of_violations is empty.
    """
    violations: List[str] = []
    sql_lower = sql.lower()

    # 1. Block dangerous keywords (word-boundary match)
    for kw in BLOCKED_KEYWORDS:
        pattern = rf"\b{re.escape(kw)}\b"
        if re.search(pattern, sql_lower):
            violations.append(kw)

    # 2. Validate every referenced table/view against the whitelist
    referenced = extract_referenced_views(sql)
    for obj in referenced:
        if obj not in WHITELISTED_VIEWS:
            if obj not in violations:  # avoid duplicates
                violations.append(obj)

    return (len(violations) == 0, violations)


def validate_syntax(sql: str, db_path: Optional[str] = None) -> bool:
    """
    Validate SQL syntax by running EXPLAIN against a DuckDB connection.

    Args:
        sql    : SQL to validate.
        db_path: Path to DuckDB file (uses warehouse if available, else in-memory).

    Returns:
        True if DuckDB can parse the SQL, False otherwise.
    """
    path = str(PROJECT_ROOT / DUCKDB_PATH) if db_path is None else db_path
    try:
        con = duckdb.connect(path)
        con.execute(f"EXPLAIN {sql}")
        con.close()
        return True
    except Exception as exc:
        log.warning("DuckDB syntax validation failed: %s", exc)
        return False


# ---------------------------------------------------------------------------
# Prompt Builders
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a DuckDB SQL generation engine for a Promotion Analytics platform.

Your ONLY job is to output a single valid DuckDB SQL SELECT statement.

## STRICT RULES

1. Use ONLY the semantic-layer views listed below — never raw tables.
2. Never invent column names — use only columns listed in the schema.
3. Never return markdown, code fences, explanations, or comments.
4. Return ONLY executable DuckDB SQL — nothing else.
5. Prefer simple, readable SQL over complex SQL.
6. Use table aliases where helpful (e.g. `s` for vw_weekly_sales).
7. All string filters must be exact-match (e.g. region = 'South').
8. Always include a GROUP BY when using aggregate functions.
9. End the statement with a semicolon.

## ALLOWED VIEWS (WHITELIST)

- vw_weekly_sales
- vw_weekly_inventory
- vw_promo_calendar

## BLOCKED OBJECTS (NEVER USE)

- sales_raw
- inventory_raw
- promotions_raw
- Any other table or view not listed above

"""


def _build_schema_block() -> str:
    """Render the schema catalog as a text block for the prompt."""
    return f"## SCHEMA\n\n{catalog_as_text()}"


def _build_few_shot_block(examples: List[dict]) -> str:
    """Render up to 3 few-shot Q→SQL examples for the prompt."""
    if not examples:
        return ""
    lines = ["## FEW-SHOT EXAMPLES\n"]
    for i, ex in enumerate(examples[:3], 1):
        lines.append(f"### Example {i}")
        lines.append(f"Question: {ex.get('question', '')}")
        lines.append(f"SQL:\n{ex.get('sql', '')}\n")
    return "\n".join(lines)


def _build_intent_block(grounded: GroundedIntent) -> str:
    """Summarise the GroundedIntent for injection into the prompt."""
    parts = [
        "## GROUNDED INTENT",
        f"- Topic              : {grounded.topic}",
        f"- Metric Definition  : {grounded.metric_definition}",
        f"- Baseline Formula   : {grounded.baseline_formula or 'N/A'}",
        f"- Comparison Window  : {grounded.comparison_window or 'N/A'}",
        f"- Region             : {grounded.region or 'any'}",
        f"- SKU                : {grounded.sku or 'any'}",
        f"- Category           : {grounded.category or 'any'}",
        f"- Time Window        : {grounded.time_window or 'all weeks'}",
    ]
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Query Generation Agent
# ---------------------------------------------------------------------------


class QueryGenerationAgent:
    """
    LangChain + Groq powered SQL Generation Agent.

    Converts a GroundedIntent into a validated DuckDB SQL statement.
    Every generated SQL is passed through a safety layer before being
    returned to the caller.
    """

    def __init__(self) -> None:
        if not GROQ_API_KEY:
            raise EnvironmentError(
                "GROQ_API_KEY is not set. Add it to your .env file."
            )

        log.info("Initialising QueryGenerationAgent (model: %s)", MODEL_NAME)

        self._llm = ChatGroq(
            model=MODEL_NAME,
            api_key=GROQ_API_KEY,
            temperature=0.0,        # Deterministic SQL
            max_retries=MAX_RETRIES,
        )

        # Pre-render static prompt sections (schema doesn't change at runtime)
        self._schema_block = _build_schema_block()
        log.info("QueryGenerationAgent ready.")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_sql(
        self,
        question: str,
        grounded_intent: GroundedIntent,
    ) -> SQLGenerationResult:
        """
        Generate a validated DuckDB SQL statement from a GroundedIntent.

        Args:
            question       : Original natural-language question.
            grounded_intent: Fully resolved GroundedIntent from Phase 4.

        Returns:
            SQLGenerationResult with SQL, referenced views, and compliance flag.

        Raises:
            SQLWhitelistViolationError: If generated SQL fails safety validation.
            RuntimeError              : If the LLM call fails.
        """
        log.info("=" * 60)
        log.info("QueryGenerationAgent.generate_sql()")
        log.info("  Question : %r", question)
        log.info("  Topic    : %r", grounded_intent.topic)

        t_start = time.perf_counter()

        # ---- Build messages -----------------------------------------------
        few_shot_block = _build_few_shot_block(grounded_intent.few_shot_examples)
        intent_block = _build_intent_block(grounded_intent)

        human_content = (
            f"{self._schema_block}\n\n"
            f"{few_shot_block}\n\n"
            f"{intent_block}\n\n"
            f"## USER QUESTION\n{question}\n\n"
            "Generate the DuckDB SQL query. Return ONLY the SQL statement."
        )

        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=human_content),
        ]

        # ---- LLM call -------------------------------------------------------
        try:
            response = self._llm.invoke(messages)
            raw_sql: str = response.content
        except Exception as exc:
            log.exception("LLM call failed for question: %r", question)
            raise RuntimeError(
                f"QueryGenerationAgent failed for: {question!r}"
            ) from exc

        latency_ms = (time.perf_counter() - t_start) * 1000
        log.info("  LLM latency : %.0f ms", latency_ms)

        # ---- Clean up response (strip markdown fences) ----------------------
        sql = self._clean_sql(raw_sql)
        log.info("  Generated SQL:\n%s", sql)

        # ---- Safety validation ----------------------------------------------
        is_compliant, violations = validate_whitelist(sql)
        referenced = extract_referenced_views(sql)
        used_views = [v for v in referenced if v in WHITELISTED_VIEWS]

        log.info("  Referenced views : %s", referenced)
        log.info("  Whitelist result : %s", "PASS" if is_compliant else f"FAIL {violations}")

        if not is_compliant:
            for v in violations:
                log.error("  VIOLATION: %r in SQL", v)
            raise SQLWhitelistViolationError(
                offending_object=", ".join(violations),
                offending_sql=sql,
            )

        return SQLGenerationResult(
            sql=sql,
            used_views=used_views,
            is_whitelist_compliant=True,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _clean_sql(raw: str) -> str:
        """
        Strip markdown code fences and leading/trailing whitespace
        from the LLM response to get clean, executable SQL.
        """
        # Remove ```sql ... ``` or ``` ... ``` fences
        cleaned = re.sub(r"```(?:sql)?", "", raw, flags=re.IGNORECASE)
        cleaned = cleaned.strip()

        # If the model prefixed with "SQL:" or similar, remove it
        cleaned = re.sub(r"^(?:sql|query)\s*[:\-]\s*", "", cleaned, flags=re.IGNORECASE)

        return cleaned.strip()


# ---------------------------------------------------------------------------
# CLI validation
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import json
    from agents.intent_classifier import IntentClassifier
    from agents.query_grounding import QueryGroundingAgent

    SAMPLE_QUESTIONS = [
        "Did PROMO_001 improve sales in South region?",
        "Did inventory reduce in West region?",
        "Compare North and South sales.",
        "Which campaign performed best?",
        "Which category generated highest revenue?",
    ]

    clf = IntentClassifier()
    grounder = QueryGroundingAgent()
    gen = QueryGenerationAgent()

    print("\n" + "=" * 70)
    print("QUERY GENERATION AGENT — VALIDATION RUN")
    print("=" * 70)

    for i, q in enumerate(SAMPLE_QUESTIONS, 1):
        print(f"\n[Query {i}] {q}")
        intent = clf.classify(q)
        grounded = grounder.ground(q, intent)
        result = gen.generate_sql(q, grounded)
        print(json.dumps(result.model_dump(), indent=2))
        print("-" * 60)
