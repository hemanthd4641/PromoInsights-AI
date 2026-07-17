"""
agents/executor.py
-------------------
Phase 7 -- Execution & Aggregation Agent.

Sits between SQL Validation (Phase 6) and Response Synthesis.
Responsibilities:
  1. Execute validated SQL against DuckDB.
  2. Return results as a pandas DataFrame.
  3. Compute delta (current - previous) when a numeric + week column is present.
  4. Compute pct_change when a numeric + week column is present.
  5. Check the rollup cache before executing (cache-first strategy).
  6. Track row count and execution latency.
  7. Return a fully structured ExecutionResult.

Design guarantee: this module NEVER raises to its caller.
All errors are captured and returned as structured exceptions with metadata.

Usage:
    from agents.executor import ExecutionAgent

    agent = ExecutionAgent()
    result = agent.execute(sql)
    print(result.metadata.model_dump_json(indent=2))
    print(result.dataframe.head())
"""

import logging
import sys
import time
from pathlib import Path
from typing import Any, Optional

import duckdb
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

# ---------------------------------------------------------------------------
# Bootstrap: project root on sys.path
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import DUCKDB_PATH, LOG_LEVEL
from db.cache import RollupCache, get_global_cache

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
# Cache key detection patterns
# (ordered — first match wins)
# ---------------------------------------------------------------------------
_CACHE_PATTERNS = [
    # (substring_triggers, cache_key)
    (["vw_weekly_sales", "region", "group by region"],      "weekly_sales_region"),
    (["vw_weekly_inventory", "region", "group by region"],  "weekly_inventory_region"),
    (["vw_weekly_sales", "category", "group by category"],  "weekly_category_revenue"),
    (["vw_weekly_sales", "promo_id", "group by promo_id"],  "promo_revenue_ranking"),
    (["vw_weekly_sales", "region", "category"],             "region_category_sales"),
]

# Numeric columns that represent "metrics" eligible for delta/pct_change
_METRIC_COLS = {
    "revenue", "total_revenue", "units_sold", "total_units",
    "stock_level", "avg_stock", "total_units_sold",
}

# Column used as the time axis for delta computation
_TIME_COL = "week"


# ---------------------------------------------------------------------------
# Pydantic Models
# ---------------------------------------------------------------------------


class ExecutionMetadata(BaseModel):
    """
    Metadata captured during a single execute() call.

    Fields
    ------
    row_count          : Number of rows returned by the query.
    execution_time_ms  : Total time from cache-check to return (ms).
    cache_hit          : True if result was served from the rollup cache.
    delta_computed     : True if a 'delta' column was added to the DataFrame.
    pct_change_computed: True if a 'pct_change' column was added.
    """

    row_count: int = Field(default=0, ge=0)
    execution_time_ms: float = Field(default=0.0, ge=0.0)
    cache_hit: bool = Field(default=False)
    delta_computed: bool = Field(default=False)
    pct_change_computed: bool = Field(default=False)


class ExecutionResult(BaseModel):
    """
    Full result returned by ExecutionAgent.execute().

    Fields
    ------
    dataframe: pandas DataFrame containing the query results.
    metadata : ExecutionMetadata with timing, cache, and enrichment info.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    dataframe: Any = Field(description="Query result as a pandas DataFrame.")
    metadata: ExecutionMetadata = Field(description="Execution metadata.")


# ---------------------------------------------------------------------------
# Execution Agent
# ---------------------------------------------------------------------------


class ExecutionAgent:
    """
    Execution & Aggregation Agent — executes validated DuckDB SQL and
    enriches the result with delta / pct_change analytics.

    Cache-first strategy:
      1. Check RollupCache for a matching pre-computed result.
      2. On cache miss, execute SQL against DuckDB.
      3. Enrich the DataFrame with delta/pct_change if applicable.
      4. Return ExecutionResult.
    """

    def __init__(
        self,
        db_path: Optional[str] = None,
        cache: Optional[RollupCache] = None,
    ) -> None:
        self._db_path = db_path or str(PROJECT_ROOT / DUCKDB_PATH)
        self._cache = cache or get_global_cache()
        log.info("ExecutionAgent initialised (db: %s)", self._db_path)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _connect(self) -> duckdb.DuckDBPyConnection:
        return duckdb.connect(self._db_path, read_only=True)

    # ------------------------------------------------------------------
    # Function 1 — execute_sql
    # ------------------------------------------------------------------

    def execute_sql(self, sql: str) -> pd.DataFrame:
        """
        Execute a SQL string against DuckDB and return a pandas DataFrame.

        Args:
            sql: Validated SQL string.

        Returns:
            pandas DataFrame with query results.

        Raises:
            RuntimeError: If the DuckDB execution fails.
        """
        log.info("Executing SQL: %s", sql[:120].strip())
        try:
            con = self._connect()
            df: pd.DataFrame = con.execute(sql).df()
            con.close()
            log.info("  Executed -- %d rows returned", len(df))
            return df
        except duckdb.Error as exc:
            log.error("DuckDB execution error: %s", exc)
            raise RuntimeError(f"SQL execution failed: {exc}") from exc
        except Exception as exc:
            log.error("Unexpected execution error: %s", exc)
            raise RuntimeError(f"Unexpected error during SQL execution: {exc}") from exc

    # ------------------------------------------------------------------
    # Function 2 — compute_delta
    # ------------------------------------------------------------------

    def compute_delta(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute a 'delta' column (current - previous row) for the first
        numeric metric column found, sorted by the time column ('week').

        Only adds the column if:
          - 'delta' does not already exist.
          - A recognised metric column is present.
          - The time column ('week') is present (used for sorting).

        Args:
            df: Input DataFrame.

        Returns:
            DataFrame with an optional 'delta' column appended.
        """
        if "delta" in df.columns:
            log.debug("  compute_delta: 'delta' already present — skipping")
            return df

        metric_col = self._find_metric_col(df)
        if metric_col is None:
            log.debug("  compute_delta: no metric column found — skipping")
            return df

        # Sort by time axis if available
        if _TIME_COL in df.columns:
            df = df.sort_values(_TIME_COL).reset_index(drop=True)

        df["delta"] = df[metric_col].diff()
        log.info("  compute_delta: '%s' -> delta added", metric_col)
        return df

    # ------------------------------------------------------------------
    # Function 3 — compute_pct_change
    # ------------------------------------------------------------------

    def compute_pct_change(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute a 'pct_change' column ((current - previous) / previous * 100)
        for the first numeric metric column found.

        Only adds the column if:
          - 'pct_change' does not already exist.
          - A recognised metric column is present.

        Divide-by-zero results in NaN (safe by pandas default).

        Args:
            df: Input DataFrame.

        Returns:
            DataFrame with an optional 'pct_change' column appended.
        """
        if "pct_change" in df.columns:
            log.debug("  compute_pct_change: 'pct_change' already present — skipping")
            return df

        metric_col = self._find_metric_col(df)
        if metric_col is None:
            log.debug("  compute_pct_change: no metric column found — skipping")
            return df

        # Sort by time axis if available
        if _TIME_COL in df.columns:
            df = df.sort_values(_TIME_COL).reset_index(drop=True)

        # pandas pct_change handles divide-by-zero as NaN automatically
        df["pct_change"] = df[metric_col].pct_change() * 100
        log.info("  compute_pct_change: '%s' -> pct_change added", metric_col)
        return df

    # ------------------------------------------------------------------
    # Function 4 — detect_cache_candidate
    # ------------------------------------------------------------------

    def detect_cache_candidate(self, sql: str) -> Optional[str]:
        """
        Detect if a SQL query matches a known rollup pattern.

        Matching is done by checking whether ALL trigger substrings
        in a pattern are present in the lower-cased SQL.

        Args:
            sql: SQL string to inspect.

        Returns:
            A cache key string, or None if no pattern matches.
        """
        sql_lower = sql.lower()
        for triggers, key in _CACHE_PATTERNS:
            if all(t.lower() in sql_lower for t in triggers):
                log.info("  Cache candidate detected: '%s'", key)
                return key
        return None

    # ------------------------------------------------------------------
    # Function 5 — execute (full pipeline)
    # ------------------------------------------------------------------

    def execute(self, sql: str) -> ExecutionResult:
        """
        Full execution pipeline:
          1. Check rollup cache.
          2. Execute SQL on cache miss.
          3. Enrich with delta / pct_change.
          4. Track row count and latency.
          5. Return ExecutionResult.

        NEVER raises — all errors are captured and surfaced in a
        RuntimeError with a descriptive message if something is truly
        unrecoverable (the caller should handle this gracefully).

        Args:
            sql: Validated SQL string (passed from Phase 6).

        Returns:
            ExecutionResult with DataFrame and ExecutionMetadata.
        """
        log.info("=" * 60)
        log.info("ExecutionAgent.execute()")
        log.info("  SQL: %s", sql[:120].strip())

        t_start = time.perf_counter()
        cache_hit = False
        delta_computed = False
        pct_change_computed = False

        # ---- Step 1-2: Cache check ----------------------------------------
        cache_key = self.detect_cache_candidate(sql)
        df: Optional[pd.DataFrame] = None

        if cache_key:
            df = self._cache.get_cached_result(cache_key)
            if df is not None:
                cache_hit = True
                log.info("  Cache HIT: '%s' (%d rows)", cache_key, len(df))

        # ---- Step 3: Execute SQL on miss ------------------------------------
        if df is None:
            log.info("  Cache MISS — executing SQL against DuckDB")
            try:
                df = self.execute_sql(sql)
            except RuntimeError as exc:
                # Return a well-structured empty result rather than crashing
                elapsed = (time.perf_counter() - t_start) * 1000
                log.error("  Execution failed: %s", exc)
                return ExecutionResult(
                    dataframe=pd.DataFrame(),
                    metadata=ExecutionMetadata(
                        row_count=0,
                        execution_time_ms=round(elapsed, 2),
                        cache_hit=False,
                        delta_computed=False,
                        pct_change_computed=False,
                    ),
                )

        # ---- Step 4: Enrich with delta / pct_change -------------------------
        before_cols = set(df.columns)

        df = self.compute_delta(df)
        if "delta" in df.columns and "delta" not in before_cols:
            delta_computed = True

        df = self.compute_pct_change(df)
        if "pct_change" in df.columns and "pct_change" not in before_cols:
            pct_change_computed = True

        # ---- Step 5: Capture metrics ----------------------------------------
        elapsed_ms = (time.perf_counter() - t_start) * 1000
        row_count = len(df)

        log.info(
            "  Done: rows=%d | latency=%.1f ms | cache=%s | delta=%s | pct_change=%s",
            row_count, elapsed_ms, cache_hit, delta_computed, pct_change_computed,
        )

        # ---- Step 6: Return ExecutionResult ---------------------------------
        return ExecutionResult(
            dataframe=df,
            metadata=ExecutionMetadata(
                row_count=row_count,
                execution_time_ms=round(elapsed_ms, 2),
                cache_hit=cache_hit,
                delta_computed=delta_computed,
                pct_change_computed=pct_change_computed,
            ),
        )

    # ------------------------------------------------------------------
    # Private utility
    # ------------------------------------------------------------------

    @staticmethod
    def _find_metric_col(df: pd.DataFrame) -> Optional[str]:
        """
        Return the first numeric column in the DataFrame that is recognised
        as a business metric. Returns None if no match found.
        """
        for col in df.columns:
            if col.lower() in _METRIC_COLS and pd.api.types.is_numeric_dtype(df[col]):
                return col
        # Fallback: any numeric column except 'week', 'sale_id', etc.
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]) and col.lower() not in {
                "week", "sale_id", "inventory_id", "promo_id"
            }:
                return col
        return None


# ---------------------------------------------------------------------------
# CLI validation
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import json

    agent = ExecutionAgent()

    SAMPLE_QUERIES = [
        ("SELECT * FROM vw_weekly_sales LIMIT 10", "Basic SELECT"),
        ("SELECT region, SUM(revenue) AS total_revenue FROM vw_weekly_sales GROUP BY region ORDER BY region",
         "Aggregation by region"),
        ("SELECT week, SUM(revenue) AS revenue FROM vw_weekly_sales GROUP BY week ORDER BY week LIMIT 5",
         "Weekly revenue series"),
    ]

    print("\n" + "=" * 70)
    print("EXECUTION AGENT -- VALIDATION RUN")
    print("=" * 70)

    for sql, label in SAMPLE_QUERIES:
        print(f"\n[{label}]")
        result = agent.execute(sql)
        print(f"  Metadata: {result.metadata.model_dump_json()}")
        print(f"  DataFrame head:\n{result.dataframe.head(3).to_string(index=False)}")
        print("-" * 60)
