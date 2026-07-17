"""
db/cache.py
------------
Phase 7 -- Rollup Cache Layer.

Provides an in-memory cache for commonly requested DuckDB rollups.
Rollups are pre-computed at startup and can be refreshed on demand
or automatically after a configurable TTL (default: 24 hours).

Cached rollups:
  - weekly_sales_region       : Weekly revenue/units by region
  - weekly_inventory_region   : Weekly avg stock level by region
  - weekly_category_revenue   : Weekly revenue by category
  - promo_revenue_ranking     : Total revenue per promo_id
  - region_category_sales     : Revenue by region x category

Usage:
    from db.cache import RollupCache

    cache = RollupCache()
    cache.initialize_cache()

    df = cache.get_cached_result("weekly_sales_region")
    cache.set_cached_result("my_key", my_df)
    cache.refresh_cache()
"""

import logging
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional

import duckdb
import pandas as pd

# ---------------------------------------------------------------------------
# Bootstrap: project root on sys.path
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import DUCKDB_PATH, LOG_LEVEL

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
# Cache TTL
# ---------------------------------------------------------------------------
CACHE_TTL_HOURS: int = 24

# ---------------------------------------------------------------------------
# Pre-defined rollup queries (only whitelisted views)
# ---------------------------------------------------------------------------
ROLLUP_QUERIES: Dict[str, str] = {
    "weekly_sales_region": """
        SELECT
            region,
            week,
            SUM(units_sold)  AS total_units,
            SUM(revenue)     AS total_revenue
        FROM vw_weekly_sales
        GROUP BY region, week
        ORDER BY region, week
    """,
    "weekly_inventory_region": """
        SELECT
            region,
            week,
            AVG(stock_level) AS avg_stock
        FROM vw_weekly_inventory
        GROUP BY region, week
        ORDER BY region, week
    """,
    "weekly_category_revenue": """
        SELECT
            category,
            week,
            SUM(revenue) AS total_revenue
        FROM vw_weekly_sales
        GROUP BY category, week
        ORDER BY category, week
    """,
    "promo_revenue_ranking": """
        SELECT
            promo_id,
            SUM(revenue)    AS total_revenue,
            SUM(units_sold) AS total_units
        FROM vw_weekly_sales
        GROUP BY promo_id
        ORDER BY total_revenue DESC
    """,
    "region_category_sales": """
        SELECT
            region,
            category,
            SUM(revenue)    AS total_revenue,
            SUM(units_sold) AS total_units
        FROM vw_weekly_sales
        GROUP BY region, category
        ORDER BY region, category
    """,
}

# ---------------------------------------------------------------------------
# Cache Entry Type
# ---------------------------------------------------------------------------

class _CacheEntry:
    """Internal container for a cached DataFrame + metadata."""

    __slots__ = ("dataframe", "timestamp", "hit_count")

    def __init__(self, dataframe: pd.DataFrame) -> None:
        self.dataframe: pd.DataFrame = dataframe
        self.timestamp: datetime = datetime.now()
        self.hit_count: int = 0

    def is_expired(self) -> bool:
        return datetime.now() > self.timestamp + timedelta(hours=CACHE_TTL_HOURS)


# ---------------------------------------------------------------------------
# RollupCache
# ---------------------------------------------------------------------------


class RollupCache:
    """
    In-memory cache for pre-computed DuckDB rollups.

    The cache is keyed by arbitrary strings. Built-in rollups are seeded
    via initialize_cache() / refresh_cache(). External code may store
    custom results via set_cached_result().
    """

    def __init__(self, db_path: Optional[str] = None) -> None:
        self._db_path: str = db_path or str(PROJECT_ROOT / DUCKDB_PATH)
        self._store: Dict[str, _CacheEntry] = {}
        self._initialized: bool = False
        log.info("RollupCache created (db: %s, TTL: %dh)", self._db_path, CACHE_TTL_HOURS)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _connect(self) -> duckdb.DuckDBPyConnection:
        return duckdb.connect(self._db_path, read_only=True)

    def _execute_rollup(self, key: str, sql: str) -> Optional[pd.DataFrame]:
        """Execute a single rollup query and return a DataFrame (or None on error)."""
        try:
            con = self._connect()
            df: pd.DataFrame = con.execute(sql).df()
            con.close()
            log.info("  Cached rollup '%s' -- %d rows", key, len(df))
            return df
        except Exception as exc:
            log.warning("  Rollup '%s' failed: %s", key, exc)
            return None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def initialize_cache(self) -> None:
        """
        Pre-compute all built-in rollups and store them in the cache.
        Safe to call multiple times — will not re-compute if already initialised
        and TTL has not expired.
        """
        if self._initialized:
            log.info("RollupCache already initialised — skipping.")
            return

        log.info("Initialising RollupCache (%d rollups)...", len(ROLLUP_QUERIES))
        t0 = time.perf_counter()

        for key, sql in ROLLUP_QUERIES.items():
            df = self._execute_rollup(key, sql)
            if df is not None:
                self._store[key] = _CacheEntry(df)

        elapsed = (time.perf_counter() - t0) * 1000
        log.info(
            "RollupCache initialised: %d/%d entries in %.0f ms",
            len(self._store), len(ROLLUP_QUERIES), elapsed,
        )
        self._initialized = True

    def refresh_cache(self) -> None:
        """
        Force-refresh all built-in rollups regardless of TTL.
        Useful for daily scheduled refreshes or manual triggers.
        """
        log.info("Refreshing RollupCache...")
        self._initialized = False
        # Preserve any custom (non-rollup) entries
        custom_keys = set(self._store.keys()) - set(ROLLUP_QUERIES.keys())
        for key in custom_keys:
            pass  # keep custom entries as-is

        for key, sql in ROLLUP_QUERIES.items():
            df = self._execute_rollup(key, sql)
            if df is not None:
                self._store[key] = _CacheEntry(df)

        self._initialized = True
        log.info("RollupCache refreshed: %d entries", len(self._store))

    def get_cached_result(self, cache_key: str) -> Optional[pd.DataFrame]:
        """
        Return a cached DataFrame for the given key, or None if not found / expired.

        Args:
            cache_key: Lookup key (e.g. 'weekly_sales_region').

        Returns:
            Cached DataFrame or None.
        """
        entry = self._store.get(cache_key)
        if entry is None:
            log.debug("Cache MISS: '%s'", cache_key)
            return None

        if entry.is_expired():
            log.info("Cache EXPIRED: '%s' -- removing", cache_key)
            del self._store[cache_key]
            return None

        entry.hit_count += 1
        log.info("Cache HIT:  '%s' (hits=%d, rows=%d)", cache_key, entry.hit_count, len(entry.dataframe))
        return entry.dataframe.copy()

    def set_cached_result(self, cache_key: str, dataframe: pd.DataFrame) -> None:
        """
        Store an arbitrary DataFrame under the given key.

        Args:
            cache_key: Storage key.
            dataframe : DataFrame to cache.
        """
        self._store[cache_key] = _CacheEntry(dataframe.copy())
        log.info("Cache SET:  '%s' (%d rows)", cache_key, len(dataframe))

    def cache_stats(self) -> Dict[str, object]:
        """Return a summary of current cache state for diagnostics."""
        return {
            "total_entries": len(self._store),
            "initialized": self._initialized,
            "entries": {
                k: {
                    "rows": len(e.dataframe),
                    "cached_at": e.timestamp.isoformat(),
                    "hit_count": e.hit_count,
                    "expired": e.is_expired(),
                }
                for k, e in self._store.items()
            },
        }


# ---------------------------------------------------------------------------
# Module-level singleton (shared across agents)
# ---------------------------------------------------------------------------
_global_cache: Optional[RollupCache] = None


def get_global_cache() -> RollupCache:
    """Return (and lazily initialize) the module-level RollupCache singleton."""
    global _global_cache
    if _global_cache is None:
        _global_cache = RollupCache()
        _global_cache.initialize_cache()
    return _global_cache
