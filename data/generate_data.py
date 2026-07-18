"""
data/generate_data.py
---------------------
Phase 1 — Synthetic Data Generation & DuckDB Loading Pipeline.

Responsibilities:
  1. Generate realistic promotion, sales, and inventory data using Faker.
  2. Create (or recreate) the DuckDB warehouse at db/warehouse.duckdb.
  3. Drop and reload: sales_raw, inventory_raw, promotions_raw.
  4. Run semantic-layer views from db/semantic_layer.sql.
  5. Print row counts and validate vw_weekly_sales.

Run:
    python data/generate_data.py
"""

import logging
import os
import random
import sys
from pathlib import Path
from typing import List, Dict, Any

import duckdb
import pandas as pd
from faker import Faker

# ---------------------------------------------------------------------------
# Bootstrap: make project root importable regardless of cwd
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import (
    DUCKDB_PATH,
    ROW_COUNT_MIN,
    ROW_COUNT_MAX,
    LOG_LEVEL,
)

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
# Constants — domain vocabulary
# ---------------------------------------------------------------------------
REGIONS: List[str] = ["North", "South", "East", "West"]
CATEGORIES: List[str] = ["Electronics", "Groceries", "Fashion", "Home", "Sports"]
SKUS: List[str] = [f"SKU{str(i).zfill(3)}" for i in range(1, 51)]  # SKU001–SKU050
WEEKS: List[int] = list(range(1, 53))  # weeks 1–52

SALES_COUNT: int = max(500, ROW_COUNT_MIN)
INVENTORY_COUNT: int = max(500, ROW_COUNT_MIN)
PROMO_COUNT: int = 50

fake = Faker()
Faker.seed(42)
random.seed(42)

# ---------------------------------------------------------------------------
# Data Generators
# ---------------------------------------------------------------------------

def generate_promotions(n: int = PROMO_COUNT) -> pd.DataFrame:
    """Generate realistic promotion records."""
    log.info("Generating %d promotion records...", n)
    records: List[Dict[str, Any]] = []

    for i in range(1, n + 1):
        start_week = random.randint(1, 45)
        end_week = start_week + random.randint(1, 6)
        end_week = min(end_week, 52)

        records.append(
            {
                "promo_id": f"PROMO_{str(i).zfill(3)}",
                "promo_name": f"PROMO_{str(i).zfill(3)}",
                "region": random.choice(REGIONS),
                "category": random.choice(CATEGORIES),
                "start_week": start_week,
                "end_week": end_week,
                "discount_pct": round(random.uniform(5.0, 40.0), 2),
            }
        )

    df = pd.DataFrame(records)
    log.info("  ✓ promotions_raw: %d rows", len(df))
    return df


def generate_sales(
    promo_ids: List[str], n: int = SALES_COUNT
) -> pd.DataFrame:
    """Generate realistic sales records linked to existing promo_ids."""
    log.info("Generating %d sales records...", n)
    records: List[Dict[str, Any]] = []

    for i in range(1, n + 1):
        sku = random.choice(SKUS)
        category = random.choice(CATEGORIES)
        units = random.randint(1, 500)
        unit_price = round(random.uniform(5.0, 500.0), 2)

        records.append(
            {
                "sale_id": f"SALE{str(i).zfill(5)}",
                "region": random.choice(REGIONS),
                "week": random.choice(WEEKS),
                "sku": sku,
                "category": category,
                "promo_id": random.choice(promo_ids),
                "units_sold": units,
                "revenue": round(units * unit_price, 2),
            }
        )

    df = pd.DataFrame(records)
    log.info("  ✓ sales_raw: %d rows", len(df))
    return df


def generate_inventory(n: int = INVENTORY_COUNT) -> pd.DataFrame:
    """Generate realistic inventory records."""
    log.info("Generating %d inventory records...", n)
    records: List[Dict[str, Any]] = []

    for i in range(1, n + 1):
        records.append(
            {
                "inventory_id": f"INV{str(i).zfill(5)}",
                "region": random.choice(REGIONS),
                "week": random.choice(WEEKS),
                "sku": random.choice(SKUS),
                "category": random.choice(CATEGORIES),
                "stock_level": random.randint(0, 10_000),
            }
        )

    df = pd.DataFrame(records)
    log.info("  ✓ inventory_raw: %d rows", len(df))
    return df


# ---------------------------------------------------------------------------
# DuckDB Loader
# ---------------------------------------------------------------------------

def load_to_duckdb(
    con: duckdb.DuckDBPyConnection,
    df: pd.DataFrame,
    table_name: str,
) -> None:
    """Drop (if exists) and recreate a table from a DataFrame."""
    log.info("Loading table: %s ...", table_name)
    con.execute(f"DROP TABLE IF EXISTS {table_name}")
    con.execute(
        f"CREATE TABLE {table_name} AS SELECT * FROM df"
    )
    count = con.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
    log.info("  ✓ %s loaded — %d rows", table_name, count)


def apply_semantic_layer(con: duckdb.DuckDBPyConnection) -> None:
    """Execute semantic_layer.sql to create/replace all views."""
    sql_path = PROJECT_ROOT / "db" / "semantic_layer.sql"
    if not sql_path.exists():
        raise FileNotFoundError(f"Semantic layer SQL not found: {sql_path}")

    log.info("Applying semantic layer from: %s", sql_path)
    sql = sql_path.read_text(encoding="utf-8")

    # Execute each statement separately (split on ';')
    statements = [s.strip() for s in sql.split(";") if s.strip()]
    for stmt in statements:
        con.execute(stmt)

    log.info("  ✓ Semantic layer views created successfully")


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate(con: duckdb.DuckDBPyConnection) -> None:
    """Run a quick validation query and print results."""
    log.info("=" * 60)
    log.info("VALIDATION — SELECT * FROM vw_weekly_sales LIMIT 5")
    log.info("=" * 60)

    result = con.execute("SELECT * FROM vw_weekly_sales LIMIT 5").df()
    print("\n" + result.to_string(index=False))
    print()

    # Row counts across all views
    views = ["vw_weekly_sales", "vw_weekly_inventory", "vw_promo_calendar"]
    log.info("View row counts:")
    for view in views:
        count = con.execute(f"SELECT COUNT(*) FROM {view}").fetchone()[0]
        log.info("  %-30s %d rows", view, count)


# ---------------------------------------------------------------------------
# Main Entry Point
# ---------------------------------------------------------------------------

def main() -> None:
    log.info("=" * 60)
    log.info("Phase 1 — Data Generation & DuckDB Loading")
    log.info("=" * 60)

    # Resolve DuckDB path relative to project root
    db_path = PROJECT_ROOT / DUCKDB_PATH
    db_path.parent.mkdir(parents=True, exist_ok=True)
    log.info("DuckDB path: %s", db_path)

    # Step 1: Generate data
    df_promos = generate_promotions(PROMO_COUNT)
    df_sales = generate_sales(df_promos["promo_id"].tolist(), SALES_COUNT)
    df_inventory = generate_inventory(INVENTORY_COUNT)

    # Step 2: Connect and load into DuckDB
    log.info("Connecting to DuckDB at: %s", db_path)
    con = duckdb.connect(str(db_path))

    try:
        load_to_duckdb(con, df_promos, "promotions_raw")
        load_to_duckdb(con, df_sales, "sales_raw")
        load_to_duckdb(con, df_inventory, "inventory_raw")

        # Step 3: Apply semantic layer views
        apply_semantic_layer(con)

        # Step 4: Validate
        validate(con)

        log.info("=" * 60)
        log.info("Phase 1 COMPLETE — warehouse.duckdb is ready.")
        log.info("=" * 60)

    except Exception as exc:
        log.exception("Pipeline failed: %s", exc)
        raise
    finally:
        con.close()


if __name__ == "__main__":
    main()
