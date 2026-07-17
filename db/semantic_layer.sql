-- =============================================================================
-- db/semantic_layer.sql
-- Phase 1 — Semantic Layer (Curated Business Views)
--
-- These views sit on top of the raw tables and expose a clean,
-- business-friendly grain to analysts and the SQL-generation agent.
-- No raw-table implementation details leak through these views.
-- =============================================================================

-- ---------------------------------------------------------------------------
-- vw_weekly_sales
-- Purpose : Weekly sales metrics at region / week / sku / category grain.
-- Source  : sales_raw
-- ---------------------------------------------------------------------------
CREATE OR REPLACE VIEW vw_weekly_sales AS
SELECT
    sale_id,
    region,
    week,
    sku,
    category,
    promo_id,
    units_sold,
    revenue
FROM sales_raw;

-- ---------------------------------------------------------------------------
-- vw_weekly_inventory
-- Purpose : Weekly inventory snapshot at region / week / sku / category grain.
-- Source  : inventory_raw
-- ---------------------------------------------------------------------------
CREATE OR REPLACE VIEW vw_weekly_inventory AS
SELECT
    inventory_id,
    region,
    week,
    sku,
    category,
    stock_level
FROM inventory_raw;

-- ---------------------------------------------------------------------------
-- vw_promo_calendar
-- Purpose : Promotion reference — one row per promotion event.
-- Source  : promotions_raw
-- ---------------------------------------------------------------------------
CREATE OR REPLACE VIEW vw_promo_calendar AS
SELECT
    promo_id,
    promo_name,
    region,
    category,
    start_week,
    end_week,
    discount_pct
FROM promotions_raw
