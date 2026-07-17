# 📊 Promotion Analytics AI Assistant

> A production-grade agentic analytics system that converts natural-language business questions into validated SQL, executes them on DuckDB, and returns grounded, structured insights through a Streamlit chat interface.

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://python.org)
[![LangChain](https://img.shields.io/badge/LangChain-0.x-green?logo=chainlink)](https://langchain.com)
[![Groq](https://img.shields.io/badge/LLM-Groq%20LLaMA--3.3--70B-orange)](https://groq.com)
[![DuckDB](https://img.shields.io/badge/DB-DuckDB-yellow)](https://duckdb.org)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-red)](https://streamlit.io)

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Folder Structure](#folder-structure)
- [Installation](#installation)
- [Environment Setup](#environment-setup)
- [Running the System](#running-the-system)
- [Running Tests](#running-tests)
- [Monitoring Dashboard](#monitoring-dashboard)
- [Fresh Clone Validation](#fresh-clone-validation)
- [Design Decisions](#design-decisions)

---

## Overview

**Promotion Analytics AI Assistant** is a multi-agent AI system for retail business analytics. Business users ask plain-English questions about:

- 🎯 **Promotion Performance** — Was PROMO_001 effective in the South region?
- 📦 **Inventory Movement** — Did stock levels decrease in West after Q2?
- 🗺 **Regional Comparisons** — Which region generated the highest revenue?
- 📣 **Campaign Impact** — Which campaign drove the most unit sales?

The system responds with a direct numeric answer, delta, percentage change, supporting data table, business explanation, and the exact SQL that produced the result.

**Key properties:**
- ✅ Grounded — metric definitions retrieved from ChromaDB, not invented by the LLM
- ✅ Safe — SQL only touches semantic-layer views, never raw tables
- ✅ Auditable — generated SQL is always shown to the user
- ✅ Observable — every pipeline run is logged to a metrics CSV and visualised in a live dashboard
- ✅ Never crashes — structured fallback responses on every failure

---

## Features

| Feature | Description |
|---|---|
| 🧠 Intent Classification | LLM classifies question topic and extracts entities (region, SKU, category, time window) |
| 🔍 Query Grounding | ChromaDB RAG resolves business terms (`lift`, `effectiveness`) to canonical definitions |
| 🛠 Text-to-SQL | LLM generates DuckDB SQL constrained to semantic-layer views only |
| ✅ SQL Validation | Syntax check + row-count bounds check with automatic retry loop (max 2) |
| ⚡ Execution Layer | Cache-first DuckDB execution with rollup cache for common queries |
| 📐 Response Synthesis | Produces answer text, delta, % change, coverage flag, and explanation |
| 💬 Streamlit Chat UI | ChatGPT-style interface with metric cards, SQL expander, coverage badges |
| 📈 Monitoring Dashboard | Real-time KPIs: accuracy, latency, confidence, validation rate, retry counts |
| 🔄 Session Memory | Multi-turn context carry-forward for follow-up questions |

---

## Architecture

```
User Question
      │
      ▼
┌───────────────────────────────────────────────────────────────┐
│  Intent Classifier  (Groq LLaMA-3.3-70B + Pydantic output)   │
│  → topic | region | SKU | category | confidence               │
└────────────────────────┬──────────────────────────────────────┘
                         │  confidence ≥ 0.70
                         ▼
┌───────────────────────────────────────────────────────────────┐
│  Query Grounding Agent  (ChromaDB RAG)                        │
│  → resolves ambiguous terms to metric definitions             │
│  → retrieves few-shot SQL examples                            │
└────────────────────────┬──────────────────────────────────────┘
                         │
                         ▼
┌───────────────────────────────────────────────────────────────┐
│  SQL Generation Agent  (Groq LLaMA-3.3-70B)                  │
│  → generates DuckDB SQL from GroundedIntent                   │
│  → constrained to whitelisted semantic-layer views            │
└────────────────────────┬──────────────────────────────────────┘
                         │
                         ▼
┌───────────────────────────────────────────────────────────────┐
│  SQL Validation Agent  (DuckDB EXPLAIN + COUNT estimation)    │
│  → syntax check → row-count bounds check                      │
│  → regeneration signal if invalid (max 2 retries)             │
└────────────────────────┬──────────────────────────────────────┘
                         │  valid SQL
                         ▼
┌───────────────────────────────────────────────────────────────┐
│  Execution & Aggregation Agent  (DuckDB + RollupCache)        │
│  → cache-first query execution                                │
│  → computes delta and % change automatically                  │
└────────────────────────┬──────────────────────────────────────┘
                         │
                         ▼
┌───────────────────────────────────────────────────────────────┐
│  Response Synthesis Agent                                     │
│  → answer text | delta | pct_change | table | explanation     │
│  → coverage flag (complete / partial)                         │
└────────────────────────┬──────────────────────────────────────┘
                         │
                         ▼
┌───────────────────────────────────────────────────────────────┐
│  Orchestrator  (Phase 9)                                      │
│  → wires all agents | session memory | metrics logging        │
│  → fallback responses on any failure                          │
└────────────────────────┬──────────────────────────────────────┘
                         │
              ┌──────────┴──────────┐
              ▼                     ▼
   ┌──────────────────┐   ┌──────────────────────┐
   │  Streamlit UI    │   │  Metrics Dashboard   │
   │  (Phase 10)      │   │  (Phase 11)          │
   └──────────────────┘   └──────────────────────┘
```

---

## Folder Structure

```
promoinsights AI/
│
├── agents/                         # AI agent modules
│   ├── intent_classifier.py        # Phase 3 — Intent + entity extraction
│   ├── query_grounding.py          # Phase 4 — ChromaDB RAG grounding
│   ├── query_gen.py                # Phase 5 — Text-to-SQL generation
│   ├── validator.py                # Phase 6 — SQL syntax + row-count validation
│   ├── executor.py                 # Phase 7 — DuckDB execution + aggregation
│   ├── synthesizer.py              # Phase 8 — Business response synthesis
│   └── orchestrator.py             # Phase 9 — Multi-agent pipeline + session memory
│
├── app/                            # Streamlit applications
│   ├── streamlit_app.py            # Phase 10 — Main chat interface
│   └── metrics_dashboard.py        # Phase 11 — Monitoring dashboard
│
├── rag/                            # RAG components
│   ├── glossary.json               # Phase 2 — Business metric definitions (8–10 terms)
│   ├── few_shot_bank.json          # Phase 2 — Few-shot SQL examples per topic
│   ├── build_index.py              # Phase 2 — ChromaDB index builder
│   └── retriever.py                # Phase 2 — Vector retrieval layer
│
├── db/                             # Database layer
│   ├── schema_catalog.py           # Phase 1 — DuckDB schema definitions
│   ├── semantic_layer.sql          # Phase 1 — Business-facing views
│   ├── cache.py                    # Phase 7 — Rollup cache (5 pre-computed rollups)
│   └── warehouse.duckdb            # Auto-generated database file
│
├── data/                           # Data generation
│   └── generate_data.py            # Phase 0 — Synthetic dataset generator (Faker)
│
├── logs/                           # Metrics and logging
│   ├── metrics_logger.py           # Phase 11 — MetricsLogger + QueryMetrics model
│   └── query_metrics.csv           # Auto-created — pipeline execution log
│
├── chroma_db/                      # ChromaDB vector store (auto-created)
│
├── tests/                          # All test suites
│   ├── test_pipeline.py            # Phase 12 — End-to-end 10-question pipeline tests
│   ├── test_orchestrator.py        # Phase 9  — Orchestrator integration tests
│   ├── test_streamlit_app.py       # Phase 10 — Streamlit app unit tests
│   ├── test_metrics_logger.py      # Phase 11 — MetricsLogger unit tests
│   ├── test_metrics_dashboard.py   # Phase 11 — Dashboard logic unit tests
│   ├── test_synthesizer.py         # Phase 8  — Response synthesis tests
│   ├── test_executor.py            # Phase 7  — Execution agent tests
│   ├── test_validator.py           # Phase 6  — SQL validation tests
│   ├── test_query_gen.py           # Phase 5  — SQL generation tests
│   ├── test_query_grounding.py     # Phase 4  — Grounding agent tests
│   └── test_intent_classifier.py   # Phase 3  — Intent classification tests
│
├── config.py                       # Centralised settings (loaded from .env)
├── run.py                          # Entry point script to launch the Streamlit application
├── requirements.txt                # Python dependencies
├── .env.example                    # Environment variable template
├── NOTES.md                        # Interview talking points & design notes
└── README.md                       # This file
```

---

## Installation

### Prerequisites

- Python 3.10 or higher
- A [Groq API key](https://console.groq.com) (free tier available)

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/hemanthd4641/PromoInsights-AI.git
cd PromoInsights-AI

# 2. Create and activate a virtual environment
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

---

## Environment Setup

Create a `.env` file in the project root:

```bash
# .env
GROQ_API_KEY=your_groq_api_key_here

# Optional overrides (defaults shown)
MODEL_NAME=llama-3.3-70b-versatile
DUCKDB_PATH=db/warehouse.duckdb
CHROMA_PATH=chroma_db
ROW_COUNT_MIN=1
ROW_COUNT_MAX=500
MAX_RETRIES=2
LOG_LEVEL=INFO
```

> **Note:** Never commit your `.env` file. It is already listed in `.gitignore`.

---

## Running the System

### Step 1 — Generate Synthetic Data

```bash
python data/generate_data.py
```

Creates `db/warehouse.duckdb` with 3 years of weekly synthetic retail data:
- `sales_data` — 3,900+ rows (region × SKU × week)
- `inventory_data` — 3,900+ rows
- `promotion_data` — 50 promotions

### Step 2 — Build the RAG Vector Index

```bash
python rag/build_index.py
```

Embeds the business metric glossary and few-shot examples into ChromaDB at `chroma_db/`.

### Step 3 — Launch the Chat Application

```bash
python run.py
```
*(Alternatively, you can run `streamlit run app/streamlit_app.py` directly)*

Opens at [http://localhost:8501](http://localhost:8501).

### Try These Questions

```
Did PROMO_001 improve sales in South region?
Compare North and South sales.
Did inventory reduce in West region?
Which category generated highest revenue?
Which campaign performed best?
```

---

## Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test suites
python tests/test_pipeline.py          # End-to-end (makes real LLM calls)
python tests/test_metrics_logger.py    # Metrics logger (no LLM, fast)
python tests/test_metrics_dashboard.py # Dashboard logic (no LLM, fast)
python tests/test_orchestrator.py      # Orchestrator integration (makes LLM calls)
```

> **Note:** Tests that make real Groq API calls (`test_pipeline.py`, `test_orchestrator.py`) will take 2–5 minutes due to rate-limit delays between calls.

---

## Monitoring Dashboard

After running several questions through the chat app, view the live metrics:

```bash
streamlit run app/metrics_dashboard.py
```

Opens at [http://localhost:8502](http://localhost:8502) (or whichever port Streamlit assigns).

**Dashboard shows:**
| Metric | Description |
|---|---|
| SQL Generation Accuracy | % queries where a valid response was generated |
| Validation Pass Rate | % queries where SQL passed all checks |
| Average Latency | Mean DuckDB execution time (ms) |
| Average Confidence | Mean intent classification score |
| Cache Hit Rate | % queries served from rollup cache |
| Latency trend | Line chart over time |
| Validation breakdown | Pass vs Fail bar chart |
| Confidence distribution | Histogram |
| Retry counts | Bar chart |
| Recent 20 queries | Full metadata table |

You can also click **"Generate Sample Data"** in the dashboard sidebar to populate 15 synthetic records for instant visualisation.

---

## Fresh Clone Validation

A new user should be able to complete the following steps with **no code changes**:

```
✅ 1. git clone https://github.com/hemanthd4641/PromoInsights-AI.git
✅ 2. python -m venv venv && venv\Scripts\activate
✅ 3. pip install -r requirements.txt
✅ 4. Create .env with GROQ_API_KEY=<your_key>
✅ 5. python data/generate_data.py
✅ 6. python rag/build_index.py
✅ 7. python run.py
✅ 8. Ask: "Did PROMO_001 improve sales in South region?"
✅ 9. Verify: answer text + delta + table + SQL shown
✅ 10. streamlit run app/metrics_dashboard.py
✅ 11. Verify: KPI cards populated from real query logs
```

---

## Design Decisions

| Decision | Rationale |
|---|---|
| **Separate grounding from generation** | Keeps metric definitions stable and auditable; SQL agent only focuses on query structure |
| **View whitelist before execution** | Prevents raw table access and prompt injection via table names |
| **Pydantic for all agent I/O** | Enforces type-safe, validated contracts between every agent |
| **Never-crash guarantee** | All `Exception` handlers return a structured `SynthesizedResponse` — the UI never shows a Python traceback |
| **DuckDB for analytics** | Zero infrastructure, in-process, OLAP-optimised — ideal for demo and prototype |
| **CSV-backed metrics** | Zero infrastructure; instantly readable by pandas for the monitoring dashboard |
| **Row-count bounds validation** | Prevents 0-row results (bad SQL) and > 500-row cartesian products (dangerous SQL) |
| **Session memory as dict** | Simple, fast, sufficient for single-server Streamlit deployment |

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

*Built as a demonstration of production-grade agentic AI engineering across 12 implementation phases.*
