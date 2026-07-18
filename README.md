# PromoInsights AI

## AI-Powered Retail Analytics Copilot

PromoInsights AI is a production-inspired multi-agent analytics assistant that transforms natural language business questions into grounded insights, validated SQL, supporting data, and business-friendly explanations.

Instead of requiring users to write SQL, navigate dashboards, or manually combine reports, PromoInsights AI enables business users to ask questions in plain English and receive actionable answers instantly.

Examples:

- Did PROMO_001 improve sales in the South region?
- Which campaign performed best?
- Compare North and South sales.
- Which category generated the highest revenue?
- Did inventory reduce in the West region?
- What trends do you see in the data?

The system combines LLM reasoning, Retrieval-Augmented Generation (RAG), Text-to-SQL, semantic modeling, validation pipelines, caching, and business-focused response synthesis to deliver reliable analytics experiences.

---

# Business Value

PromoInsights AI helps organizations:

- Reduce dependency on SQL experts
- Enable self-service analytics for business users
- Accelerate decision-making
- Improve visibility into promotions and campaigns
- Analyze inventory and regional performance quickly
- Convert business questions into actionable insights
- Bridge the gap between business stakeholders and data teams

---

# Key Features

| Feature | Description |
|----------|-------------|
| 🧠 Intent Classification | Understands business questions and extracts entities such as regions, campaigns, categories, SKUs, and time windows |
| 🔍 Query Grounding | Uses RAG to resolve ambiguous business terms such as effectiveness, uplift, reduction, and growth |
| 🛠️ Text-to-SQL | Generates validated DuckDB SQL from natural language |
| ✅ SQL Validation | Performs syntax validation and row-count safety checks before execution |
| ⚡ Query Execution | Executes validated SQL with cache-first optimization |
| 📊 Business Insights | Converts query results into executive-friendly explanations |
| 💬 Conversational Analytics | Supports natural language interaction and follow-up questions |
| 📈 Monitoring Dashboard | Tracks latency, validation rate, retries, and system health |
| 🔄 Session Memory | Maintains context across conversations |
| 🔐 Safe Query Generation | Restricts SQL generation to semantic-layer views only |
| 🎯 Business-Oriented Responses | Returns direct answers, supporting data, confidence indicators, and explanations |

---

# Tech Stack

## Frontend

- Streamlit

## Backend

- Python 3.10+

## Database

- DuckDB

## Vector Database

- ChromaDB

## AI & LLM Layer

- Groq
- OpenAI
- Gemini
- Anthropic

(Provider-agnostic architecture)

## Orchestration

- LangChain

## Data Generation

- Faker

## Monitoring

- Custom Metrics Dashboard

---

# System Architecture

```text
User Question
      │
      ▼

┌───────────────────────────────────────────────┐
│ Conversation Router                           │
│ Greeting | Help | Follow-up | Analytics       │
└──────────────────────┬────────────────────────┘
                       │
                       ▼

┌───────────────────────────────────────────────┐
│ Intent Classification Agent                   │
│ Topic Detection + Entity Extraction           │
└──────────────────────┬────────────────────────┘
                       │
                       ▼

┌───────────────────────────────────────────────┐
│ Query Grounding Agent                         │
│ RAG + Business Definitions                    │
└──────────────────────┬────────────────────────┘
                       │
                       ▼

┌───────────────────────────────────────────────┐
│ SQL Generation Agent                          │
│ Natural Language → SQL                        │
└──────────────────────┬────────────────────────┘
                       │
                       ▼

┌───────────────────────────────────────────────┐
│ SQL Validation Agent                          │
│ Syntax Validation + Safety Checks             │
└──────────────────────┬────────────────────────┘
                       │
                       ▼

┌───────────────────────────────────────────────┐
│ Execution & Aggregation Agent                 │
│ DuckDB + Query Cache                          │
└──────────────────────┬────────────────────────┘
                       │
                       ▼

┌───────────────────────────────────────────────┐
│ Response Synthesis Agent                      │
│ Business Insights + Explanations              │
└──────────────────────┬────────────────────────┘
                       │
                       ▼

┌───────────────────────────────────────────────┐
│ Orchestrator                                  │
│ Session Memory + Monitoring + Retry Logic     │
└──────────────────────┬────────────────────────┘
                       │
           ┌───────────┴───────────┐
           ▼                       ▼

     Streamlit UI         Metrics Dashboard
```

---

# Multi-Agent Workflow

## 1. Conversation Router

Determines whether the user message is:

- Greeting
- Help Request
- Follow-Up Question
- Analytics Query

---

## 2. Intent Classification Agent

Identifies:

- Question Type
- Business Topic
- Campaign
- Region
- Category
- SKU
- Time Window

Example:

```json
{
  "topic": "promotion",
  "region": "South",
  "campaign": "PROMO_001"
}
```

---

## 3. Query Grounding Agent

Uses ChromaDB RAG to resolve ambiguous business terms.

Examples:

| User Term | Business Definition |
|------------|-------------------|
| effectiveness | uplift vs baseline |
| reduction | negative week-over-week change |
| impact | revenue improvement |
| growth | percentage increase |

---

## 4. SQL Generation Agent

Generates SQL using:

- Intent
- Grounded Business Definitions
- Schema Catalog
- Few-Shot Examples

Only approved semantic-layer views are accessible.

---

## 5. SQL Validation Agent

Validates:

- SQL Syntax
- Query Plan
- Row Count Bounds
- Semantic Layer Compliance

Invalid SQL is automatically regenerated.

---

## 6. Execution Agent

Executes validated SQL.

Features:

- Query Caching
- Aggregation
- Delta Computation
- Percentage Change Calculation
- Metadata Collection

---

## 7. Response Synthesis Agent

Transforms raw query results into:

- Direct Answer
- Business Insight
- Supporting Table
- Explanation
- Follow-Up Suggestions

Example:

```text
PROMO_001 increased sales by 18.4%
compared to the four-week baseline,
generating an additional ₹125,000 in revenue.
```

---

## 8. Orchestrator

Coordinates the entire workflow.

Responsibilities:

- Session Management
- Retry Handling
- Context Memory
- Error Recovery
- Metrics Logging

---

# Semantic Layer

The assistant never queries raw tables directly.

Instead, it uses curated business views:

```sql
vw_weekly_sales
vw_weekly_inventory
vw_promo_calendar
```

Benefits:

- Consistent metrics
- Safe SQL generation
- Simplified schema
- Reduced hallucinations

---

# Example Questions

## Promotion Analysis

- Did PROMO_001 improve sales?
- Which campaign performed best?
- Compare the top two campaigns.
- Which promotion generated the highest revenue?

## Inventory Analysis

- Did inventory reduce in West region?
- Which SKU is most at risk?
- Show inventory trends.

## Regional Analysis

- Compare North and South sales.
- Which region generated the highest revenue?
- Which region grew fastest?

## Revenue Analysis

- Highest revenue category
- Lowest revenue category
- Revenue trends
- Revenue growth analysis

## Executive Insights

- Summarize business performance.
- What trends do you see?
- What stands out in the data?

---

# Project Structure

```text
PromoInsights-AI/

├── agents/
│   ├── intent_classifier.py
│   ├── query_grounding.py
│   ├── query_gen.py
│   ├── validator.py
│   ├── executor.py
│   ├── synthesizer.py
│   └── orchestrator.py
│
├── app/
│   ├── streamlit_app.py
│   └── metrics_dashboard.py
│
├── rag/
│   ├── glossary.json
│   ├── few_shot_bank.json
│   ├── build_index.py
│   └── retriever.py
│
├── db/
│   ├── warehouse.duckdb
│   ├── semantic_layer.sql
│   ├── schema_catalog.py
│   └── cache.py
│
├── data/
│   └── generate_data.py
│
├── logs/
│   └── metrics_logger.py
│
├── tests/
│   └── test_pipeline.py
│
├── .env
├── requirements.txt
├── README.md
└── run.py
```

---

# Installation

## Clone Repository

```bash
git clone https://github.com/hemanthd4641/PromoInsights-AI.git

cd PromoInsights-AI
```

---

## Create Virtual Environment

```bash
python -m venv venv
```

### Windows

```bash
venv\Scripts\activate
```

### Linux / Mac

```bash
source venv/bin/activate
```

---

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

# Environment Configuration

Create a `.env` file:

```env
LLM_PROVIDER=groq

MODEL_NAME=llama-3.3-70b-versatile

GROQ_API_KEY=your_key_here

DUCKDB_PATH=db/warehouse.duckdb

CHROMA_PATH=chroma_db

ROW_COUNT_MIN=1

ROW_COUNT_MAX=500

MAX_RETRIES=2

LOG_LEVEL=INFO
```

---

# Running the Project

## Step 1: Generate Demo Dataset

```bash
python data/generate_data.py
```

Creates:

- Sales Data
- Inventory Data
- Promotion Data

---

## Step 2: Build Vector Index

```bash
python rag/build_index.py
```

Builds:

- Metric Glossary Index
- Few-Shot SQL Index

---

## Step 3: Launch Application

```bash
python run.py
```

or

```bash
streamlit run app/streamlit_app.py
```

Open:

```text
http://localhost:8501
```

---

# Monitoring Dashboard

Launch:

```bash
streamlit run app/metrics_dashboard.py
```

Tracks:

- Query Success Rate
- SQL Validation Rate
- Response Latency
- Retry Count
- Confidence Score
- Cache Hit Rate

---

# Testing

Run all tests:

```bash
pytest tests/ -v
```

Includes:

- Intent Classification Tests
- Query Grounding Tests
- SQL Generation Tests
- Validation Tests
- Orchestrator Tests
- End-to-End Pipeline Tests

---

# Future Enhancements

- Real Database Integration
- Snowflake Support
- PostgreSQL Support
- Azure OpenAI Support
- Role-Based Access Control
- Dashboard Export
- Report Generation
- Voice Analytics Interface
- Multi-Tenant Support

---

# Why This Project Matters

PromoInsights AI demonstrates how modern AI systems can move beyond simple chatbots and become reliable business copilots.

By combining:

- Multi-Agent AI
- Retrieval-Augmented Generation
- Text-to-SQL
- Semantic Modeling
- Validation Pipelines
- Business Intelligence

the project provides a practical example of production-oriented AI engineering for analytics workflows.

---

# License

MIT License