"""
config.py
---------
Centralized configuration for the Promotion Analytics AI Assistant.
All settings are loaded from environment variables with sensible defaults.
Secrets (e.g. API keys) must NEVER be hardcoded here — use a .env file.
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env file (if present)
load_dotenv()

# ---------------------------------------------------------------------------
# LLM / Model Settings
# ---------------------------------------------------------------------------
MODEL_NAME: str = os.getenv("MODEL_NAME", "llama-3.3-70b-versatile")

# ---------------------------------------------------------------------------
# Database Paths
# ---------------------------------------------------------------------------
DUCKDB_PATH: str = os.getenv("DUCKDB_PATH", "db/warehouse.duckdb")
CHROMA_PATH: str = os.getenv("CHROMA_PATH", "chroma_db")

# ---------------------------------------------------------------------------
# Data Generation Settings
# ---------------------------------------------------------------------------
ROW_COUNT_MIN: int = int(os.getenv("ROW_COUNT_MIN", 1))
ROW_COUNT_MAX: int = int(os.getenv("ROW_COUNT_MAX", 500))

# ---------------------------------------------------------------------------
# Retry / Resilience Settings
# ---------------------------------------------------------------------------
MAX_RETRIES: int = int(os.getenv("MAX_RETRIES", 2))

# ---------------------------------------------------------------------------
# Logging / Debugging
# ---------------------------------------------------------------------------
LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
DEBUG_MODE: bool = os.getenv("DEBUG_MODE", "true").lower() in {"1", "true", "yes", "on"}

# ---------------------------------------------------------------------------
# API Keys (loaded from environment — never hardcoded)
# ---------------------------------------------------------------------------
GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
