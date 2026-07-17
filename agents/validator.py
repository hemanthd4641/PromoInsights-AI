"""
agents/validator.py
--------------------
Phase 6 -- SQL Validation Agent.

Sits between the Query Generation Agent (Phase 5) and the Execution Agent.
Validates generated SQL before it reaches the database by running:

  1. Syntax validation   -- EXPLAIN <sql> against DuckDB
  2. Row-count check     -- SELECT COUNT(*) FROM (<sql>) t
  3. Regeneration signal -- structured signal to trigger re-generation

Design guarantee: this module NEVER raises an exception to the caller.
All errors are captured and returned as structured ValidationResult objects.

Usage:
    from agents.validator import SQLValidator

    validator = SQLValidator()
    result = validator.validate(sql, retries_used=0)
    signal = validator.create_regeneration_signal(result)
    print(result.model_dump_json(indent=2))
    print(signal.model_dump_json(indent=2))
"""

import logging
import sys
from pathlib import Path
from typing import Optional

import duckdb
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Bootstrap: project root on sys.path
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import DUCKDB_PATH, LOG_LEVEL, MAX_RETRIES, ROW_COUNT_MAX, ROW_COUNT_MIN

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
# Stage labels used in ValidationResult
# ---------------------------------------------------------------------------
STAGE_SYNTAX    = "syntax"
STAGE_ROW_COUNT = "row_count"
STAGE_PASSED    = "passed"
STAGE_EXCEPTION = "exception"


# ---------------------------------------------------------------------------
# Pydantic Models
# ---------------------------------------------------------------------------


class ValidationResult(BaseModel):
    """
    Structured output from a single validation attempt.

    Fields
    ------
    is_valid          : True only if all validation stages passed.
    validation_stage  : The last stage executed ('syntax', 'row_count', 'passed').
    failure_reason    : Human-readable reason for failure (None on success).
    row_count         : Estimated result row count (None if not reached).
    retry_required    : True if the failure warrants a regeneration attempt.
    retries_used      : How many regeneration attempts have been made so far.
    sql               : The SQL string that was validated.
    """

    is_valid: bool = Field(description="True if all validation stages passed.")
    validation_stage: str = Field(description="Last validation stage executed.")
    failure_reason: Optional[str] = Field(
        default=None, description="Reason for failure, or None on success."
    )
    row_count: Optional[int] = Field(
        default=None, description="Estimated row count from the query."
    )
    retry_required: bool = Field(
        default=False, description="True if a regeneration attempt should be made."
    )
    retries_used: int = Field(
        default=0, description="Number of regeneration retries already consumed."
    )
    sql: str = Field(description="The SQL string that was validated.")


class RegenerationSignal(BaseModel):
    """
    Signal emitted after validation to instruct the orchestrator
    whether to regenerate the SQL query.

    Fields
    ------
    should_regenerate : True if validation failed and retries remain.
    reason            : Human-readable explanation for the decision.
    retries_remaining : Retries left before giving up (0 = no more retries).
    """

    should_regenerate: bool = Field(
        description="True if the SQL should be regenerated."
    )
    reason: str = Field(description="Reason for the regeneration decision.")
    retries_remaining: int = Field(
        description="Number of remaining regeneration attempts.", ge=0
    )


# ---------------------------------------------------------------------------
# SQLValidator
# ---------------------------------------------------------------------------


class SQLValidator:
    """
    SQL Validation Agent — validates DuckDB SQL through a multi-stage pipeline.

    Stage 1 -- Syntax:    EXPLAIN <sql> via DuckDB
    Stage 2 -- Row count: SELECT COUNT(*) FROM (<sql>) t vs. configured bounds

    All validation paths are wrapped in exception handlers so this class
    NEVER raises to its caller — failures are always returned as
    ValidationResult objects with is_valid=False.
    """

    def __init__(self, db_path: Optional[str] = None) -> None:
        """
        Args:
            db_path: Path to the DuckDB warehouse file.
                     Defaults to DUCKDB_PATH from config.py.
        """
        self._db_path = db_path or str(PROJECT_ROOT / DUCKDB_PATH)
        log.info("SQLValidator initialised — db: %s", self._db_path)
        log.info(
            "  Row count bounds: [%d, %d] | MAX_RETRIES: %d",
            ROW_COUNT_MIN, ROW_COUNT_MAX, MAX_RETRIES,
        )

    # ------------------------------------------------------------------
    # Internal DuckDB connection helper
    # ------------------------------------------------------------------

    def _connect(self) -> duckdb.DuckDBPyConnection:
        """Open a read-only DuckDB connection to the warehouse."""
        return duckdb.connect(self._db_path, read_only=True)

    # ------------------------------------------------------------------
    # Stage 1 — Syntax Validation
    # ------------------------------------------------------------------

    def validate_syntax(self, sql: str, retries_used: int = 0) -> ValidationResult:
        """
        Run EXPLAIN <sql> against DuckDB to catch syntax and planning errors.

        Args:
            sql         : SQL string to validate.
            retries_used: Number of retries already consumed (passed through).

        Returns:
            ValidationResult with is_valid=True on success, False on failure.
        """
        log.info("--- [Syntax] Validating SQL ---")
        log.debug("SQL: %s", sql[:200])

        try:
            con = self._connect()
            con.execute(f"EXPLAIN {sql}")
            con.close()

            log.info("  [Syntax] PASS")
            return ValidationResult(
                is_valid=True,
                validation_stage=STAGE_SYNTAX,
                failure_reason=None,
                row_count=None,
                retry_required=False,
                retries_used=retries_used,
                sql=sql,
            )

        except duckdb.Error as exc:
            reason = str(exc).strip()
            log.warning("  [Syntax] FAIL -- %s", reason)
            return ValidationResult(
                is_valid=False,
                validation_stage=STAGE_SYNTAX,
                failure_reason=reason,
                row_count=None,
                retry_required=True,
                retries_used=retries_used,
                sql=sql,
            )

        except Exception as exc:          # pragma: no cover
            reason = f"Unexpected error during syntax check: {exc}"
            log.error("  [Syntax] EXCEPTION -- %s", reason)
            return ValidationResult(
                is_valid=False,
                validation_stage=STAGE_EXCEPTION,
                failure_reason=reason,
                row_count=None,
                retry_required=True,
                retries_used=retries_used,
                sql=sql,
            )

    # ------------------------------------------------------------------
    # Row-count estimation helper
    # ------------------------------------------------------------------

    def estimate_row_count(self, sql: str) -> int:
        """
        Estimate the number of rows the query will return.

        Wraps the SQL in SELECT COUNT(*) FROM (<sql>) t.
        Returns -1 if the count cannot be computed (e.g. syntax error).

        Args:
            sql: Base SQL query.

        Returns:
            Row count integer, or -1 on error.
        """
        count_sql = f"SELECT COUNT(*) FROM ({sql}) t"
        log.debug("  [RowCount] count SQL: %s", count_sql[:150])
        try:
            con = self._connect()
            result = con.execute(count_sql).fetchone()
            con.close()
            count = int(result[0]) if result else 0
            log.info("  [RowCount] estimated rows = %d", count)
            return count
        except Exception as exc:
            log.warning("  [RowCount] estimation failed: %s", exc)
            return -1

    # ------------------------------------------------------------------
    # Stage 2 — Row-Count Validation
    # ------------------------------------------------------------------

    def validate_row_count(self, sql: str, retries_used: int = 0) -> ValidationResult:
        """
        Validate that the query returns a row count within the configured bounds
        [ROW_COUNT_MIN, ROW_COUNT_MAX].

        Args:
            sql         : SQL string to validate.
            retries_used: Number of retries already consumed.

        Returns:
            ValidationResult — passes if ROW_COUNT_MIN <= count <= ROW_COUNT_MAX.
        """
        log.info("--- [RowCount] Validating row count ---")

        count = self.estimate_row_count(sql)

        # Count estimation itself failed
        if count == -1:
            reason = "Row count estimation failed (query may be invalid)."
            log.warning("  [RowCount] FAIL -- %s", reason)
            return ValidationResult(
                is_valid=False,
                validation_stage=STAGE_ROW_COUNT,
                failure_reason=reason,
                row_count=None,
                retry_required=True,
                retries_used=retries_used,
                sql=sql,
            )

        # Below minimum threshold
        if count < ROW_COUNT_MIN:
            reason = (
                f"Row count {count} is below minimum threshold {ROW_COUNT_MIN}. "
                "Query may be too restrictive or data may be missing."
            )
            log.warning("  [RowCount] FAIL -- %s", reason)
            return ValidationResult(
                is_valid=False,
                validation_stage=STAGE_ROW_COUNT,
                failure_reason=reason,
                row_count=count,
                retry_required=True,
                retries_used=retries_used,
                sql=sql,
            )

        # Above maximum threshold
        if count > ROW_COUNT_MAX:
            reason = (
                f"Row count {count} exceeds maximum threshold {ROW_COUNT_MAX}. "
                "Consider adding filters or a LIMIT clause."
            )
            log.warning("  [RowCount] FAIL -- %s", reason)
            return ValidationResult(
                is_valid=False,
                validation_stage=STAGE_ROW_COUNT,
                failure_reason=reason,
                row_count=count,
                retry_required=True,
                retries_used=retries_used,
                sql=sql,
            )

        # Within bounds -- pass
        log.info("  [RowCount] PASS -- %d rows (bounds: [%d, %d])",
                 count, ROW_COUNT_MIN, ROW_COUNT_MAX)
        return ValidationResult(
            is_valid=True,
            validation_stage=STAGE_ROW_COUNT,
            failure_reason=None,
            row_count=count,
            retry_required=False,
            retries_used=retries_used,
            sql=sql,
        )

    # ------------------------------------------------------------------
    # Full Validation Pipeline
    # ------------------------------------------------------------------

    def validate(self, sql: str, retries_used: int = 0) -> ValidationResult:
        """
        Run the full validation pipeline on a SQL string.

        Pipeline:
          Stage 1: Syntax check  (EXPLAIN)
          Stage 2: Row-count check

        The pipeline short-circuits on the first failure.
        No exception will escape this method.

        Args:
            sql         : SQL string to validate.
            retries_used: Number of prior regeneration attempts.

        Returns:
            ValidationResult describing the outcome of the validation.
        """
        log.info("=" * 60)
        log.info("SQLValidator.validate() [retries_used=%d]", retries_used)
        log.info("  SQL: %s", sql[:120].strip())

        # Stage 1 — Syntax
        syntax_result = self.validate_syntax(sql, retries_used)
        if not syntax_result.is_valid:
            log.info("  Pipeline stopped at: syntax")
            return syntax_result

        # Stage 2 — Row Count
        row_result = self.validate_row_count(sql, retries_used)
        if not row_result.is_valid:
            log.info("  Pipeline stopped at: row_count")
            return row_result

        # All stages passed
        final = ValidationResult(
            is_valid=True,
            validation_stage=STAGE_PASSED,
            failure_reason=None,
            row_count=row_result.row_count,
            retry_required=False,
            retries_used=retries_used,
            sql=sql,
        )
        log.info("  All validation stages PASSED -- row_count=%d", final.row_count or 0)
        return final

    # ------------------------------------------------------------------
    # Regeneration Signal
    # ------------------------------------------------------------------

    def create_regeneration_signal(
        self, validation_result: ValidationResult
    ) -> RegenerationSignal:
        """
        Create a RegenerationSignal from a ValidationResult.

        Rules:
          - If validation passed: should_regenerate = False.
          - If validation failed and retries remain: should_regenerate = True.
          - If validation failed and no retries remain: should_regenerate = False
            (but reason communicates exhaustion).

        Args:
            validation_result: The result from validate().

        Returns:
            RegenerationSignal for the orchestrator to act on.
        """
        retries_remaining = max(0, MAX_RETRIES - validation_result.retries_used)

        if validation_result.is_valid:
            signal = RegenerationSignal(
                should_regenerate=False,
                reason="Validation passed — no regeneration required.",
                retries_remaining=retries_remaining,
            )
        elif not validation_result.retry_required:
            signal = RegenerationSignal(
                should_regenerate=False,
                reason=(
                    f"Validation failed at stage '{validation_result.validation_stage}' "
                    "but retry is not warranted for this failure type."
                ),
                retries_remaining=retries_remaining,
            )
        elif retries_remaining > 0:
            signal = RegenerationSignal(
                should_regenerate=True,
                reason=(
                    f"Validation failed at stage '{validation_result.validation_stage}': "
                    f"{validation_result.failure_reason or 'unknown reason'}. "
                    f"Triggering regeneration ({retries_remaining} retries remaining)."
                ),
                retries_remaining=retries_remaining,
            )
        else:
            signal = RegenerationSignal(
                should_regenerate=False,
                reason=(
                    f"Validation failed at stage '{validation_result.validation_stage}': "
                    f"{validation_result.failure_reason or 'unknown reason'}. "
                    "No retries remaining — returning final failure."
                ),
                retries_remaining=0,
            )

        log.info(
            "  RegenerationSignal: regenerate=%s | retries_remaining=%d",
            signal.should_regenerate,
            signal.retries_remaining,
        )
        return signal


# ---------------------------------------------------------------------------
# CLI validation
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import json

    validator = SQLValidator()

    SAMPLES = [
        # (label, sql, retries_used)
        ("Valid SELECT",         "SELECT * FROM vw_weekly_sales LIMIT 10",            0),
        ("Valid aggregation",    "SELECT region, SUM(revenue) FROM vw_weekly_sales GROUP BY region", 0),
        ("Broken syntax",        "SELEC * FROM vw_weekly_sales",                       0),
        ("Missing column",       "SELECT revenuex FROM vw_weekly_sales",               1),
        ("Missing table",        "SELECT * FROM unknown_table",                        1),
        ("Row count too low",    "SELECT * FROM vw_weekly_sales WHERE 1=0",            0),
    ]

    print("\n" + "=" * 70)
    print("SQL VALIDATOR -- VALIDATION RUN")
    print("=" * 70)

    for label, sql, retries in SAMPLES:
        print(f"\n[{label}]")
        result = validator.validate(sql, retries_used=retries)
        signal = validator.create_regeneration_signal(result)
        print(f"  ValidationResult : {result.model_dump_json()}")
        print(f"  RegenerationSignal: {signal.model_dump_json()}")
        print("-" * 60)
