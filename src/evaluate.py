"""Score reconstructed Genie One MCP turns (code scorers + optional LLM judges)."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Callable

NUMBER = re.compile(r"-?\d+(?:\.\d+)?")


def _yes(ok: bool, rationale: str) -> dict[str, Any]:
    return {"value": "yes" if ok else "no", "rationale": rationale}


def has_response(turn: dict[str, Any]) -> dict[str, Any]:
    resp = (turn.get("outputs") or {}).get("response")
    if resp and str(resp).strip():
        return _yes(True, f"{len(str(resp))} chars")
    return _yes(False, "No text response")


def no_error(turn: dict[str, Any]) -> dict[str, Any]:
    err = (turn.get("outputs") or {}).get("error")
    if err:
        return _yes(False, str(err)[:300])
    return _yes(True, "No errors on reconstructed turn")


def turn_completed(turn: dict[str, Any]) -> dict[str, Any]:
    status = str((turn.get("outputs") or {}).get("status") or "").lower()
    ok = status in ("completed", "complete", "success", "ok")
    return _yes(ok, f"status={status or 'missing'}")


def has_sql(turn: dict[str, Any]) -> dict[str, Any]:
    question = str((turn.get("inputs") or {}).get("question") or "")
    analytical = bool(re.search(r"\b(how many|sum|avg|average|top|revenue|count|trend)\b", question, re.I))
    sql = (turn.get("outputs") or {}).get("generated_sql")
    if sql and str(sql).strip():
        return _yes(True, "SQL present")
    if not analytical:
        return _yes(True, "Question does not require SQL")
    return _yes(False, "Analytical question but no generated SQL")


def has_citations(turn: dict[str, Any]) -> dict[str, Any]:
    citations = (turn.get("outputs") or {}).get("citations") or []
    if citations:
        return _yes(True, f"{len(citations)} citation(s)")
    return _yes(False, "No citations or deep links")


def poll_count_reasonable(turn: dict[str, Any], max_polls: int = 30) -> dict[str, Any]:
    n = int(turn.get("poll_count") or 0)
    return _yes(n <= max_polls, f"poll_count={n} (max {max_polls})")


def latency_under_sla(turn: dict[str, Any], sla_ms: float = 180_000) -> dict[str, Any]:
    ms = turn.get("latency_ms")
    if ms is None:
        return _yes(False, "Missing latency")
    return _yes(ms <= sla_ms, f"{ms:.0f} ms (SLA {sla_ms:.0f} ms)")


def _row_numbers(rows: list[Any]) -> set[float]:
    found: set[float] = set()
    for n in NUMBER.findall(json.dumps(rows)):
        try:
            found.add(float(n))
        except ValueError:
            continue
    return found


def numbers_grounded_in_rows(turn: dict[str, Any]) -> dict[str, Any]:
    """Deterministic groundedness: numbers in the answer should appear in query rows."""
    outputs = turn.get("outputs") or {}
    response = str(outputs.get("response") or "")
    rows = outputs.get("query_rows") or []
    nums = NUMBER.findall(response)
    if not nums:
        return _yes(True, "No numeric claims in the answer")
    if not rows:
        return _yes(False, "Numeric claims but no query rows to ground them")
    present = _row_numbers(rows)
    missing = []
    for n in nums:
        try:
            val = float(n)
        except ValueError:
            missing.append(n)
            continue
        if not any(abs(val - p) < 1e-6 for p in present):
            missing.append(n)
    if missing:
        return _yes(False, f"Numbers not in query rows: {missing[:5]}")
    return _yes(True, "Numeric claims appear in query rows")


CODE_SCORERS: list[tuple[str, Callable[[dict[str, Any]], dict[str, Any]]]] = [
    ("has_response", has_response),
    ("no_error", no_error),
    ("turn_completed", turn_completed),
    ("has_sql", has_sql),
    ("has_citations", has_citations),
    ("poll_count_reasonable", poll_count_reasonable),
    ("latency_under_sla", latency_under_sla),
    ("numbers_grounded_in_rows", numbers_grounded_in_rows),
]


def score_turn_code(turn: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {name: fn(turn) for name, fn in CODE_SCORERS}


def evaluate_turns_code(turns: list[dict[str, Any]]) -> list[dict[str, Any]]:
    results = []
    for turn in turns:
        scores = score_turn_code(turn)
        passed = sum(1 for s in scores.values() if s["value"] == "yes")
        results.append(
            {
                "turn_id": turn.get("turn_id"),
                "conversation_id": turn.get("conversation_id"),
                "question": (turn.get("inputs") or {}).get("question"),
                "status": (turn.get("outputs") or {}).get("status"),
                "passed": passed,
                "total": len(scores),
                "scores": scores,
            }
        )
    return results


def load_expectations(path: str | Path | None = None) -> dict[str, dict[str, Any]]:
    if path is None:
        path = Path(__file__).resolve().parents[1] / "data" / "sample_eval_expectations.json"
    payload = json.loads(Path(path).read_text())
    records = payload.get("expectations", payload)
    by_id: dict[str, dict[str, Any]] = {}
    for rec in records:
        by_id[rec.get("turn_id") or rec.get("response_id") or ""] = rec
    return by_id


def to_eval_records(
    turns: list[dict[str, Any]],
    expectations: dict[str, dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """MLflow genai.evaluate answer-sheet records."""
    expectations = expectations or {}
    records = []
    for turn in turns:
        rec: dict[str, Any] = {
            "inputs": turn.get("inputs") or {"question": ""},
            "outputs": {
                "response": (turn.get("outputs") or {}).get("response"),
                "generated_sql": (turn.get("outputs") or {}).get("generated_sql"),
                "query_rows": (turn.get("outputs") or {}).get("query_rows"),
                "error": (turn.get("outputs") or {}).get("error"),
            },
        }
        exp = expectations.get(turn.get("turn_id") or "") or expectations.get(
            turn.get("response_id") or ""
        )
        if exp and exp.get("expected_facts"):
            rec["expectations"] = {"expected_facts": exp["expected_facts"]}
        records.append(rec)
    return records


def evaluate_with_mlflow(
    turns: list[dict[str, Any]],
    *,
    experiment_name: str | None = None,
    use_llm_judges: bool = True,
) -> dict[str, Any]:
    """
    Run mlflow.genai.evaluate in existing-trace / answer-sheet mode.

    Falls back to code-only scores if MLflow is not installed or judges fail.
    """
    code_results = evaluate_turns_code(turns)
    summary: dict[str, Any] = {
        "mode": "code_only",
        "results": code_results,
        "mlflow_run_id": None,
        "error": None,
    }
    if not use_llm_judges:
        return summary

    try:
        import mlflow
        from mlflow.genai.scorers import Guidelines, RelevanceToQuery, Safety
    except ImportError as exc:
        summary["error"] = f"mlflow not available: {exc}"
        return summary

    experiment_name = experiment_name or os.environ.get("MLFLOW_EXPERIMENT_NAME")
    if experiment_name:
        mlflow.set_experiment(experiment_name)

    records = to_eval_records(turns, load_expectations())
    scorers: list[Any] = [
        RelevanceToQuery(),
        Safety(),
        Guidelines(
            name="genie_response_quality",
            guidelines=[
                "The response must directly address the user's data question rather than a vague reply.",
                "If SQL was generated, the response must include a data-driven answer, not only echo the SQL.",
                "Numeric claims in the response must be consistent with query_rows when present.",
            ],
        ),
        Guidelines(
            name="genie_sql_quality",
            guidelines=[
                "If SQL is present, aggregations and filters must match the user's intent.",
                "The SQL must not use SELECT * on large tables without LIMIT or a specific filter.",
            ],
        ),
    ]

    try:
        eval_results = mlflow.genai.evaluate(data=records, scorers=scorers)
        summary["mode"] = "mlflow.genai.evaluate"
        run = getattr(eval_results, "run_id", None) or getattr(
            getattr(eval_results, "run", None), "info", None
        )
        summary["mlflow_run_id"] = str(run) if run else None
        summary["mlflow_metrics"] = getattr(eval_results, "metrics", None)
    except Exception as exc:  # judges / tracking may be unconfigured
        summary["error"] = str(exc)
        summary["mode"] = "code_only_mlflow_failed"
    return summary
