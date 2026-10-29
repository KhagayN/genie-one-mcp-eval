"""Reconstruct Genie One MCP turns from Unity Gateway OTel spans."""

from __future__ import annotations

import json
from collections import defaultdict
from typing import Any

GENIE_ASK_TOOLS = {"genie_ask", "view_ask"}


def attr_get(attributes: dict[str, Any], key: str, default: Any = None) -> Any:
    if not attributes:
        return default
    if key in attributes:
        return attributes[key]
    # VARIANT sometimes nests under string keys with different quoting
    for k, v in attributes.items():
        if k.replace("`", "") == key:
            return v
    return default


def parse_jsonish(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (dict, list)):
        return value
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return value
    return value


def tool_name(span: dict[str, Any]) -> str:
    return str(attr_get(span.get("attributes") or {}, "gen_ai.tool.name") or "")


def tool_args(span: dict[str, Any]) -> dict[str, Any]:
    raw = attr_get(span.get("attributes") or {}, "gen_ai.tool.call.arguments")
    if raw is None:
        raw = attr_get(span.get("attributes") or {}, "mlflow.spanInputs")
    parsed = parse_jsonish(raw)
    return parsed if isinstance(parsed, dict) else {}


def tool_result(span: dict[str, Any]) -> dict[str, Any]:
    raw = attr_get(span.get("attributes") or {}, "gen_ai.tool.call.result")
    if raw is None:
        raw = attr_get(span.get("attributes") or {}, "mlflow.spanOutputs")
    parsed = parse_jsonish(raw)
    return parsed if isinstance(parsed, dict) else {"value": parsed}


def status_code(span: dict[str, Any]) -> str:
    status = span.get("status") or {}
    if isinstance(status, dict):
        return str(status.get("code") or "")
    return str(status)


def latency_ms(span: dict[str, Any]) -> float | None:
    start = span.get("start_time_unix_nano")
    end = span.get("end_time_unix_nano")
    if start is None or end is None:
        return None
    return max(0.0, (int(end) - int(start)) / 1_000_000)


def _turn_key(args: dict[str, Any], result: dict[str, Any]) -> tuple[str, str]:
    conv = (
        args.get("conversation_id")
        or result.get("conversation_id")
        or args.get("conversationId")
        or result.get("conversationId")
        or ""
    )
    resp = (
        args.get("response_id")
        or result.get("response_id")
        or args.get("responseId")
        or result.get("responseId")
        or ""
    )
    return str(conv), str(resp)


def reconstruct_turns(spans: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Group MCP tool-call spans into Genie conversation turns."""
    buckets: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    orphans: list[dict[str, Any]] = []

    # First pass: assign ids from ask/poll results so later polls can join.
    for span in sorted(spans, key=lambda s: s.get("start_time_unix_nano") or 0):
        args = tool_args(span)
        result = tool_result(span)
        conv, resp = _turn_key(args, result)
        if conv or resp:
            buckets[(conv, resp)].append(span)
        else:
            orphans.append(span)

    # Attach get_query_result spans that only have a query/response id in args.
    for span in list(orphans):
        args = tool_args(span)
        result = tool_result(span)
        qid = str(args.get("query_id") or args.get("response_id") or result.get("query_id") or "")
        if not qid:
            continue
        for (conv, resp), group in buckets.items():
            if qid in (conv, resp):
                group.append(span)
                orphans.remove(span)
                break

    turns: list[dict[str, Any]] = []
    for (conv, resp), group in buckets.items():
        turns.append(_build_turn(conv, resp, group))
    for span in orphans:
        conv, resp = span.get("trace_id") or "", span.get("span_id") or ""
        turns.append(_build_turn(conv, resp, [span]))

    turns.sort(key=lambda t: t.get("started_unix_nano") or 0)
    return turns


def _build_turn(conversation_id: str, response_id: str, spans: list[dict[str, Any]]) -> dict[str, Any]:
    spans = sorted(spans, key=lambda s: s.get("start_time_unix_nano") or 0)
    ask = next((s for s in spans if tool_name(s) in GENIE_ASK_TOOLS), None)
    polls = [s for s in spans if tool_name(s) == "genie_poll_response"]
    query_spans = [s for s in spans if tool_name(s) == "genie_get_query_result"]

    ask_args = tool_args(ask) if ask else {}
    ask_result = tool_result(ask) if ask else {}
    question = (
        ask_args.get("question")
        or ask_args.get("query")
        or ask_args.get("prompt")
        or ""
    )

    last_poll_result: dict[str, Any] = {}
    for poll in polls:
        last_poll_result = tool_result(poll) or last_poll_result

    query_result: dict[str, Any] = {}
    generated_sql = None
    query_rows: list[Any] = []
    for qs in query_spans:
        query_result = tool_result(qs) or query_result
        generated_sql = (
            query_result.get("sql")
            or query_result.get("generated_sql")
            or query_result.get("query")
            or generated_sql
        )
        rows = query_result.get("rows") or query_result.get("data") or []
        if isinstance(rows, list):
            query_rows = rows

    generated_sql = (
        generated_sql
        or last_poll_result.get("generated_sql")
        or last_poll_result.get("sql")
        or _sql_from_attachments(last_poll_result)
    )

    response_text = (
        last_poll_result.get("answer")
        or last_poll_result.get("text")
        or last_poll_result.get("response")
        or last_poll_result.get("narration")
        or ""
    )
    citations = (
        last_poll_result.get("citations")
        or last_poll_result.get("deep_links")
        or last_poll_result.get("sources")
        or []
    )
    turn_status = (
        last_poll_result.get("status")
        or ask_result.get("status")
        or "unknown"
    )

    error = None
    for span in spans:
        sc = status_code(span)
        if sc in ("STATUS_CODE_ERROR", "ERROR"):
            error = (span.get("status") or {}).get("message") or sc
        rpc = attr_get(span.get("attributes") or {}, "rpc.response.status_code")
        if rpc not in (None, 0, "0"):
            error = error or f"rpc.response.status_code={rpc}"
        res_err = tool_result(span).get("error")
        if res_err:
            error = str(res_err)

    start = min((s.get("start_time_unix_nano") or 0) for s in spans) if spans else None
    end = max((s.get("end_time_unix_nano") or 0) for s in spans) if spans else None
    wall_ms = None
    if start and end:
        wall_ms = max(0.0, (end - start) / 1_000_000)

    conversation_id = conversation_id or str(ask_result.get("conversation_id") or "")
    response_id = response_id or str(ask_result.get("response_id") or "")

    return {
        "turn_id": f"{conversation_id}:{response_id}" if conversation_id or response_id else (spans[0].get("span_id") if spans else ""),
        "conversation_id": conversation_id,
        "response_id": response_id,
        "started_unix_nano": start,
        "ended_unix_nano": end,
        "latency_ms": wall_ms,
        "poll_count": len(polls),
        "tool_names": [tool_name(s) for s in spans],
        "span_count": len(spans),
        "error_spans": sum(
            1
            for s in spans
            if status_code(s) in ("STATUS_CODE_ERROR", "ERROR")
        ),
        "inputs": {"question": question},
        "outputs": {
            "response": response_text,
            "generated_sql": generated_sql,
            "query_rows": query_rows,
            "status": turn_status,
            "citations": citations,
            "progress_steps": last_poll_result.get("progress_steps") or [],
            "error": error,
        },
        "spans": spans,
    }


def _sql_from_attachments(result: dict[str, Any]) -> str | None:
    attachments = result.get("attachments") or []
    if not isinstance(attachments, list):
        return None
    for att in attachments:
        if not isinstance(att, dict):
            continue
        query = att.get("query") or att.get("sql")
        if isinstance(query, dict):
            return query.get("query") or query.get("sql")
        if isinstance(query, str):
            return query
    return None


def overview_stats(turns: list[dict[str, Any]]) -> dict[str, Any]:
    if not turns:
        return {
            "turn_count": 0,
            "conversation_count": 0,
            "error_turns": 0,
            "error_rate": 0.0,
            "p50_latency_ms": None,
            "p95_latency_ms": None,
            "tool_mix": {},
        }
    latencies = sorted(t["latency_ms"] for t in turns if t.get("latency_ms") is not None)
    error_turns = sum(1 for t in turns if t["outputs"].get("error") or t["outputs"].get("status") in ("failed", "incomplete"))
    mix: dict[str, int] = defaultdict(int)
    for t in turns:
        for name in t.get("tool_names") or []:
            mix[name] += 1
    convs = {t["conversation_id"] for t in turns if t.get("conversation_id")}

    def pct(p: float) -> float | None:
        if not latencies:
            return None
        idx = min(len(latencies) - 1, max(0, int(round((p / 100) * (len(latencies) - 1)))))
        return latencies[idx]

    return {
        "turn_count": len(turns),
        "conversation_count": len(convs),
        "error_turns": error_turns,
        "error_rate": error_turns / len(turns),
        "p50_latency_ms": pct(50),
        "p95_latency_ms": pct(95),
        "tool_mix": dict(mix),
    }
