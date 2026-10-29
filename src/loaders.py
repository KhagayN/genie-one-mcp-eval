"""Load Genie One MCP spans from the Unity Gateway unified trace table (or JSON fallback)."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

GENIE_TOOLS = (
    "genie_ask",
    "view_ask",
    "genie_poll_response",
    "genie_get_query_result",
    "genie_cancel_response",
)

_IDENT_PART = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def quote_table_fqn(fqn: str) -> str:
    """Validate catalog.schema.table and backtick-quote each part."""
    parts = [p.strip().strip("`") for p in fqn.split(".")]
    if len(parts) != 3 or not all(_IDENT_PART.match(p) for p in parts):
        raise ValueError(
            "TRACE_TABLE_FQN must be catalog.schema.table "
            "(letters, numbers, underscore only)."
        )
    return ".".join(f"`{p}`" for p in parts)


def _parse_variant(value: Any) -> Any:
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return {"raw": value}
    # databricks-sql may return a mapping-like object
    try:
        return dict(value)
    except (TypeError, ValueError):
        return {"raw": str(value)}


def normalize_span(row: dict[str, Any]) -> dict[str, Any]:
    """Normalize a SQL row or JSON object into a consistent span dict."""
    attrs = _parse_variant(row.get("attributes"))
    status = row.get("status")
    if isinstance(status, str):
        try:
            status = json.loads(status)
        except json.JSONDecodeError:
            status = {"code": status}
    elif hasattr(status, "_asdict"):
        status = status._asdict()
    elif status is not None and not isinstance(status, dict):
        try:
            status = dict(status)
        except (TypeError, ValueError):
            status = {"code": str(status)}

    return {
        "trace_id": row.get("trace_id"),
        "span_id": row.get("span_id"),
        "parent_span_id": row.get("parent_span_id"),
        "name": row.get("name"),
        "service_name": row.get("service_name"),
        "status": status or {},
        "time": str(row.get("time") or ""),
        "start_time_unix_nano": _as_int(row.get("start_time_unix_nano")),
        "end_time_unix_nano": _as_int(row.get("end_time_unix_nano")),
        "attributes": attrs,
    }


def _as_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def load_spans_from_json(path: str | Path) -> list[dict[str, Any]]:
    payload = json.loads(Path(path).read_text())
    rows = payload.get("spans", payload) if isinstance(payload, dict) else payload
    return [normalize_span(row) for row in rows]


def load_spans_from_table(
    table_fqn: str,
    *,
    warehouse_id: str,
    lookback_hours: int = 24,
    service_name: str | None = None,
    max_rows: int = 2000,
) -> list[dict[str, Any]]:
    """Query Unity Gateway unified trace table for Genie One MCP tool calls."""
    from databricks import sql
    from databricks.sdk.core import Config

    quoted = quote_table_fqn(table_fqn)
    hours = int(lookback_hours)
    if hours < 1 or hours > 24 * 90:
        raise ValueError("lookback_hours must be between 1 and 2160")
    limit = int(max_rows)
    if limit < 1 or limit > 50_000:
        raise ValueError("max_rows must be between 1 and 50000")

    tools_list = ", ".join(f"'{t}'" for t in GENIE_TOOLS)
    service_clause = ""
    params: list[Any] = []
    if service_name:
        service_clause = "AND service_name = ?"
        params.append(service_name)

    # Dotted VARIANT keys: CAST(attributes:`key`) is empty on some unified
    # tables. JSON path on to_json(attributes) works.
    tool_name_expr = "get_json_object(to_json(attributes), '$[\"gen_ai.tool.name\"]')"
    mcp_method_expr = "get_json_object(to_json(attributes), '$[\"mcp.method.name\"]')"

    query = f"""
    SELECT
      trace_id,
      span_id,
      parent_span_id,
      name,
      service_name,
      status,
      time,
      start_time_unix_nano,
      end_time_unix_nano,
      attributes
    FROM {quoted}
    WHERE time >= current_timestamp() - INTERVAL {hours} HOURS
      AND {mcp_method_expr} = 'tools/call'
      AND {tool_name_expr} IN ({tools_list})
      AND kind = 'SPAN_KIND_SERVER'
      {service_clause}
    ORDER BY start_time_unix_nano ASC
    LIMIT {limit}
    """

    cfg = Config()
    host = (cfg.host or "").replace("https://", "").replace("http://", "")
    with sql.connect(
        server_hostname=host,
        http_path=f"/sql/1.0/warehouses/{warehouse_id}",
        credentials_provider=lambda: cfg.authenticate,
    ) as conn:
        with conn.cursor() as cur:
            if params:
                cur.execute(query, params)
            else:
                cur.execute(query)
            cols = [d[0] for d in cur.description]
            rows = [dict(zip(cols, row)) for row in cur.fetchall()]
    return [normalize_span(row) for row in rows]


def resolve_data_source(
    *,
    table_fqn: str | None = None,
    warehouse_id: str | None = None,
    lookback_hours: int = 24,
    service_name: str | None = None,
    fixture_path: str | Path | None = None,
) -> tuple[list[dict[str, Any]], str]:
    """
    Load spans from the unified table when configured; otherwise JSON fixtures.

    Returns (spans, source_label).
    """
    table_fqn = table_fqn or os.environ.get("TRACE_TABLE_FQN")
    warehouse_id = warehouse_id or os.environ.get("DATABRICKS_WAREHOUSE_ID")
    service_name = service_name or os.environ.get("SERVICE_NAME") or None
    lookback_hours = int(os.environ.get("LOOKBACK_HOURS", lookback_hours))

    if table_fqn and warehouse_id:
        spans = load_spans_from_table(
            table_fqn,
            warehouse_id=warehouse_id,
            lookback_hours=lookback_hours,
            service_name=service_name or None,
        )
        return spans, f"unity_gateway:{table_fqn}"

    if fixture_path is None:
        fixture_path = Path(__file__).resolve().parents[1] / "data" / "sample_spans.json"
    spans = load_spans_from_json(fixture_path)
    return spans, f"fixture:{fixture_path}"
