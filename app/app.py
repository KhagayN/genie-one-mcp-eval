"""Streamlit explorer for Genie One MCP traces from the Unity Gateway unified table."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluate import evaluate_turns_code, evaluate_with_mlflow  # noqa: E402
from src.loaders import resolve_data_source  # noqa: E402
from src.reconstruct import overview_stats, reconstruct_turns  # noqa: E402

st.set_page_config(page_title="Genie One MCP traces", layout="wide")

FIXTURE = ROOT / "data" / "sample_spans.json"


def _sidebar_config() -> dict:
    st.sidebar.header("Data source")
    table_fqn = st.sidebar.text_input(
        "TRACE_TABLE_FQN",
        value=os.environ.get("TRACE_TABLE_FQN", ""),
        help="Unity Gateway unified trace table: catalog.schema.table",
    )
    warehouse_id = st.sidebar.text_input(
        "DATABRICKS_WAREHOUSE_ID",
        value=os.environ.get("DATABRICKS_WAREHOUSE_ID", ""),
    )
    service_name = st.sidebar.text_input(
        "SERVICE_NAME (optional)",
        value=os.environ.get("SERVICE_NAME", ""),
    )
    lookback_hours = st.sidebar.number_input(
        "LOOKBACK_HOURS",
        min_value=1,
        max_value=2160,
        value=int(os.environ.get("LOOKBACK_HOURS", "24")),
    )
    force_fixture = st.sidebar.checkbox("Use bundled fixture only", value=False)
    if st.sidebar.button("Reload traces"):
        st.cache_data.clear()
    return {
        "table_fqn": table_fqn.strip() or None,
        "warehouse_id": warehouse_id.strip() or None,
        "service_name": service_name.strip() or None,
        "lookback_hours": int(lookback_hours),
        "force_fixture": force_fixture,
    }


@st.cache_data(show_spinner=True)
def _load(table_fqn: str | None, warehouse_id: str | None, service_name: str | None, hours: int, force_fixture: bool):
    if force_fixture or not (table_fqn and warehouse_id):
        from src.loaders import load_spans_from_json

        spans = load_spans_from_json(FIXTURE)
        source = f"fixture:{FIXTURE}"
        return spans, source
    return resolve_data_source(
        table_fqn=table_fqn,
        warehouse_id=warehouse_id,
        lookback_hours=hours,
        service_name=service_name,
        fixture_path=FIXTURE,
    )


def _fmt_ms(value: float | None) -> str:
    if value is None:
        return "—"
    if value >= 1000:
        return f"{value / 1000:.1f} s"
    return f"{value:.0f} ms"


cfg = _sidebar_config()
st.title("Genie One MCP traces")
st.caption(
    "Reads Unity Gateway unified-trace-table spans for Genie One MCP tool calls, "
    "reconstructs conversation turns, and scores them. "
    "This is not Genie Space benchmark evaluation."
)

if "load_error" not in st.session_state:
    st.session_state.load_error = None

try:
    spans, source = _load(
        cfg["table_fqn"],
        cfg["warehouse_id"],
        cfg["service_name"],
        cfg["lookback_hours"],
        cfg["force_fixture"],
    )
    st.session_state.load_error = None
except Exception as exc:
    spans, source = [], ""
    st.session_state.load_error = str(exc)

if st.session_state.load_error:
    st.error(st.session_state.load_error)
    st.info("Fix TRACE_TABLE_FQN / warehouse access, or enable the bundled fixture.")
    st.stop()

if source.startswith("fixture:"):
    st.warning(
        "Showing bundled mock spans. Set TRACE_TABLE_FQN and DATABRICKS_WAREHOUSE_ID "
        "to read live Unity Gateway traces."
    )
else:
    st.success(f"Loaded {len(spans)} spans from `{source}`")

if not spans:
    st.info("No Genie One MCP tool-call spans in this window.")
    st.stop()

turns = reconstruct_turns(spans)
stats = overview_stats(turns)

overview_tab, explorer_tab, eval_tab = st.tabs(["Overview", "Explorer", "Evaluate"])

with overview_tab:
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Turns", stats["turn_count"])
    c2.metric("Conversations", stats["conversation_count"])
    c3.metric("Error turns", stats["error_turns"])
    c4.metric("Error rate", f"{stats['error_rate']:.0%}")
    c5.metric("P95 latency", _fmt_ms(stats["p95_latency_ms"]))
    st.subheader("Tool mix")
    st.dataframe(
        [{"tool": k, "calls": v} for k, v in sorted(stats["tool_mix"].items())],
        hide_index=True,
        use_container_width=True,
    )

with explorer_tab:
    labels = [
        f"{t.get('conversation_id') or '—'} / {t.get('response_id') or '—'} — "
        f"{(t.get('inputs') or {}).get('question', '')[:80]}"
        for t in turns
    ]
    idx = st.selectbox("Turn", range(len(turns)), format_func=lambda i: labels[i])
    turn = turns[idx]
    outputs = turn.get("outputs") or {}
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Status", str(outputs.get("status") or "—"))
    m2.metric("Latency", _fmt_ms(turn.get("latency_ms")))
    m3.metric("Polls", turn.get("poll_count") or 0)
    m4.metric("Spans", turn.get("span_count") or 0)
    if outputs.get("error"):
        st.error(outputs["error"])
    st.markdown("**Question**")
    st.write((turn.get("inputs") or {}).get("question") or "—")
    st.markdown("**Answer**")
    st.write(outputs.get("response") or "—")
    st.caption("AI-generated — verify against SQL and query rows.")
    sql = outputs.get("generated_sql")
    if sql:
        st.markdown("**Generated SQL**")
        st.code(sql, language="sql")
    rows = outputs.get("query_rows") or []
    if rows:
        st.markdown("**Query rows**")
        st.dataframe(rows, hide_index=True, use_container_width=True)
    citations = outputs.get("citations") or []
    if citations:
        st.markdown("**Citations**")
        st.json(citations)
    st.markdown("**Span waterfall**")
    water = []
    for span in turn.get("spans") or []:
        attrs = span.get("attributes") or {}
        water.append(
            {
                "tool": attrs.get("gen_ai.tool.name"),
                "status": (span.get("status") or {}).get("code"),
                "latency_ms": None
                if span.get("start_time_unix_nano") is None
                else max(
                    0,
                    (int(span["end_time_unix_nano"] or 0) - int(span["start_time_unix_nano"] or 0))
                    / 1_000_000,
                ),
                "trace_id": span.get("trace_id"),
                "span_id": span.get("span_id"),
            }
        )
    st.dataframe(water, hide_index=True, use_container_width=True)
    with st.expander("Raw spans"):
        st.json(turn.get("spans") or [])

with eval_tab:
    st.write(
        "Code scorers always run locally on reconstructed turns. "
        "Optional `mlflow.genai.evaluate()` LLM judges require MLflow + a judge endpoint."
    )
    use_llm = st.checkbox("Also run MLflow LLM judges", value=False)
    experiment = st.text_input(
        "MLFLOW_EXPERIMENT_NAME (optional)",
        value=os.environ.get("MLFLOW_EXPERIMENT_NAME", ""),
    )
    if st.button("Run evaluation"):
        code_rows = evaluate_turns_code(turns)
        st.session_state["eval_code"] = code_rows
        if use_llm:
            with st.spinner("Running mlflow.genai.evaluate"):
                st.session_state["eval_mlflow"] = evaluate_with_mlflow(
                    turns,
                    experiment_name=experiment or None,
                    use_llm_judges=True,
                )
        else:
            st.session_state["eval_mlflow"] = None

    code_rows = st.session_state.get("eval_code")
    if code_rows:
        table = []
        for row in code_rows:
            flat = {
                "turn_id": row["turn_id"],
                "question": (row.get("question") or "")[:80],
                "passed": f"{row['passed']}/{row['total']}",
            }
            for name, score in (row.get("scores") or {}).items():
                flat[name] = score.get("value")
            table.append(flat)
        st.dataframe(table, hide_index=True, use_container_width=True)
        with st.expander("Scorer rationales"):
            st.json(
                [
                    {
                        "turn_id": row["turn_id"],
                        "scores": row["scores"],
                    }
                    for row in code_rows
                ]
            )
        st.download_button(
            "Download scores JSON",
            data=json.dumps(code_rows, indent=2),
            file_name="genie_mcp_eval_scores.json",
            mime="application/json",
        )

    mlflow_out = st.session_state.get("eval_mlflow")
    if mlflow_out:
        st.subheader("MLflow")
        st.write(f"Mode: `{mlflow_out.get('mode')}`")
        if mlflow_out.get("mlflow_run_id"):
            st.write(f"Run: `{mlflow_out['mlflow_run_id']}`")
        if mlflow_out.get("error"):
            st.warning(mlflow_out["error"])
        if mlflow_out.get("mlflow_metrics"):
            st.json(mlflow_out["mlflow_metrics"])
