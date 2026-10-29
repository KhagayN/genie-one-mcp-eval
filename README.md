# Genie One MCP traces and evaluations

This sample **runs on dummy data by default**. Bundled spans in [`data/sample_spans.json`](data/sample_spans.json) match the Unity Gateway OpenTelemetry shape so you can explore reconstruction and scoring without warehouse access or a live trace table.

When you are ready to score real Genie One MCP traffic, point the same code at the **[Unity Gateway unified trace table](https://docs.databricks.com/aws/en/ai-gateway/unified-trace-table#what-is-the-unified-trace-table)**. No workspace, catalog, or table name is hardcoded.

Genie One MCP (`system.ai.genie_one_mcp`) does not produce a Genie Space benchmark. Each Gateway row is one MCP `tools/call`. This package groups those rows into turns and scores the reconstructed question / SQL / answer.

This is **not**:

- `system.ai_gateway.usage` (usage metrics only — no tool payloads)
- `system.access.audit` `mcpCall` events (audit trail, not eval inputs)
- Native Genie Space `genie-create-eval-run`

## Run the dummy demo

```bash
cd genie-one-mcp-eval
pip install -r requirements.txt
streamlit run app/app.py
```

Leave `TRACE_TABLE_FQN` and `DATABRICKS_WAREHOUSE_ID` unset. The UI shows a banner that the data is mock. Use **Overview**, **Explorer**, and **Evaluate** on the five synthetic retail turns (happy path, multi-turn, groundedness fail, SQL-only answer, failed poll).

## Unified tracing (optional live source)

Genie One MCP calls through Unity Gateway can land as OpenTelemetry spans in one Unity Catalog table. That is the [unified trace table](https://docs.databricks.com/aws/en/ai-gateway/unified-trace-table#what-is-the-unified-trace-table): one place for model-service and MCP traffic, including `gen_ai.tool.name`, `gen_ai.tool.call.arguments`, and `gen_ai.tool.call.result`.

### 1. Enable and create the table (workspace / metastore)

This is a one-time metastore-admin setup. The sample app does not create the table.

1. An account admin turns on **Enhanced Unity Gateway** from the account console Previews page (the table is in Beta). See [Manage Databricks previews](https://docs.databricks.com/aws/en/admin/workspace-settings/manage-previews).
2. A **metastore admin** opens the workspace: **AI Gateway → Govern → Traces → Set up tracing**.
3. Choose a dedicated catalog and schema (recommended so permissions stay isolated). Databricks creates the table, for example `your_catalog.your_schema.unity_gateway_otel_spans`.
4. Grant **`SELECT`** to identities that will query it. By default only the admin who created the table can read it. Use Unity Catalog grants plus ABAC row filters before broad access. See [Permissions and access control](https://docs.databricks.com/aws/en/ai-gateway/unified-trace-table#permissions-and-access-control) on that page.

Schema details: [Unified trace table schema reference](https://docs.databricks.com/aws/en/ai-gateway/unified-trace-table-reference). MCP spans use `mcp.method.name`, `gen_ai.tool.name`, and the `gen_ai.tool.call.*` attributes this loader expects.

### 2. Point this sample at the table

The loader queries `{TRACE_TABLE_FQN}` for `tools/call` rows whose tool is one of `genie_ask`, `view_ask`, `genie_poll_response`, `genie_get_query_result`, `genie_cancel_response`. The identity that runs SQL (your user locally, or the Databricks App service principal) needs `SELECT` on that table and warehouse **Can use**.

| Variable | Dummy demo | Live traces |
| --- | --- | --- |
| `TRACE_TABLE_FQN` | unset | `catalog.schema.table` of the unified trace table |
| `DATABRICKS_WAREHOUSE_ID` | unset | Warehouse that can query the table |
| `SERVICE_NAME` | — | Optional filter if several MCP services share the table |
| `LOOKBACK_HOURS` | — | Default `24` (table is clustered on `time`) |
| `MLFLOW_EXPERIMENT_NAME` | — | Optional experiment for LLM-judge writes |

Auth uses `databricks.sdk.core.Config()` (CLI profile, OAuth, or app credentials). Do not put tokens in the repo.

**Local (env vars):**

```bash
export TRACE_TABLE_FQN=your_catalog.your_schema.unity_gateway_otel_spans
export DATABRICKS_WAREHOUSE_ID=<warehouse-id>
# optional:
# export SERVICE_NAME=system.ai.genie_one_mcp
# export LOOKBACK_HOURS=24
streamlit run app/app.py
```

**Streamlit sidebar:** paste the same FQN and warehouse id, then **Reload traces**. Check **Use bundled fixture only** to switch back to dummy data.

**Databricks App (`app.yaml` at the package root):**

```yaml
command:
  - "streamlit"
  - "run"
  - "app/app.py"
  - "--server.port=8000"
  - "--server.address=0.0.0.0"

env:
  - name: TRACE_TABLE_FQN
    value: "your_catalog.your_schema.unity_gateway_otel_spans"
  - name: DATABRICKS_WAREHOUSE_ID
    valueFrom: sql-warehouse
```

In the Apps UI, add a SQL warehouse resource with key `sql-warehouse`, then redeploy. Until both the FQN and warehouse are set, the deployed app keeps using dummy spans.

## Reconstruct + score from Python

```python
from src.loaders import resolve_data_source
from src.reconstruct import reconstruct_turns
from src.evaluate import evaluate_turns_code, evaluate_with_mlflow

spans, source = resolve_data_source()  # fixture unless TRACE_TABLE_FQN + warehouse are set
turns = reconstruct_turns(spans)
print(source, len(turns))
print(evaluate_turns_code(turns))
# Optional LLM judges (needs MLflow + a judge endpoint):
# evaluate_with_mlflow(turns, experiment_name="/Users/you/genie_mcp_eval", use_llm_judges=True)
```

## Evaluation layers

1. **Code scorers** (always, no LLM): response present, no error, turn completed, SQL for analytical questions, citations, poll count, latency SLA, numbers in the answer appear in query rows.
2. **LLM judges** via `mlflow.genai.evaluate` on reconstructed turns (answer-sheet mode, no `predict_fn`): `RelevanceToQuery`, `Safety`, Guidelines for data-driven answers and SQL quality.
3. Optional gold facts in `data/sample_eval_expectations.json`.

Do not start with MemAlign / GEPA until SMEs have labeled failed turns.

## Databricks App deploy

Deploy the `genie-one-mcp-eval` folder (root `app.yaml`). Keep `src/` and `data/` as siblings of `app/`.
