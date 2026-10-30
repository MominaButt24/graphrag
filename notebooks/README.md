# Notebooks

| Notebook | Purpose |
|----------|---------|
| `graphrag_pipeline_unit_tests.ipynb` | Team walkthrough of the **production** GraphRAG pipeline with runnable unit tests (offline mocks + optional integration) |

## Run

From the repository root:

```bash
jupyter notebook notebooks/graphrag_pipeline_unit_tests.ipynb
```

Use **Run All** for the offline suite. For Neo4j/Milvus smoke checks:

```bash
export GRAPHRAG_RUN_INTEGRATION=1
```

Teaching flow for LangChain + branch search lives at the repo root: `graphrag_flow.ipynb`.
