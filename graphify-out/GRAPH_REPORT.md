# Graph Report - graphrag  (2026-10-22)

## Corpus Check
- cluster-only mode — file stats not available

## Summary
- 436 nodes · 863 edges · 35 communities (12 shown, 23 thin omitted)
- Extraction: 99% EXTRACTED · 1% INFERRED · 0% AMBIGUOUS · INFERRED: 8 edges (avg confidence: 0.85)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `1a7c7f40`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- agent.py
- ingestion_worker.py
- history.py
- chainlit_app.py
- settings.py
- main.py
- milvus_data_layer.py
- MilvusDataLayer
- community.py
- graph_builder.py
- PlanState
- ExecutionContext
- Liquid Level
- entrypoint.sh
- Lower Meniscus
- start_ray.sh
- graphrag

## God Nodes (most connected - your core abstractions)
1. `MilvusDataLayer` - 23 edges
2. `get_llm()` - 18 edges
3. `process_ingestion_job()` - 16 edges
4. `get_logger()` - 15 edges
5. `process_query()` - 13 edges
6. `get_driver()` - 12 edges
7. `get_embedder()` - 11 edges
8. `upload_documents()` - 11 edges
9. `PlanState` - 10 edges
10. `smart_query()` - 10 edges

## Surprising Connections (you probably didn't know these)
- `handle_plan_events()` --calls--> `subscribe_plan_events()`  [EXTRACTED]
  chainlit_app.py → src/queue/plan_subscriber.py
- `on_audio_end()` --indirect_call--> `transcribe_audio()`  [INFERRED]
  chainlit_app.py → src/voice/stt.py
- `process_query()` --indirect_call--> `answer_query()`  [INFERRED]
  chainlit_app.py → src/chat/history.py
- `process_query()` --indirect_call--> `synthesize_speech()`  [INFERRED]
  chainlit_app.py → src/voice/tts.py
- `query()` --calls--> `query_graph()`  [EXTRACTED]
  app/routes/query.py → src/services/query_service.py

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **Accurate Measurement Procedure** — data_mineru_test_Physics_9th_Ch1_Physical_Quantities_1_measuring_cylinder, data_mineru_test_Physics_9th_Ch1_Physical_Quantities_1_eye, data_mineru_test_Physics_9th_Ch1_Physical_Quantities_1_lower_meniscus, data_mineru_test_Physics_9th_Ch1_Physical_Quantities_1_parallax_error [EXTRACTED 0.75]
- **Volume Determination by Displacement** — data_mineru_test_Physics_9th_Ch1_Physical_Quantities_1_irregular_solid, data_mineru_test_Physics_9th_Ch1_Physical_Quantities_1_liquid_level, data_mineru_test_Physics_9th_Ch1_Physical_Quantities_1_volume [EXTRACTED 0.75]

## Communities (35 total, 23 thin omitted)

### Community 0 - "agent.py"
Cohesion: 0.06
Nodes (27): query(), Query, query_agent_route(), query_hybrid_route(), _extract_text(), hybrid_knowledge_base(), _parse_agent_chunk(), run_agent() (+19 more)

### Community 1 - "ingestion_worker.py"
Cohesion: 0.07
Nodes (19): chunk_documents(), embed_and_ingest(), ingest_file_to_milvus(), load_and_chunk(), process_document(), process_docx(), process_pptx(), process_xlsx() (+11 more)

### Community 2 - "history.py"
Cohesion: 0.07
Nodes (20): create_plan(), get_planner_llm(), replan(), build_messages(), generate_answer(), get_chat_history_collection(), list_user_threads(), store_turn() (+12 more)

### Community 3 - "chainlit_app.py"
Cohesion: 0.08
Nodes (22): Chainlit Welcome Markdown, auth_callback(), clean_for_tts(), format_retrieval_explorer(), handle_plan_events(), on_audio_chunk(), on_audio_end(), on_audio_start() (+14 more)

### Community 4 - "settings.py"
Cohesion: 0.08
Nodes (13): GraphRAGDeployment, parse_agent_stream(), _parse_chunk(), get_env(), Settings, plan_channel(), publish_plan_event(), plan_channel() (+5 more)

### Community 5 - "main.py"
Cohesion: 0.10
Nodes (12): create_app(), lifespan(), delete_document_route(), health(), setup_logging(), delete_document(), close_collection(), delete_document_chunks() (+4 more)

### Community 6 - "milvus_data_layer.py"
Cohesion: 0.11
Nodes (14): get_documents_route(), upload(), view_document_route(), get_documents(), upload_documents(), get_content_type(), view_document(), finish_document() (+6 more)

### Community 7 - "MilvusDataLayer"
Cohesion: 0.09
Nodes (5): get_threads_collection(), get_user_by_identifier(), get_user_identifier_by_id(), get_users_collection(), MilvusDataLayer

### Community 8 - "community.py"
Cohesion: 0.14
Nodes (9): build_all_community_summaries(), get_community_ids(), get_facts_for_community(), load_graph_from_neo4j(), run_leiden(), summarize_community(), write_communities_to_neo4j(), write_community_summary() (+1 more)

### Community 9 - "graph_builder.py"
Cohesion: 0.25
Nodes (3): build_graph_from_chunks(), get_graph(), test_build_graph_from_chunks_reports_zero_node_extract_as_failed()

### Community 12 - "Liquid Level"
Cohesion: 0.40
Nodes (5): Irregular Solid, Liquid, Liquid Level, Measuring Cylinder, Volume

### Community 14 - "Lower Meniscus"
Cohesion: 0.67
Nodes (3): Eye, Lower Meniscus, Parallax Error

## Knowledge Gaps
- **8 isolated node(s):** `API_BASE_URL`, `graphrag`, `Irregular Solid`, `Liquid`, `Eye` (+3 more)
  These have ≤1 connection - possible missing edges or undocumented components. (Counts symbols only; 176 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **23 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `MilvusDataLayer` connect `MilvusDataLayer` to `milvus_data_layer.py`?**
  _High betweenness centrality (0.072) - this node is a cross-community bridge._
- **What connects `API_BASE_URL`, `graphrag`, `Irregular Solid` to the rest of the system?**
  _8 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `agent.py` be split into smaller, more focused modules?**
  _Cohesion score 0.05926251097453907 - nodes in this community are weakly interconnected._
- **Why does `get_llm()` connect `history.py` to `agent.py`, `community.py`, `graph_builder.py`?**
  _High betweenness centrality (0.053) - this node is a cross-community bridge._
- **Should `ingestion_worker.py` be split into smaller, more focused modules?**
  _Cohesion score 0.06787330316742081 - nodes in this community are weakly interconnected._
- **Why does `get_logger()` connect `agent.py` to `ingestion_worker.py`, `main.py`, `milvus_data_layer.py`?**
  _High betweenness centrality (0.039) - this node is a cross-community bridge._
- **Should `history.py` be split into smaller, more focused modules?**
  _Cohesion score 0.07246376811594203 - nodes in this community are weakly interconnected._