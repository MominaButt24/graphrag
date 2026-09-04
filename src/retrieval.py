from src.neo4j_client import get_driver
from src.milvus_client import get_collection, get_embedder
from src.logger_config import get_logger
import sentry_sdk
from langchain.chat_models import init_chat_model
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
from dotenv import load_dotenv

load_dotenv()

logger = get_logger(__name__)

llm = init_chat_model(
    model=os.getenv("LLM_MODEL"),
    model_provider="openai",
    api_key=os.getenv("LLM_API_KEY"),
    base_url=os.getenv("LLM_BASE_URL"),
    max_tokens=int(os.getenv("LLM_MAX_TOKENS")),
)

def classify_query(question: str) -> str:
    span = sentry_sdk.get_current_span()
    # deterministic pre-check for corpus-meta questions
    meta_keywords = ["document", "database", "db ", "index", "knowledge base",
                      "corpus", "topics", "contents", "what do you have",
                      "what's in", "what is in" ,"how many topics", "list the"]
    q_lower = question.lower()
    if any(kw in q_lower for kw in meta_keywords):
        if span:
            span.set_tag("classification", "GLOBAL")
            span.set_tag("classification_method", "keyword_override")
        return "GLOBAL"

    prompt = f"""Classify this question as either LOCAL or GLOBAL.

LOCAL = asking about a specific, named entity, person, or fact (e.g. "What is Transactional Leadership?", "Who founded Apex Solutions?").
GLOBAL = asking a broad question needing synthesis across many topics, OR asking what the document/database/knowledge base contains overall, OR asking for a summary of everything indexed.

Examples:
"What is emotional intelligence?" -> LOCAL
"How does courtesy affect performance?" -> LOCAL
"What topics does this cover?" -> GLOBAL
"What's in the database?" -> GLOBAL
"Summarize this document" -> GLOBAL

Question: {question}

Answer with exactly one word: LOCAL or GLOBAL."""

    response = llm.invoke(prompt)
    result = response.content.strip().upper()
    if span:
        span.set_tag("classification", result)
        span.set_tag("classification_method", "llm")
    return result


def extract_entities(question: str) -> list[str]:
    """Pull out ALL named entities the question is asking about — can be 1 or more."""
    prompt = f"""List every specific named entity, person, or subject this question asks about.
Return them as a comma-separated list, nothing else. If there's only one, return just that one.

Question: {question}"""

    response = llm.invoke(prompt)
    return [e.strip() for e in response.content.strip().split(",") if e.strip()]


def local_search(entity_name: str | list[str], question: str):
    entities = [entity_name] if isinstance(entity_name, str) else entity_name
    driver = get_driver()
    facts = []

    def search_single_entity(session, ent: str) -> list[str]:
        """Try id-match first, then label-match, for one entity. Returns its facts."""
        result = session.run("""
            MATCH (n)-[r*1..2]-(neighbor)
            WHERE toLower(n.id) CONTAINS toLower($entity_name)
            RETURN n.id AS entity, [rel IN r | type(rel)] AS relationship_path, neighbor.id AS related_to
            LIMIT 20
        """, entity_name=ent)
        ent_facts = [f"{r['entity']} --{r['relationship_path']}--> {r['related_to']}" for r in result]

        if not ent_facts:
            label_result = session.run("""
                CALL db.labels() YIELD label
                WHERE toLower(label) CONTAINS toLower($entity_name)
                RETURN label
            """, entity_name=ent)
            matched_labels = [r["label"] for r in label_result]
            if matched_labels:
                target_label = matched_labels[0]
                label_nodes = session.run(f"""
                    MATCH (n:`{target_label}`)
                    OPTIONAL MATCH (n)-[r]-(neighbor)
                    RETURN n.id AS entity, type(r) AS relationship, neighbor.id AS related_to
                    LIMIT 20
                """)
                ent_facts = [
                    f"{r['entity']} --{r['relationship']}--> {r['related_to']}"
                    for r in label_nodes
                    if r.get('entity') and r.get('relationship') and r.get('related_to')
                ]
        return ent_facts

    with driver.session() as session:
        if len(entities) >= 2:
            # 1. Try to find a connecting path between the first two (best case: shows the actual relationship)
            result = session.run("""
                MATCH path = (a)-[r*1..3]-(b)
                WHERE toLower(a.id) CONTAINS toLower($e1)
                  AND toLower(b.id) CONTAINS toLower($e2)
                RETURN [n IN nodes(path) | n.id] AS path_nodes,
                       [rel IN relationships(path) | type(rel)] AS path_rels
                LIMIT 10
            """, e1=entities[0], e2=entities[1])
            path_facts = [f"{r['path_nodes']} via {r['path_rels']}" for r in result]
            facts.extend(path_facts)

            if not path_facts:
                logger.debug(f"[local_search] no path found between '{entities[0]}' and '{entities[1]}' — falling back to independent search")

            # 2. ALWAYS also search every entity independently — not just as a
            # fallback when the path search finds nothing. Previously, a 3rd+
            # entity (or any entity the path didn't cover) was silently
            # dropped whenever the first two entities found a path, since the
            # fallback below only fired on a totally empty result.
            for ent in entities:
                facts.extend(search_single_entity(session, ent))
        else:
            primary_entity = entities[0] if entities else ""
            facts = search_single_entity(session, primary_entity)

    logger.info(f"[local_search] entity='{entity_name}' | facts found: {len(facts)}")

    if not facts:
        return f"No graph data found matching '{entity_name}'."

    context = "\n".join(facts)
    prompt = f"""Based on these facts from a knowledge graph:
{context}

Answer this question: {question}"""
    response = llm.invoke(prompt)
    return response.content


def global_search(question: str):
    driver = get_driver()
    with driver.session() as session:
        result = session.run("MATCH (c:Community) RETURN c.community_id AS cid, c.summary AS summary")
        summaries = [(row["cid"], row["summary"]) for row in result]

    # map step: check relevance of each summary
    partial_answers = []
    for cid, summary in summaries:
        response = llm.invoke(f"Summary: {summary}\n\nDoes this help answer '{question}'? If yes, explain how in 1-2 sentences.")
        partial_answers.append(response.content)

    # reduce step: combine into one final answer
    combined_prompt = f"Combine these partial answers into one clear answer to '{question}':\n\n" + "\n".join(partial_answers)
    final = llm.invoke(combined_prompt)
    return final.content


def smart_query(question: str) -> str:
    """Single entry point — decides local vs global automatically."""
    query_type = classify_query(question)
    logger.info(f"[router] classified as: {query_type}")

    if query_type == "GLOBAL":
        return global_search(question)
    else:
        entity = extract_entities(question)
        logger.info(f"[router] extracted entity: {entity}")
        return local_search(entity, question)


from src.reranker import rerank_and_merge

# --- Phase 1: hybrid retrieval additions below ---
# smart_query/local_search/global_search above are untouched.

def vector_search(question: str, top_k: int = 5) -> list[dict]:
    """Milvus similarity search — the vector-side counterpart to smart_query."""
    span = sentry_sdk.get_current_span()
    collection = get_collection()
    collection.load()
    embedder = get_embedder()
    query_vec = embedder.encode([question]).tolist()

    results = collection.search(
        data=query_vec,
        anns_field="embedding",
        param={"metric_type": "COSINE", "params": {"ef": 64}},
        limit=top_k,
        output_fields=["text", "source"],
    )

    hits = [
        {
            "text": hit.entity.get("text"),
            "source": hit.entity.get("source"),
            "score": hit.distance,
            "origin": "vector",
        }
        for hit in results[0]
    ]
    if span:
        span.set_tag("vector_hits", len(hits))
    logger.info(f"[vector_search] question='{question}' | hits found: {len(hits)}")
    return hits


def kb_relevance_score(question: str) -> float:
    """
    Cheap relevance check — one fast vector similarity lookup, no graph
    search, no reranking, no LLM call. Used to decide whether a question
    is even worth running the full (slow) hybrid pipeline on, without
    hardcoding what topics happen to be in the corpus today. Scales
    automatically to whatever gets uploaded, since it's just asking
    "does anything in Milvus actually look like this question."
    """
    hits = vector_search(question, top_k=1)
    return hits[0]["score"] if hits else 0.0


def hybrid_search(question: str, vector_top_k: int = 5) -> dict:
    """
    Runs smart_query (graph) and vector_search (Milvus) in parallel for the
    same question.

    Phase 1 only: results are returned unmerged, tagged by origin. Note the
    shape mismatch — graph_result is smart_query's already-synthesized
    answer string, vector_results is a list of scored raw chunks. Reconciling
    that (raw graph facts + a match-confidence signal, comparable to the
    vector scores) is Phase 2's reranker work, not done here.
    """
    span = sentry_sdk.get_current_span()
    output = {
        "graph_result": None,
        "vector_results": [],
        "graph_error": None,
        "vector_error": None,
    }

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = {
            executor.submit(smart_query, question): "graph",
            executor.submit(vector_search, question, vector_top_k): "vector",
        }
        for future in as_completed(futures):
            source = futures[future]
            try:
                result = future.result()
                if source == "graph":
                    output["graph_result"] = result
                else:
                    output["vector_results"] = result
            except Exception as e:
                logger.error(f"[hybrid_search] {source} search failed: {e}")
                output[f"{source}_error"] = str(e)

    if span:
        span.set_tag("hybrid_vector_hits", len(output["vector_results"]))
        span.set_tag("hybrid_graph_error", bool(output["graph_error"]))
    logger.info(f"[hybrid_search] question='{question}' | vector hits: {len(output['vector_results'])} | graph_error: {output['graph_error']}")
    return output


def hybrid_answer(question: str, vector_top_k: int = 5, rerank_top_k: int = 5) -> str:
    """
    Full Phase 1 + Phase 2 pipeline: hybrid_search -> rerank_and_merge
    (from src.reranker) -> one final LLM synthesis over the top-ranked,
    mixed-source context.
    """
    hybrid_result = hybrid_search(question, vector_top_k=vector_top_k)
    ranked = rerank_and_merge(question, hybrid_result, top_k=rerank_top_k)

    if not ranked:
        return "No relevant information found in either the graph or vector store."

    context = "\n\n".join(f"[{c['origin']}] {c['text']}" for c in ranked)
    prompt = f"""Based on this combined context from a knowledge graph and a vector search:
{context}

Answer this question: {question}"""
    response = llm.invoke(prompt)
    return response.content