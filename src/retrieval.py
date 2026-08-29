from src.neo4j_client import get_driver
from langchain.chat_models import init_chat_model
import os
from dotenv import load_dotenv

load_dotenv()

llm = init_chat_model(
    model=os.getenv("LLM_MODEL"),
    model_provider="openai",
    api_key=os.getenv("LLM_API_KEY"),
    base_url=os.getenv("LLM_BASE_URL"),
    max_tokens=int(os.getenv("LLM_MAX_TOKENS")),
)

def classify_query(question: str) -> str:
    """Decide if a question needs LOCAL (specific entity) or GLOBAL (whole-corpus) search."""
    prompt = f"""Classify this question as either LOCAL or GLOBAL.
LOCAL = asking about a specific person, thing, or fact.
GLOBAL = asking a broad question needing a summary across many topics/themes.

Question: {question}

Answer with exactly one word: LOCAL or GLOBAL."""

    response = llm.invoke(prompt)
    return response.content.strip().upper()


def extract_entity(question: str) -> str:
    """For LOCAL questions, pull out the main entity being asked about."""
    prompt = f"""What is the single main entity, topic, or subject this question is asking about?
Answer with just the name, nothing else.

Question: {question}"""

    response = llm.invoke(prompt)
    return response.content.strip()

def local_search(entity_name: str, question: str):
    driver = get_driver()
    with driver.session() as session:
        result = session.run("""
            MATCH (n)-[r]-(neighbor)
            WHERE toLower(n.id) CONTAINS toLower($entity_name)
            RETURN n.id AS entity, type(r) AS relationship, neighbor.id AS related_to
            LIMIT 20
        """, entity_name=entity_name)
        facts = [f"{r['entity']} --{r['relationship']}--> {r['related_to']}" for r in result]

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
    print(f"[router] classified as: {query_type}")

    if query_type == "GLOBAL":
        return global_search(question)
    else:
        entity = extract_entity(question)
        print(f"[router] extracted entity: {entity}")
        return local_search(entity, question)