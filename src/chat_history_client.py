# -----------------------
import os
import time
import random
from pymilvus import connections, utility, Collection, FieldSchema, CollectionSchema, DataType
from dotenv import load_dotenv

# Reuse the embedder singleton from milvus_client.py instead of loading a
# second copy of all-MiniLM-L6-v2 into memory.
# Adjust this import path if your module lives elsewhere (e.g. src.milvus_client).
from src.milvus_client import get_embedder

load_dotenv()

COLLECTION_NAME = "chat_history"
EMBED_DIM = 384

_collection = None  # module-level singleton, mirrors _driver in neo4j_client.py


def get_chat_history_collection():
    global _collection
    if _collection is None:
        connections.connect(
            alias="default",
            uri=os.getenv("MILVUS_URI"),
            token=os.getenv("MILVUS_TOKEN"),
        )
        if utility.has_collection(COLLECTION_NAME):
            _collection = Collection(COLLECTION_NAME)
        else:
            fields = [
                FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
                FieldSchema(name="thread_id", dtype=DataType.VARCHAR, max_length=64),
                FieldSchema(name="user_id", dtype=DataType.VARCHAR, max_length=64),
                FieldSchema(name="role", dtype=DataType.VARCHAR, max_length=16),      # "user" or "assistant"
                FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=8192),
                FieldSchema(name="timestamp", dtype=DataType.INT64),                  # unix epoch, for ordering
                FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=EMBED_DIM),
            ]
            schema = CollectionSchema(fields, description="Chat history per thread, keyed by thread_id")
            _collection = Collection(COLLECTION_NAME, schema)
            _collection.create_index(
                field_name="embedding",
                index_params={
                    "index_type": "HNSW",
                    "metric_type": "COSINE",
                    "params": {"M": 16, "efConstruction": 200},
                },
            )
        _collection.load()
    return _collection


def close_collection():
    global _collection
    _collection = None


# ---------- Real LLM — now that access is back ----------

from langchain.chat_models import init_chat_model
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

_llm = None  # module-level singleton, same pattern as _collection


def get_llm():
    global _llm
    if _llm is None:
        _llm = init_chat_model(
            model=os.getenv("LLM_MODEL"),
            model_provider="openai",
            api_key=os.getenv("LLM_API_KEY"),
            base_url=os.getenv("LLM_BASE_URL"),
            max_tokens=int(os.getenv("LLM_MAX_TOKENS")),
        )
    return _llm


def build_messages(history: list[dict], current_query: str, retrieved_context: str = "") -> list:
    """Turn Milvus history rows + RAG context into LangChain message objects."""
    system_text = "You are a helpful assistant. Use the conversation history and the retrieved context below to answer the user's latest question."
    if retrieved_context:
        system_text += f"\n\nRetrieved context:\n{retrieved_context}"

    messages = [SystemMessage(content=system_text)]
    for turn in history:
        if turn["role"] == "user":
            messages.append(HumanMessage(content=turn["content"]))
        else:
            messages.append(AIMessage(content=turn["content"]))
    messages.append(HumanMessage(content=current_query))
    return messages


from langfuse.langchain import CallbackHandler

# Reads LANGFUSE_SECRET_KEY / LANGFUSE_PUBLIC_KEY / LANGFUSE_BASE_URL from
# .env automatically — same pattern as src/agent.py
langfuse_handler = CallbackHandler()


def generate_answer(current_query: str, history: list[dict], retrieved_context: str = "") -> str:
    """Real answer generation — replaces dummy_llm() now that the LLM server is up."""
    messages = build_messages(history, current_query, retrieved_context)
    response = get_llm().invoke(messages, config={"callbacks": [langfuse_handler]})
    return response.content


# Your existing hybrid retrieval — adjust import path if retrieval.py lives elsewhere.
from src.agent import run_agent


def answer_query(thread_id: str, user_id: str, query: str) -> str:
    """Full turn: store user msg, pull history, run through the agent (tool
    routing + Tavily fallback + Langfuse tracing), store assistant msg."""
    store_turn(thread_id, user_id, "user", query)

    history = get_relevant_history(thread_id, user_id, query)
    # history includes the user turn we just stored — drop it so it's not
    # duplicated as both "history" and "question" when we call run_agent
    history = [h for h in history if h["content"] != query]
    history_messages = [{"role": h["role"], "content": h["content"]} for h in history]

    agent_result = run_agent(query, history=history_messages)

    answer = agent_result["answer"]
    retrieval = agent_result.get("retrieval")

    from src.logger_config import get_logger
    logger = get_logger(__name__)
    logger.info(f"[chat_history_client][answer_query] FINAL retrieval being returned to Chainlit: {retrieval is not None}")

    store_turn(thread_id, user_id, "assistant", answer)
    return {
        "answer": answer,
        "retrieval": retrieval,
    }


# ---------- Mock LLM — kept only as an offline fallback for future testing ----------

DUMMY_RESPONSES = [
    "That's a great question — here's a placeholder answer while the real LLM is offline.",
    "Based on what you asked, here's a mock response for testing thread continuity.",
    "This is dummy output #{} — swap in the real Groq call once the office LLM server is up.",
]


def dummy_llm(query: str, history: list[dict]) -> str:
    base = random.choice(DUMMY_RESPONSES).format(random.randint(1, 999))
    return f"{base}\n\n(You asked: '{query}' | history had {len(history)} prior turns)"


# ---------- Store + fetch ----------

def store_turn(thread_id: str, user_id: str, role: str, content: str):
    collection = get_chat_history_collection()
    embedding = get_embedder().encode(content).tolist()
    collection.insert([{
        "thread_id": thread_id,
        "user_id": user_id,
        "role": role,
        "content": content,
        "timestamp": int(time.time()),
        "embedding": embedding,
    }])
    collection.flush()


def get_relevant_history(thread_id: str, user_id: str, current_query: str, recent_k=3, relevant_k=5):
    collection = get_chat_history_collection()

    # Recent turns — always included, for immediate conversational flow (pronouns, "it", etc.)
    recent = collection.query(
        expr=f'thread_id == "{thread_id}"',
        output_fields=["role", "content", "timestamp"],
        limit=recent_k * 2,  # user+assistant pairs
    )
    recent = sorted(recent, key=lambda r: r["timestamp"])[-recent_k * 2:]

    # Semantically relevant older turns — vector search, scoped to this thread only.
    # Skip entirely when relevant_k=0 (Milvus rejects limit=0 as invalid) — this is
    # the case for callers that only want recency, like the "is this the first
    # message in this thread" check in on_message.
    relevant_hits = []
    if relevant_k > 0:
        query_embedding = get_embedder().encode(current_query).tolist()
        relevant = collection.search(
            data=[query_embedding],
            anns_field="embedding",
            param={"metric_type": "COSINE", "params": {"ef": 64}},
            limit=relevant_k,
            expr=f'user_id == "{user_id}"',
            output_fields=["role", "content", "timestamp"],
        )
        relevant_hits = [hit.entity.to_dict() for hit in relevant[0]]

    # Dedupe — a message pulled in by both recency and relevance shouldn't appear twice
    seen = set()
    merged = []
    for r in recent + relevant_hits:
        key = (r.get("timestamp"), r.get("content"))
        if key not in seen:
            seen.add(key)
            merged.append(r)

    return sorted(merged, key=lambda r: r["timestamp"])


def list_user_threads(user_id: str):
    """For the chat-list sidebar: distinct thread_ids belonging to one user."""
    collection = get_chat_history_collection()
    rows = collection.query(
        expr=f'user_id == "{user_id}"',
        output_fields=["thread_id", "content", "timestamp"],
        limit=10000,
    )
    threads = {}
    for r in sorted(rows, key=lambda r: r["timestamp"]):
        threads.setdefault(r["thread_id"], r["content"])  # first message = title
    return threads


# ---------- Quick manual test with dummy data (run this tonight) ----------

if __name__ == "__main__":
    collection = get_chat_history_collection()
    print("Starting entities:", collection.num_entities)

    thread_id = "test-thread-001"
    user_id = "test-user-001"

    # Simulate ~15 turns across two subtopics, so recency alone would miss the pricing one
    seed_messages = [
        ("user", "What's your pricing model for the pro plan?"),
        ("assistant", "The pro plan is $20/month, billed annually for a discount."),
        ("user", "Does it include team seats?"),
        ("assistant", "Yes, up to 5 seats are included in the pro plan."),
    ] + [
        (role, f"Random unrelated {role} message #{i} about deployment steps")
        for i in range(6)
        for role in ("user", "assistant")
    ]  # generates 12 turns of "noise" after the pricing turns

    for role, content in seed_messages:
        store_turn(thread_id, user_id, role, content)

    print("After insert:", collection.num_entities)

    followup_query = "wait, how many seats did you say were included again?"
    history = get_relevant_history(thread_id, user_id, followup_query)

    print(f"\nRetrieved {len(history)} turns for follow-up: '{followup_query}'")
    for h in history:
        print(f"  [{h['role']}] {h['content'][:70]}")

    found_seats_turn = any("seats" in h["content"].lower() for h in history)
    print("\nSemantic recall worked:", found_seats_turn)

    answer = generate_answer(followup_query, history)
    print("\nLLM answer:\n", answer)

    print("\nUser's thread list:", list_user_threads(user_id))