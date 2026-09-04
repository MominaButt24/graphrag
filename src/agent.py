"""
Phase 3: agent + tool routing, sitting above the Phase 1+2 hybrid
retrieval pipeline. Two tools: the hybrid knowledge base (GraphRAG +
Milvus, reranked) and Tavily web search. The system prompt tries the
knowledge base first and only reaches for Tavily when it comes back
empty or insufficient — this is "not in the graph" turning into a real
fallback instead of a dead end.
"""

import os
from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_tavily import TavilySearch
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model

from langfuse.langchain import CallbackHandler

from src.retrieval import hybrid_answer, kb_relevance_score
from src.logger_config import get_logger

load_dotenv()

logger = get_logger(__name__)

# Reads LANGFUSE_SECRET_KEY / LANGFUSE_PUBLIC_KEY / LANGFUSE_BASE_URL from
# .env automatically — nothing to pass in here.
langfuse_handler = CallbackHandler()

llm = init_chat_model(
    model=os.getenv("LLM_MODEL"),
    model_provider="openai",
    api_key=os.getenv("LLM_API_KEY"),
    base_url=os.getenv("LLM_BASE_URL"),
    max_tokens=int(os.getenv("LLM_MAX_TOKENS")),
)

# Tune this against your own corpus — run a few genuinely-relevant and
# genuinely-irrelevant test questions through kb_relevance_score() and
# see where the scores actually split. 0.25 is a starting point, not a
# measured value.
RELEVANCE_THRESHOLD = 0.25


@tool
def hybrid_knowledge_base(question: str) -> str:
    """
    Search the internal knowledge base (graph + vector search, reranked)
    for an answer. Always safe to try — it runs a fast relevance check
    against whatever documents are currently indexed before committing
    to the full search, so it works regardless of what topics have been
    uploaded and self-reports when nothing relevant is found.
    """
    score = kb_relevance_score(question)
    logger.info(f"[hybrid_knowledge_base] relevance score: {score:.3f}")
    if score < RELEVANCE_THRESHOLD:
        return "No relevant information found in the knowledge base for this question."
    return hybrid_answer(question)


tavily_search = TavilySearch(max_results=5)

tools = [hybrid_knowledge_base, tavily_search]

SYSTEM_PROMPT = """You are a research assistant with two tools:

1. hybrid_knowledge_base — searches the internal knowledge base. It's
   fast to try and self-reports when nothing relevant is found.

2. tavily_search — live web search. Use this when hybrid_knowledge_base
   reports no relevant information was found, or when its answer is
   clearly insufficient.

CRITICAL RULE: you must call hybrid_knowledge_base for EVERY question
that asks for information — including questions that sound generic,
broad, or like something you could already answer from your own
training (e.g. "what is leadership", "what is X"), and including
meta-questions about the knowledge base itself (e.g. "what topics do
you cover", "what's in your knowledge base", "summarize your
documents"). Do NOT answer such questions from your own general
knowledge, even if you're confident you know the answer — the correct
answer must come from the tool, because the indexed documents may
define or cover the topic differently than your training data does,
and you have no way of knowing that without checking. Never describe
"your knowledge base" from your own training data — if asked what it
contains, call the tool and let it answer.

The ONLY exception: pure greetings and small talk with no actual
question in them ("hello", "how are you", "thanks") — respond to those
directly, no tool needed.

If you are ever unsure whether a question needs the tool, call the
tool. Never skip straight to answering from your own knowledge just
because a question seems simple or general.

CONVERSATION MEMORY: you may see prior user and assistant messages
above the current question. Those are the real, persisted history of
this conversation, retrieved from storage — treat them exactly as you
would treat your own memory of what was said. If the user asks what
you discussed previously, what their earlier messages were, or refers
back to something from earlier in the thread ("that", "it", "the
second one", "what did you say about X again"), answer directly from
those prior messages. Do NOT say you have no memory of the
conversation, that this is your first interaction, or that you can't
recall previous messages — the messages provided to you above ARE
that memory, and denying it is incorrect."""

agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt=SYSTEM_PROMPT,
)


def _extract_text(content) -> str:
    """Some models return AIMessage.content as a list of content blocks
    instead of a plain string on certain turns. Chainlit's cl.Message needs
    a plain string, so normalize here rather than passing whatever comes back."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict):
                parts.append(block.get("text", block.get("content", "")))
            else:
                parts.append(str(block))
        return "\n".join(p for p in parts if p)
    return str(content)


def run_agent(question: str, history: list[dict] | None = None) -> str:
    """
    history: list of {"role": "user"|"assistant", "content": str} dicts,
    oldest first, from chat_history_client.get_relevant_history(). Passed
    in as prior messages so the agent has real conversational memory —
    optional so existing single-question callers (e.g. Gradio) still work
    unchanged.
    """
    messages = list(history or [])
    messages.append({"role": "user", "content": question})

    result = agent.invoke(
        {"messages": messages},
        config={"callbacks": [langfuse_handler]},
    )
    final_message = result["messages"][-1]
    answer = _extract_text(final_message.content)
    logger.info(f"[run_agent] content type was {type(final_message.content).__name__}, extracted {len(answer)} chars")
    return answer


if __name__ == "__main__":
    from src.logger_config import setup_logging
    setup_logging()

    test_questions = [
        "How does workplace courtesy affect team performance?",  # should hit KB only
        "What are the official rules of carrom?",  # known out-of-scope case — should fall back to Tavily
    ]
    for q in test_questions:
        print(f"\n{'='*60}\nQ: {q}\n{'='*60}")
        print(run_agent(q))