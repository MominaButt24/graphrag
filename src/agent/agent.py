"""
agent + tool routing, sitting above the hybrid
retrieval pipeline. Two tools: the hybrid knowledge base (GraphRAG +
Milvus, reranked) and Tavily web search. The system prompt tries the
knowledge base first and only reaches for Tavily when it comes back
empty or insufficient this is "not in the graph" turning into a real
fallback instead of a dead end.
"""
from langchain_core.tools import tool
from langchain_tavily import TavilySearch
from langchain.agents import create_agent

from langfuse.langchain import CallbackHandler
from contextvars import ContextVar

from src.retrieval.hybrid import hybrid_answer
from src.retrieval.relevance import get_retrieval_metadata
from src.retrieval.vector import kb_relevance_score
from src.config.logging import get_logger
from src.generation.llm import get_llm

llm = get_llm()

logger = get_logger(__name__)

langfuse_handler = CallbackHandler()


RELEVANCE_THRESHOLD = 0.25

current_retrieval_query: ContextVar[str | None] = ContextVar(
    "current_retrieval_query",
    default=None,
)


@tool
def hybrid_knowledge_base(question: str) -> str:
    """
    Search the internal knowledge base (graph + vector search, reranked)
    for an answer. Always safe to try — it runs a fast relevance check
    against whatever documents are currently indexed before committing
    to the full search, so it works regardless of what topics have been
    uploaded and self-reports when nothing relevant is found.
    """
    retrieval_query = current_retrieval_query.get() or question
    score = kb_relevance_score(retrieval_query)
    logger.info(f"[hybrid_knowledge_base] relevance score: {score:.3f}")
    if score < RELEVANCE_THRESHOLD:
        return "No relevant information found in the knowledge base for this question."
    result = hybrid_answer(
        question=question,
        retrieval_question=retrieval_query,
    )
    logger.info(f"[hybrid_knowledge_base] hybrid_answer() returned, checking metadata immediately: {get_retrieval_metadata() is not None}")
    return result


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



def run_agent(
    question: str,
    history: list[dict] | None = None,
    execution_context: str | None = None,
    retrieval_query: str | None = None,
) -> dict:

    messages = list(history or [])

    if execution_context:
        question = f"""
Previous task execution context:

{execution_context}

Current task:
{question}

Use the previous task results above when performing the current task.
"""

    if retrieval_query:
        question = f"""
Original task:
{question}

Canonical retrieval query:
{retrieval_query}

Use the canonical retrieval query when the knowledge-base tool is called.
You may still choose the appropriate tool and decide how to answer.
"""

    messages.append({"role": "user", "content": question})

    query_token = current_retrieval_query.set(retrieval_query)
    try:
        result = agent.invoke(
            {"messages": messages},
            config={"callbacks": [langfuse_handler]},
        )
    finally:
        current_retrieval_query.reset(query_token)

    final_message = result["messages"][-1]
    answer = _extract_text(final_message.content)

    logger.info(
        f"[run_agent] content type was "
        f"{type(final_message.content).__name__}, "
        f"extracted {len(answer)} chars"
    )

    retrieval = get_retrieval_metadata()

    logger.info(
        f"[run_agent] get_retrieval_metadata() after agent.invoke() "
        f"returned: {retrieval is not None}"
    )

    return {
        "answer": answer,
        "retrieval": retrieval,
    }

if __name__ == "__main__":
    from src.config.logging import setup_logging
    setup_logging()

    test_questions = [
        "How does workplace courtesy affect team performance?",
        "What are the official rules of carrom?",
    ]
    for q in test_questions:
        print(f"\n{'='*60}\nQ: {q}\n{'='*60}")
        print(run_agent(q))