from contextvars import ContextVar

from deepagents import create_deep_agent
from langchain.agents.middleware import TodoListMiddleware
from langchain_tavily import TavilySearch
from langchain_core.tools import tool
from langfuse.langchain import CallbackHandler

from src.retrieval.hybrid import hybrid_answer
from src.retrieval.relevance import get_retrieval_metadata
from src.retrieval.vector import kb_relevance_score
from src.config.logging import get_logger
from src.generation.llm import get_llm
from src.queue.plan_events import publish_plan_event

logger = get_logger(__name__)

llm = get_llm()
langfuse_handler = CallbackHandler()

RELEVANCE_THRESHOLD = 0.25

current_retrieval_query: ContextVar[str | None] = ContextVar(
    "current_retrieval_query",
    default=None,
)


@tool
def hybrid_knowledge_base(question: str) -> str:
    """
    Search the internal GraphRAG knowledge base using graph + vector
    retrieval with reranking.
    """
    retrieval_query = current_retrieval_query.get() or question

    score = kb_relevance_score(retrieval_query)

    logger.info(
        f"[hybrid_knowledge_base] relevance score: {score:.3f}"
    )

    if score < RELEVANCE_THRESHOLD:
        return (
            "KNOWLEDGE_BASE_INSUFFICIENT: "
            "No relevant information was found in the internal knowledge base. "
            "You MUST use tavily_search to obtain the information before answering."
        )

    result = hybrid_answer(
        question=question,
        retrieval_question=retrieval_query,
    )

    logger.info(
        "[hybrid_knowledge_base] hybrid_answer completed"
    )

    return result


tavily_search = TavilySearch(max_results=5)


DEEP_AGENT_PROMPT = """
You are the main research agent for a GraphRAG assistant.

You have two research tools:

1. hybrid_knowledge_base
   - Searches the internal GraphRAG knowledge base.
   - Uses graph retrieval + vector retrieval + reranking.
   - Use this whenever the user's question may be answered from
     the internal knowledge base.

2. tavily_search
   - Searches the live web.
   - Use it when the internal knowledge base does not contain
     sufficient information or when current external information
     is required.

WORKFLOW:
PLANNING:

- For every information-seeking question, create a concise todo plan
  before performing research or calling a tool.
- This includes definition, explanation, comparison, factual,
  analytical, and knowledge-base questions.
- Greetings, casual conversation, and trivial conversational responses
  do not require a todo plan.
- If you use hybrid_knowledge_base or tavily_search, the todo plan
  must exist before the tool call.
- Keep todos concise, concrete, and user-visible.
- Each todo should represent a meaningful piece of work.
- Mark a todo in_progress before working on it.
- Mark it completed only after the required work is actually finished.
- If new information reveals missing work, update the todo list.
- Do not recreate tasks that have already been completed.


KNOWLEDGE BASE RULE:

For information-seeking questions, ALWAYS check the internal
knowledge base before answering from general model knowledge,
unless the question clearly requires live/current web information.

Use hybrid_knowledge_base first for questions that could reasonably
be answered from the internal knowledge base.

If hybrid_knowledge_base reports that the knowledge base is
insufficient, then use tavily_search.

For questions requiring current or time-sensitive information,
use tavily_search.

Do not answer an information-seeking question purely from your
own general knowledge when the internal knowledge base should
first be checked.
"""


agent = create_deep_agent(
    model=llm,
    tools=[
        hybrid_knowledge_base,
        tavily_search,
    ],
    system_prompt=DEEP_AGENT_PROMPT,
    middleware=[
        TodoListMiddleware(),
    ],
)


def _extract_text(content) -> str:
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts = []

        for block in content:
            if isinstance(block, dict):
                text = block.get(
                    "text",
                    block.get("content", ""),
                )
                if text:
                    parts.append(str(text))
            else:
                parts.append(str(block))

        return "\n".join(parts)

    return str(content)


# def run_agent(
#     question: str,
#     history: list[dict] | None = None,
#     execution_context: str | None = None,
#     retrieval_query: str | None = None,
# ) -> dict:

#     messages = list(history or [])

#     prompt_parts = []

#     if execution_context:
#         prompt_parts.append(
#             f"""
# Previous execution context:

# {execution_context}
# """
#         )

#     prompt_parts.append(
#         f"""
# Original user request:

# {question}
# """
#     )

#     if retrieval_query:
#         prompt_parts.append(
#             f"""
# Canonical retrieval query:

# {retrieval_query}

# Use this query when calling the knowledge-base tool.
# """
#         )

#     messages.append(
#         {
#             "role": "user",
#             "content": "\n".join(prompt_parts),
#         }
#     )

#     query_token = current_retrieval_query.set(
#         retrieval_query
#     )

#     try:
#         result = agent.invoke(
#             {"messages": messages},
#             config={
#                 "callbacks": [langfuse_handler],
#             },
#         )
#     finally:
#         current_retrieval_query.reset(query_token)

#     final_message = result["messages"][-1]

#     answer = _extract_text(
#         final_message.content
#     )

#     retrieval = get_retrieval_metadata()

#     logger.info(
#         "[run_agent] Deep Agent completed | "
#         f"answer_chars={len(answer)} | "
#         f"todos={result.get('todos')}"
#     )

#     return {
#         "answer": answer,
#         "retrieval": retrieval,
#         "todos": result.get("todos", []),
#         "messages": result.get("messages", []),
#     }

def _parse_agent_chunk(chunk: dict):
    """
    Convert a Deep Agent stream chunk into UI events.
    """

    # -------------------------
    # TODO / TOOL EVENTS
    # -------------------------
    if "tools" in chunk:
        data = chunk["tools"]

        todos = data.get("todos")

        if todos is not None:
            yield {
                "type": "todo_updated",
                "todos": todos,
            }

        for message in data.get("messages", []):
            name = getattr(
                message,
                "name",
                None,
            )

            if name == "write_todos":
                continue

            if name:
                yield {
                    "type": "tool_completed",
                    "tool": name,
                    "tool_call_id": getattr(
                        message,
                        "tool_call_id",
                        None,
                    ),
                }

    # -------------------------
    # MODEL EVENTS
    # -------------------------
    if "model" in chunk:

        for message in chunk["model"].get(
            "messages",
            [],
        ):

            for call in getattr(
                message,
                "tool_calls",
                [],
            ):

                tool_name = call.get("name")

                if tool_name == "write_todos":
                    continue

                yield {
                    "type": "tool_started",
                    "tool": tool_name,
                    "tool_call_id": call.get("id"),
                    "args": call.get("args", {}),
                }

def run_agent(
    question: str,
    history: list[dict] | None = None,
    execution_context: str | None = None,
    retrieval_query: str | None = None,
    run_id: str | None = None,
) -> dict:

    messages = list(history or [])

    prompt_parts = []

    if execution_context:
        prompt_parts.append(
            f"""
Previous execution context:

{execution_context}
"""
        )

    prompt_parts.append(
        f"""
Original user request:

{question}
"""
    )

    if retrieval_query:
        prompt_parts.append(
            f"""
Canonical retrieval query:

{retrieval_query}

Use this query when calling the knowledge-base tool.
"""
        )

    messages.append(
        {
            "role": "user",
            "content": "\n".join(prompt_parts),
        }
    )

    query_token = current_retrieval_query.set(
        retrieval_query
    )

    try:
        stream = agent.stream(
            {"messages": messages},
            config={
                "callbacks": [langfuse_handler],
            },
            stream_mode="updates",
        )

        final_state = {}

        for chunk in stream:

            # --------------------------------
            # Convert stream chunk into UI event
            # --------------------------------
            for event in _parse_agent_chunk(chunk):

                if run_id:
                    publish_plan_event(
                        run_id,
                        event["type"],
                        **{
                            key: value
                            for key, value in event.items()
                            if key != "type"
                        },
                    )

            # --------------------------------
            # Keep latest agent state
            # --------------------------------
            for key, value in chunk.items():
                if isinstance(value, dict):
                    final_state.update(value)

    finally:
        current_retrieval_query.reset(query_token)

    final_messages = final_state.get(
        "messages",
        []
    )

    if not final_messages:
        raise RuntimeError(
            "Deep Agent completed without returning messages."
        )

    final_message = final_messages[-1]

    answer = _extract_text(
        final_message.content
    )

    retrieval = get_retrieval_metadata()

    todos = final_state.get(
        "todos",
        []
    )

    # --------------------------------
    # Publish terminal event
    # --------------------------------
    if run_id:
        publish_plan_event(
            run_id,
            "finished",
            todos=todos,
        )

    logger.info(
        "[run_agent] Deep Agent completed | "
        f"answer_chars={len(answer)} | "
        f"todos={todos}"
    )

    return {
        "answer": answer,
        "retrieval": retrieval,
        "todos": todos,
        "messages": final_messages,
    }

def test_agent_stream(question: str):
    result = agent.stream(
        {
            "messages": [
                {"role": "user", "content": question}
            ]
        },
        stream_mode="updates",
    )

    for chunk in result:
        print("\n--- STREAM EVENT ---")
        print(chunk)