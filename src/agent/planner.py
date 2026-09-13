import os
import json
from dotenv import load_dotenv

from langchain.chat_models import init_chat_model
from langchain_core.messages import SystemMessage, HumanMessage

load_dotenv()


# ============================================================
# LLM
# ============================================================

_llm = None


def get_planner_llm():
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


# ============================================================
# PLANNER PROMPT
# ============================================================

PLANNER_SYSTEM_PROMPT = """
You are the planning component of an AI assistant.

Your job is to analyze the user's request and break it into
the necessary high-level tasks required to answer it.

You are ONLY responsible for creating the plan.

Do NOT:
- answer the user's question
- execute tools
- perform searches
- invent results

Create a short, logical sequence of tasks.

Each task should describe WHAT needs to be accomplished,
not the exact implementation details.

Return ONLY valid JSON in this format:

{
    "tasks": [
        {
            "id": 1,
            "description": "..."
        },
        {
            "id": 2,
            "description": "..."
        }
    ]
}

Rules:
- Create only the tasks that are actually necessary.
- Prefer 2-6 tasks.
- Simple questions may require only 1-2 tasks.
- Complex questions may require more tasks.
- Do not create unnecessary steps.
- Tasks must describe actions the agent needs to perform.
- Do not create meta-tasks such as "understand the question",
  "interpret the request", or "plan the answer".
- Do not create a separate task for simply "thinking".
- Prefer tasks involving retrieval, searching, analyzing,
  comparing, validating, or generating the final answer.
- The tasks should represent meaningful work that can be
  marked as completed during execution.
"""


# ============================================================
# CREATE PLAN
# ============================================================

def create_plan(
    question: str,
    history: list[dict] | None = None,
) -> dict:

    messages = [
        SystemMessage(content=PLANNER_SYSTEM_PROMPT),
    ]

    # Optional conversation context
    if history:
        for turn in history[-6:]:
            messages.append(
                HumanMessage(
                    content=f"{turn['role']}: {turn['content']}"
                )
            )

    messages.append(
        HumanMessage(
            content=question
        )
    )

    response = get_planner_llm().invoke(messages)

    raw_content = response.content

    # --------------------------------------------------------
    # Parse JSON returned by LLM
    # --------------------------------------------------------

    if isinstance(raw_content, list):
        raw_content = "".join(
            block.get("text", "")
            if isinstance(block, dict)
            else str(block)
            for block in raw_content
        )

    raw_content = raw_content.strip()

    try:
        plan = json.loads(raw_content)

    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Planner returned invalid JSON: {raw_content}"
        ) from exc

    # --------------------------------------------------------
    # Validate basic structure
    # --------------------------------------------------------

    if not isinstance(plan, dict):
        raise ValueError("Planner output must be a JSON object.")

    tasks = plan.get("tasks")

    if not isinstance(tasks, list) or not tasks:
        raise ValueError(
            "Planner must return a non-empty tasks list."
        )

    # --------------------------------------------------------
    # Add runtime state
    #
    # The LLM creates the task.
    # Python controls the status.
    # --------------------------------------------------------

    for task in tasks:

        if "id" not in task or "description" not in task:
            raise ValueError(
                "Each task must contain id and description."
            )

        task["status"] = "pending"

    return {
        "tasks": tasks
    }

if __name__ == "__main__":

    question = (
        # "Compare conflict management and emotional intelligence "
        # "using the information available in the knowledge base."
        "what is emotionl intelligence"
        # "hello"
    )

    plan = create_plan(question)

    print("\nGenerated Plan:\n")

    for task in plan["tasks"]:
        print(
            f"{task['id']}. "
            f"{task['description']} "
            f"[{task['status']}]"
        )