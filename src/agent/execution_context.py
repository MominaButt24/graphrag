class ExecutionContext:
    """
    Shared context for one planned task execution.

    Stores:
    - original user question
    - previous task results
    """

    def __init__(self, user_question: str):
        self.user_question = user_question
        self.task_results: list[dict] = []

    def add_result(
        self,
        task_id: int,
        task_description: str,
        result: dict,
    ):
        self.task_results.append({
            "task_id": task_id,
            "task_description": task_description,
            "answer": result.get("answer", ""),
            "retrieval": result.get("retrieval"),
        })

    def get_context(self) -> str:
        if not self.task_results:
            return ""

        parts = [
            "RESULTS FROM PREVIOUS PLAN TASKS:"
        ]

        for item in self.task_results:
            parts.append(
                f"""
Task {item['task_id']}: {item['task_description']}

Result:
{item['answer']}
"""
            )

        return "\n".join(parts)