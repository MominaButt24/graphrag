class PlanState:
    def __init__(self, plan: dict):
        self.tasks = plan["tasks"]

    def start_task(self, task_id: int):
        task = self._get_task(task_id)
        task["status"] = "in_progress"

    def update_plan(self, plan: dict):
        completed_task_ids = {
            task["id"]
            for task in self.tasks
            if task["status"] == "completed"
        }

        self.tasks = plan["tasks"]

        # The planner may accidentally omit or reset a completed task. Keeping
        # runtime execution state authoritative over the LLM's plan output.
        for task in self.tasks:
            if task["id"] in completed_task_ids:
                task["status"] = "completed"

    def complete_task(self, task_id: int):
        task = self._get_task(task_id)
        task["status"] = "completed"

    def fail_task(self, task_id: int):
        task = self._get_task(task_id)
        task["status"] = "failed"

    def get_plan(self) -> dict:
        return {"tasks": self.tasks}

    def get_pending_tasks(self) -> list[dict]:
        return [
            task for task in self.tasks
            if task["status"] == "pending"
        ]

    def _get_task(self, task_id: int) -> dict:
        for task in self.tasks:
            if task["id"] == task_id:
                return task

        raise ValueError(f"Task {task_id} not found")