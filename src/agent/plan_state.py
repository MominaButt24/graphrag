class PlanState:
    def __init__(self, plan: dict):
        self.tasks = plan["tasks"]

    def start_task(self, task_id: int):
        task = self._get_task(task_id)
        task["status"] = "in_progress"

    def complete_task(self, task_id: int):
        task = self._get_task(task_id)
        task["status"] = "completed"

    def fail_task(self, task_id: int):
        task = self._get_task(task_id)
        task["status"] = "failed"

    def get_plan(self) -> dict:
        return {
            "tasks": self.tasks
        }

    def _get_task(self, task_id: int) -> dict:
        for task in self.tasks:
            if task["id"] == task_id:
                return task

        raise ValueError(f"Task {task_id} not found")