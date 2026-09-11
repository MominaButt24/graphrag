from src.planner import create_plan
from src.plan_state import PlanState


question = "What is emotional intelligence?"

# LLM dynamically creates the plan
plan = create_plan(question)

print("\nINITIAL PLAN")
print("=" * 50)

for task in plan["tasks"]:
    print(f"{task['id']}. {task['description']} [{task['status']}]")


# Python takes ownership of the plan state
plan_state = PlanState(plan)


print("\nSTARTING TASK 1")
plan_state.start_task(1)

for task in plan_state.get_plan()["tasks"]:
    print(f"{task['id']}. {task['description']} [{task['status']}]")


print("\nCOMPLETING TASK 1")
plan_state.complete_task(1)

for task in plan_state.get_plan()["tasks"]:
    print(f"{task['id']}. {task['description']} [{task['status']}]")


print("\nSTARTING TASK 2")
plan_state.start_task(2)

for task in plan_state.get_plan()["tasks"]:
    print(f"{task['id']}. {task['description']} [{task['status']}]")


print("\nCOMPLETING TASK 2")
plan_state.complete_task(2)

for task in plan_state.get_plan()["tasks"]:
    print(f"{task['id']}. {task['description']} [{task['status']}]")