from src.chat_history_client import answer_query


thread_id = "test-plan-thread"
user_id = "test-plan-user"

query = "What is emotional intelligence?"

print("\n" + "=" * 60)
print("TESTING PLANNER INTEGRATION")
print("=" * 60)

result = answer_query(
    thread_id=thread_id,
    user_id=user_id,
    query=query,
)

print("\n" + "=" * 60)
print("GENERATED PLAN")
print("=" * 60)

for task in result["plan"]["tasks"]:
    print(
        f"{task['id']}. "
        f"{task['description']} "
        f"[{task['status']}]"
    )

print("\n" + "=" * 60)
print("ANSWER")
print("=" * 60)

print(result["answer"])

print("\n" + "=" * 60)
print("RETRIEVAL AVAILABLE")
print("=" * 60)

print(result["retrieval"] is not None)