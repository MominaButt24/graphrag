import chainlit as cl
import chainlit.data as cl_data

from src.chat.history import get_relevant_history, answer_query

from src.storage.milvus_data_layer import get_threads_collection  # noqa: F401


# ============================================================
# AUTH
# ============================================================

@cl.password_auth_callback
def auth_callback(username: str, password: str):

  
    # ... Replace this with real user database later...
    valid_users = {
        "momna": "changeme"
    }

    if valid_users.get(username) == password:
        return cl.User(identifier=username)

    return None


# ============================================================
# NEW CHAT
# ============================================================

@cl.on_chat_start
async def on_chat_start():

    # --------------------------------------------------------
    # Create / get current thread ID
    # --------------------------------------------------------

    thread_id = cl.context.session.thread_id

    cl.user_session.set(
        "thread_id",
        thread_id
    )

    # --------------------------------------------------------
    # Get current Chainlit user
    # --------------------------------------------------------

    user = cl.user_session.get("user")

    if not user:
        user_identifier = "anonymous"
        user_id = "anonymous"

    else:
        user_identifier = user.identifier

        # ----------------------------------------------------
        # Get the persisted Chainlit user from our data layer.
        #
        # This gives us:
        #
        # id         = UUID
        # identifier = momna
        # ----------------------------------------------------

        persisted_user = await cl_data.get_data_layer().get_user(
            identifier=user_identifier
        )

        if persisted_user:
            user_id = persisted_user.id
        else:
            user_id = user_identifier

    # --------------------------------------------------------
    # Store both values in the session
    # --------------------------------------------------------

    cl.user_session.set(
        "user_id",
        user_id
    )

    cl.user_session.set(
        "user_identifier",
        user_identifier
    )

    # --------------------------------------------------------
    # NOTE:
    # Thread is intentionally NOT persisted here anymore.
    #
    # The thread only gets created/titled in on_message,
    # when the user sends the first real message.
    # --------------------------------------------------------

    # --------------------------------------------------------
    # Initial chat message
    #
    # Documents are no longer displayed here.
    # They are available through the persistent Documents
    # button in the sidebar.
    # --------------------------------------------------------

    await cl.Message(
        content="Hi! Ask me anything."
    ).send()


# ============================================================
# RETRIEVAL EXPLORER
# ============================================================

def format_retrieval_explorer(
    retrieval: dict | None
) -> str | None:
    """
    Builds the Retrieval Explorer display from the metadata
    captured by retrieval.py / hybrid_answer().

    Returns a message explaining that no KB retrieval ran
    when this turn did not use the hybrid retrieval pipeline.
    """

    if not retrieval:
        return (
            "🔍 **Retrieval Explorer**\n\n"
            "_No knowledge-base retrieval ran for this turn — either "
            "a simple exchange (like a greeting) that didn't need a "
            "tool, or the answer came from a live web search instead "
            "of your documents._"
        )

    stats = retrieval.get("stats", {}) or {}

    lines = [
        "🔍 **Retrieval Explorer**",
        ""
    ]

    lines.append(
        f"- Vector candidates: "
        f"{stats.get('vector_candidates', 0)}"
    )

    lines.append(
        f"- Graph available: "
        f"{'Yes' if stats.get('graph_available') else 'No'}"
    )

    lines.append(
        f"- Reranked candidates: "
        f"{stats.get('reranked_candidates', 0)}"
    )

    lines.append(
        f"- Final results used: "
        f"{stats.get('final_results', 0)}"
    )

    ranked = retrieval.get("ranked_results") or []

    if ranked:

        lines.append("")
        lines.append("**Top sources used (post-rerank):**")

        for i, r in enumerate(ranked[:5], start=1):

            origin = r.get("origin", "?")

            score = r.get("score")

            score_str = (
                f"{score:.3f}"
                if isinstance(score, (int, float))
                else "n/a"
            )

            snippet = (
                (r.get("text") or "")
                .replace("\n", " ")[:100]
            )

            lines.append(
                f"{i}. `[{origin}]` "
                f"score={score_str} — "
                f"{snippet}..."
            )

    return "\n".join(lines)


# ============================================================
# PER MESSAGE
# ============================================================

@cl.on_message
async def on_message(message: cl.Message):

    thread_id = cl.user_session.get("thread_id")
    user_id = cl.user_session.get("user_id")

    # --------------------------------------------------------
    # Documents shortcut
    #
    # The actual Documents screen is available through the
    # persistent Documents button in the sidebar.
    #
    # /docs is kept as a convenience shortcut.
    # --------------------------------------------------------

    if message.content.strip().lower() == "/docs":

        await cl.Message(
            content="📚 [Open Documents](/documents)"
        ).send()

        return

    # --------------------------------------------------------
    # Document uploads are NOT handled here anymore.
    #
    # Uploading is now handled by:
    #
    # Documents UI
    #      ↓
    # FastAPI /api/v1/upload
    #      ↓
    # MinIO
    #      ↓
    # Redis queue
    #      ↓
    # Ingestion worker
    #      ↓
    # Milvus + Neo4j
    #
    # This prevents having two separate document-ingestion
    # pipelines inside the application.
    # --------------------------------------------------------

    # --------------------------------------------------------
    # Check whether this is the first real message
    # --------------------------------------------------------

    is_first_message = (
        len(
            get_relevant_history(
                thread_id,
                user_id,
                "",
                recent_k=1,
                relevant_k=0,
            )
        )
        == 0
    )

    # --------------------------------------------------------
    # Update thread title
    # --------------------------------------------------------

    if is_first_message:

        title = (
            message.content[:60]
            + (
                "..."
                if len(message.content) > 60
                else ""
            )
        )

        await cl_data.get_data_layer().update_thread(
            thread_id=thread_id,
            name=title,
            user_id=user_id,
        )

    # --------------------------------------------------------
    # Generate answer
    # --------------------------------------------------------

    result = answer_query(
        thread_id,
        user_id,
        message.content,
    )

    # --------------------------------------------------------
    # Send answer
    # --------------------------------------------------------

    await cl.Message(
        content=(
            result["answer"]
            + "\n\n"
            + "📚 [View documents](/documents)"
        )
    ).send()

    # --------------------------------------------------------
    # Retrieval Explorer
    #
    # Shows the actual retrieval cycle:
    #
    # vector candidates
    # graph availability
    # reranking
    # final sources
    # --------------------------------------------------------

    explorer_content = format_retrieval_explorer(
        result.get("retrieval")
    )

    if explorer_content:

        await cl.Message(
            content=explorer_content
        ).send()


# ============================================================
# RESUME EXISTING CHAT
# ============================================================

@cl.on_chat_resume
async def on_chat_resume(thread):

    thread_id = thread["id"]

    cl.user_session.set(
        "thread_id",
        thread_id
    )

    # --------------------------------------------------------
    # Get current authenticated user
    # --------------------------------------------------------

    user = cl.user_session.get("user")

    if not user:

        cl.user_session.set(
            "user_id",
            "anonymous"
        )

        cl.user_session.set(
            "user_identifier",
            "anonymous"
        )

        return

    # --------------------------------------------------------
    # Get persisted user
    # --------------------------------------------------------

    user_identifier = user.identifier

    persisted_user = await cl_data.get_data_layer().get_user(
        identifier=user_identifier
    )

    if persisted_user:

        cl.user_session.set(
            "user_id",
            persisted_user.id
        )

    else:

        cl.user_session.set(
            "user_id",
            user_identifier
        )

    cl.user_session.set(
        "user_identifier",
        user_identifier
    )