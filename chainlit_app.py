# -----------------------------
import os
import chainlit as cl
import chainlit.data as cl_data

from src.chat_history_client import answer_query, get_relevant_history
from src.milvus_data_layer import get_threads_collection  # noqa: F401


# ============================================================
# AUTH
# ============================================================

@cl.password_auth_callback
def auth_callback(username: str, password: str):

    # TODO:
    # Replace this with your real user database later.
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

def format_retrieval_explorer(retrieval: dict | None) -> str | None:
    """
    Builds the Phase 2 "Retrieval Explorer" display from the metadata
    retrieval.py already captures in hybrid_answer().

    Returns None when this turn never ran the hybrid pipeline.
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
        f"- Vector candidates: {stats.get('vector_candidates', 0)}"
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
    # Handle uploaded files
    # --------------------------------------------------------

    if message.elements:

        import uuid as uuid_lib
        import shutil

        from src.ingest import ingest_file_to_milvus
        from src.milvus_data_layer import (
            start_document,
            finish_document,
        )

        UPLOAD_DIR = "data/uploads"

        os.makedirs(
            UPLOAD_DIR,
            exist_ok=True
        )

        for element in message.elements:

            if hasattr(element, "path"):

                filename = (
                    getattr(element, "name", None)
                    or os.path.basename(element.path)
                )

                # ------------------------------------------------
                # Create document ID
                # ------------------------------------------------

                doc_id = str(
                    uuid_lib.uuid4()
                )

                # ------------------------------------------------
                # Register document as processing
                # ------------------------------------------------

                uploaded_at = start_document(
                    doc_id,
                    filename,
                    user_id
                )

                # ------------------------------------------------
                # Save a permanent copy
                #
                # Chainlit's element.path is temporary.
                # ------------------------------------------------

                permanent_path = os.path.join(
                    UPLOAD_DIR,
                    f"{doc_id}_{filename}"
                )

                shutil.copy(
                    element.path,
                    permanent_path
                )

                try:

                    # ------------------------------------------------
                    # Ingest document into Milvus
                    # ------------------------------------------------

                    result = ingest_file_to_milvus(
                        element.path
                    )

                    # ------------------------------------------------
                    # Determine chunk count
                    # ------------------------------------------------

                    chunk_count = (
                        result
                        if isinstance(result, int)
                        else len(result)
                        if hasattr(result, "__len__")
                        else 0
                    )

                    # ------------------------------------------------
                    # Mark document as completed
                    # ------------------------------------------------

                    finish_document(
                        doc_id,
                        filename,
                        user_id,
                        uploaded_at,
                        chunk_count,
                        status="done",
                    )

                except Exception:

                    # ------------------------------------------------
                    # Mark document as failed
                    # ------------------------------------------------

                    finish_document(
                        doc_id,
                        filename,
                        user_id,
                        uploaded_at,
                        0,
                        status="failed",
                    )

                    raise

        # --------------------------------------------------------
        # Upload completed
        # --------------------------------------------------------

        await cl.Message(
            content=(
                f"Ingested {len(message.elements)} file(s). "
                "Ask away.\n\n"
                "📚 [View documents](/documents)"
            )
        ).send()

        return

    # --------------------------------------------------------
    # Documents shortcut
    #
    # The actual Documents screen is now available through
    # the persistent Documents button in the sidebar.
    #
    # /docs is kept as a simple redirect for convenience.
    # --------------------------------------------------------

    if message.content.strip().lower() == "/docs":

        await cl.Message(
            content="📚 [Open Documents](/documents)"
        ).send()

        return

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