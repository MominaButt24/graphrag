# -----------------------------
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
    # Create a new thread ID
    # --------------------------------------------------------

    # thread_id = str(uuid.uuid4())
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
        # IMPORTANT:
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
    # Create thread
    #
    # IMPORTANT:
    # user_id is now the UUID, NOT "momna".
    # --------------------------------------------------------

    await cl_data.get_data_layer().update_thread(
        thread_id=thread_id,
        name="New chat",
        user_id=user_id,
    )

    await cl.Message(
        content="Hi! Ask me anything."
    ).send()


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

        from src.ingest import ingest_file_to_milvus

        for element in message.elements:

            if hasattr(element, "path"):
                ingest_file_to_milvus(
                    element.path
                )

        await cl.Message(
            content=f"Ingested {len(message.elements)} file(s). Ask away."
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
            + ("..." if len(message.content) > 60 else "")
        )

        await cl_data.get_data_layer().update_thread(
            thread_id=thread_id,
            name=title,
            user_id=user_id,
        )

    # --------------------------------------------------------
    # Generate answer
    # --------------------------------------------------------

    answer = answer_query(
        thread_id,
        user_id,
        message.content,
    )

    await cl.Message(
        content=answer
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