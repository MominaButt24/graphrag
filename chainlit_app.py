import wave
import tempfile
import re

import uuid
import os
import chainlit as cl
import chainlit.data as cl_data
import asyncio
import json
import redis.asyncio as aioredis

from src.voice.stt import transcribe_audio
from src.voice.tts import synthesize_speech

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
    # Voice audio buffer
    # --------------------------------------------------------

    cl.user_session.set(
        "audio_buffer",
        bytearray()
    )

    cl.user_session.set(
        "audio_mime",
        None
    )
    # --------------------------------------------------------
    # NOTE:
    # Thread is intentionally NOT persisted here anymore.
    #
    # The thread only gets created/titled in ge,
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



#-------------- plan show streamimg ---------------
def plan_channel(run_id: str) -> str:
    return f"graphrag:plan:{run_id}"


async def update_plan_message(plan_msg: cl.Message, content: str):
    """Update a Chainlit message using the installed Chainlit API."""
    plan_msg.content = content
    await plan_msg.update()


async def subscribe_plan_events(run_id: str, plan_msg: cl.Message):
    redis_client = aioredis.from_url(
        os.getenv("REDIS_URL", "redis://localhost:6379/0"),
        decode_responses=True,
    )

    pubsub = redis_client.pubsub()

    await pubsub.subscribe(plan_channel(run_id))

    try:
        async for message in pubsub.listen():

            if message["type"] != "message":
                continue

            event = json.loads(message["data"])
            event_type = event["type"]

            if event_type == "plan_created":
                tasks = event["plan"]["tasks"]

                content = "### 📋 Plan\n\n"
                content += "\n".join(
                    f"⏳ {task['description']}"
                    for task in tasks
                )

                await update_plan_message(plan_msg, content)

            elif event_type == "task_started":
                await update_plan_message(
                    plan_msg,
                    f"### 📋 Plan\n\n"
                    f"🔄 **Task {event['task_id']}** — "
                    f"{event['description']}"
                )

            elif event_type == "task_completed":
                await update_plan_message(
                    plan_msg,
                    f"### 📋 Plan\n\n"
                    f"✅ **Task {event['task_id']}** — "
                    f"{event['description']}"
                )

            elif event_type == "plan_updated":
                tasks = event["plan"]["tasks"]

                content = "### 📋 Plan\n\n"

                for task in tasks:
                    if task["status"] == "completed":
                        icon = "✅"
                    elif task["status"] == "in_progress":
                        icon = "🔄"
                    elif task["status"] == "failed":
                        icon = "❌"
                    else:
                        icon = "⏳"

                    content += (
                        f"{icon} **Task {task['id']}** — "
                        f"{task['description']}\n"
                    )

                await update_plan_message(plan_msg, content)

            elif event_type == "task_failed":
                await update_plan_message(
                    plan_msg,
                    f"### 📋 Plan\n\n"
                    f"❌ **Task {event['task_id']} failed** — "
                    f"{event['description']}"
                )

            elif event_type == "finished":
                tasks = event["plan"]["tasks"]

                content = "### 📋 Plan\n\n"

                for task in tasks:
                    content += (
                        f"✅ **Task {task['id']}** — "
                        f"{task['description']}\n"
                    )

                await update_plan_message(plan_msg, content)

                break

    finally:
        await pubsub.unsubscribe(plan_channel(run_id))
        await pubsub.close()
        await redis_client.close()


# ============================================================
# VOICE INPUT
# ============================================================

def clean_for_tts(text: str) -> str:
    """Convert Markdown-formatted answer into natural speech text."""

    # Remove fenced code blocks completely
    text = re.sub(
        r"```.*?```",
        "",
        text,
        flags=re.DOTALL,
    )

    # Markdown headings
    text = re.sub(
        r"^#{1,6}\s*",
        "",
        text,
        flags=re.MULTILINE,
    )

    # Bold
    text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)

    # Italic
    text = re.sub(r"\*(.*?)\*", r"\1", text)

    # Underscore bold / italic
    text = re.sub(r"__(.*?)__", r"\1", text)
    text = re.sub(r"_(.*?)_", r"\1", text)

    # Bullet points
    text = re.sub(
        r"^\s*[-*+]\s+",
        "",
        text,
        flags=re.MULTILINE,
    )

    # Numbered lists
    text = re.sub(
        r"^\s*\d+\.\s+",
        "",
        text,
        flags=re.MULTILINE,
    )

    # Markdown links: [text](url) → text
    text = re.sub(
        r"\[([^\]]+)\]\([^)]+\)",
        r"\1",
        text,
    )

    # Inline code
    text = re.sub(
        r"`([^`]+)`",
        r"\1",
        text,
    )

    # Remove remaining Markdown formatting characters
    text = re.sub(
        r"[#*_~]",
        "",
        text,
    )

    # Clean excessive whitespace
    text = re.sub(
        r"\n{3,}",
        "\n\n",
        text,
    )

    text = re.sub(
        r"[ \t]+",
        " ",
        text,
    )

    return text.strip()


@cl.on_audio_start
async def on_audio_start():
    print("[VOICE] Audio recording started")

    cl.user_session.set(
        "audio_buffer",
        bytearray()
    )

    cl.user_session.set(
        "audio_mime",
        None
    )

    return True

@cl.on_audio_chunk
async def on_audio_chunk(chunk: cl.InputAudioChunk):

    audio_buffer = cl.user_session.get("audio_buffer")

    if audio_buffer is None:
        audio_buffer = bytearray()

    # New recording
    if chunk.isStart:
        audio_buffer = bytearray()

        cl.user_session.set(
            "audio_mime",
            chunk.mimeType
        )

        print(
            f"[VOICE] Recording started | "
            f"mime={chunk.mimeType}"
        )

    # Add this chunk's audio bytes
    audio_buffer.extend(chunk.data)

    cl.user_session.set(
        "audio_buffer",
        audio_buffer
    )

    print(
        f"[VOICE] chunk received | "
        f"elapsed={chunk.elapsedTime:.2f}s | "
        f"bytes={len(chunk.data)} | "
        f"total={len(audio_buffer)}"
    )

@cl.on_audio_end
async def on_audio_end():

    audio_buffer = cl.user_session.get("audio_buffer")

    if not audio_buffer:
        print("[VOICE] No audio received")
        return

    print(
        f"[VOICE] Recording ended | "
        f"total bytes={len(audio_buffer)}"
    )

    with tempfile.NamedTemporaryFile(
        suffix=".wav",
        delete=False,
    ) as temp_file:

        audio_path = temp_file.name

    with wave.open(audio_path, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(24000)
        wav_file.writeframes(bytes(audio_buffer))

    print(f"[VOICE] WAV saved: {audio_path}")

    try:
        transcript = await asyncio.to_thread(
            transcribe_audio,
            audio_path,
        )

        print(f"[VOICE] Transcript: {transcript}")

        if not transcript.strip():
            await cl.Message(
                content="🎤 I couldn't detect any speech."
            ).send()
            return

        await cl.Message(
            content=transcript.strip(),
            author="You",
            type="user_message",
        ).send()

        cl.user_session.set(
            "voice_query",
            True
        )

        await process_query(
            transcript.strip()
        )

    except Exception as exc:

        print(f"[VOICE] STT error: {exc}")

        await cl.Message(
            content=f"❌ Voice transcription failed: {exc}"
        ).send()
    


async def process_query(query: str):
    thread_id = cl.user_session.get("thread_id")
    user_id = cl.user_session.get("user_id")

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

    if is_first_message:
        title = (
            query[:60]
            + ("..." if len(query) > 60 else "")
        )

        await cl_data.get_data_layer().update_thread(
            thread_id=thread_id,
            name=title,
            user_id=user_id,
        )

    run_id = str(uuid.uuid4())

    plan_msg = cl.Message(
        content="### 📋 Plan\n\n⏳ Creating plan..."
    )

    await plan_msg.send()

    subscriber_task = asyncio.create_task(
        subscribe_plan_events(
            run_id,
            plan_msg,
        )
    )

    try:
        result = await asyncio.to_thread(
            answer_query,
            thread_id,
            user_id,
            query,
            run_id,
        )
    finally:
        subscriber_task.cancel()

        try:
            await subscriber_task
        except asyncio.CancelledError:
            pass

    answer = result["answer"]

    is_voice_query = cl.user_session.get(
        "voice_query",
        False,
    )

    if is_voice_query:

        try:
            tts_text = clean_for_tts(answer)

            print(f"[VOICE] TTS text: {tts_text}")

            audio_bytes = await asyncio.to_thread(
                synthesize_speech,
                tts_text,
            )

            await cl.Message(
                content=answer,
                elements=[
                    cl.Audio(
                        name="answer.mp3",
                        content=audio_bytes,
                        mime="audio/mpeg",
                        display="inline",
                        auto_play=True,
                    )
                ],
            ).send()

        except Exception as exc:

            print(f"[VOICE] TTS error: {exc}")

            # Even if TTS fails, don't lose the text answer.
            await cl.Message(
                content=answer
            ).send()

    else:

        await cl.Message(
            content=answer
        ).send()

    explorer_content = format_retrieval_explorer(
        result.get("retrieval")
    )

    if explorer_content:
        await cl.Message(
            content=explorer_content
        ).send()

# ============================================================
# PER MESSAGE
# ============================================================
@cl.on_message
async def on_message(message: cl.Message):

    if message.content.strip().lower() == "/docs":
        await cl.Message(
            content="📚 [Open Documents](/documents)"
        ).send()
        return

    cl.user_session.set(
        "voice_query",
        False
    )

    await process_query(
        message.content.strip()
    )


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
# ---------------------------