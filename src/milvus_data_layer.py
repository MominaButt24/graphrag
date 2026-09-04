# -----------------------
import os
import time
import uuid as uuid_lib

from pymilvus import (
    connections,
    utility,
    Collection,
    FieldSchema,
    CollectionSchema,
    DataType,
)

import chainlit.data as cl_data
import chainlit as cl

from dotenv import load_dotenv

from src.chat_history_client import get_chat_history_collection


load_dotenv()


# ============================================================
# CONFIGURATION
# ============================================================

THREADS_COLLECTION = "chat_threads"
USERS_COLLECTION = "chat_users"

# Dummy vector dimension.
# These collections are being used mainly for metadata storage.
DIM_PLACEHOLDER = 8


_threads_collection = None
_users_collection = None


# ============================================================
# THREADS COLLECTION
# ============================================================

def get_threads_collection():
    global _threads_collection

    if _threads_collection is None:

        connections.connect(
            alias="default",
            uri=os.getenv("MILVUS_URI"),
            token=os.getenv("MILVUS_TOKEN"),
        )

        if utility.has_collection(THREADS_COLLECTION):

            _threads_collection = Collection(
                THREADS_COLLECTION
            )

        else:

            fields = [
                FieldSchema(
                    name="id",
                    dtype=DataType.VARCHAR,
                    max_length=64,
                    is_primary=True,
                ),

                FieldSchema(
                    name="user_id",
                    dtype=DataType.VARCHAR,
                    max_length=64,
                ),

                FieldSchema(
                    name="name",
                    dtype=DataType.VARCHAR,
                    max_length=512,
                ),

                FieldSchema(
                    name="created_at",
                    dtype=DataType.INT64,
                ),

                FieldSchema(
                    name="embedding",
                    dtype=DataType.FLOAT_VECTOR,
                    dim=DIM_PLACEHOLDER,
                ),
            ]

            schema = CollectionSchema(
                fields,
                description="Chat thread metadata for Chainlit sidebar",
            )

            _threads_collection = Collection(
                THREADS_COLLECTION,
                schema,
            )

            _threads_collection.create_index(
                field_name="embedding",
                index_params={
                    "index_type": "FLAT",
                    "metric_type": "L2",
                    "params": {},
                },
            )

        _threads_collection.load()

    return _threads_collection


# ============================================================
# USERS COLLECTION
# ============================================================

def get_users_collection():
    global _users_collection

    if _users_collection is None:

        connections.connect(
            alias="default",
            uri=os.getenv("MILVUS_URI"),
            token=os.getenv("MILVUS_TOKEN"),
        )

        if utility.has_collection(USERS_COLLECTION):

            _users_collection = Collection(
                USERS_COLLECTION
            )

        else:

            fields = [
                FieldSchema(
                    name="id",
                    dtype=DataType.VARCHAR,
                    max_length=64,
                    is_primary=True,
                ),

                FieldSchema(
                    name="identifier",
                    dtype=DataType.VARCHAR,
                    max_length=128,
                ),

                FieldSchema(
                    name="created_at",
                    dtype=DataType.INT64,
                ),

                FieldSchema(
                    name="embedding",
                    dtype=DataType.FLOAT_VECTOR,
                    dim=DIM_PLACEHOLDER,
                ),
            ]

            schema = CollectionSchema(
                fields,
                description="Chainlit users, minimal fields",
            )

            _users_collection = Collection(
                USERS_COLLECTION,
                schema,
            )

            _users_collection.create_index(
                field_name="embedding",
                index_params={
                    "index_type": "FLAT",
                    "metric_type": "L2",
                    "params": {},
                },
            )

        _users_collection.load()

    return _users_collection


# ============================================================
# USER HELPERS
# ============================================================

async def get_user_by_identifier(identifier: str):
    """
    Find a persisted user using their Chainlit identifier.

    Example:

        identifier = "momna"

    returns something like:

        {
            "id": "3d6a028f-09ec-...",
            "identifier": "momna",
            "created_at": ...
        }
    """

    collection = get_users_collection()

    rows = collection.query(
        expr=f'identifier == "{identifier}"',
        output_fields=[
            "id",
            "identifier",
            "created_at",
        ],
        limit=1,
    )

    if not rows:
        return None

    return rows[0]


def get_user_identifier_by_id(user_id: str) -> str:
    """
    Convert persisted user UUID -> Chainlit identifier.

    Example:

        UUID
        3d6a028f-09ec-...

    becomes:

        momna

    This is needed because Chainlit's authorization check
    uses the user's identifier.

    Also supports your old data where user_id was stored
    directly as "momna".
    """

    if not user_id:
        return ""

    collection = get_users_collection()

    # --------------------------------------------------------
    # NORMAL CASE
    #
    # user_id contains the persisted UUID
    # --------------------------------------------------------

    rows = collection.query(
        expr=f'id == "{user_id}"',
        output_fields=[
            "identifier"
        ],
        limit=1,
    )

    if rows:
        return rows[0]["identifier"]

    # --------------------------------------------------------
    # BACKWARD COMPATIBILITY
    #
    # Older threads may have:
    #
    # user_id = "momna"
    #
    # instead of:
    #
    # user_id = "3d6a028f-..."
    # --------------------------------------------------------

    rows = collection.query(
        expr=f'identifier == "{user_id}"',
        output_fields=[
            "identifier"
        ],
        limit=1,
    )

    if rows:
        return rows[0]["identifier"]

    return ""


# ============================================================
# CHAINLIT DATA LAYER
# ============================================================

class MilvusDataLayer(cl_data.BaseDataLayer):

    # ========================================================
    # USER
    # ========================================================

    async def get_user(self, identifier: str):

        row = await get_user_by_identifier(
            identifier
        )

        if not row:
            return None

        return cl.PersistedUser(
            id=row["id"],
            createdAt=str(row["created_at"]),
            identifier=row["identifier"],
        )

    # ========================================================
    # CREATE USER
    # ========================================================

    async def create_user(self, user: cl.User):

        collection = get_users_collection()

        user_id = str(
            uuid_lib.uuid4()
        )

        now = int(
            time.time()
        )

        collection.upsert(
            [
                {
                    "id": user_id,
                    "identifier": user.identifier,
                    "created_at": now,
                    "embedding": [0.0] * DIM_PLACEHOLDER,
                }
            ]
        )

        collection.flush()

        return cl.PersistedUser(
            id=user_id,
            createdAt=str(now),
            identifier=user.identifier,
        )

    # ========================================================
    # DELETE USER SESSION
    # ========================================================

    async def delete_user_session(
        self,
        id: str,
    ) -> bool:

        return True

    # ========================================================
    # THREAD AUTHOR
    # ========================================================

    async def get_thread_author(
        self,
        thread_id: str,
    ) -> str:

        """
        Chainlit calls this when checking whether the
        logged-in user is allowed to access a thread.

        Chainlit expects:

            "momna"

        rather than:

            "3d6a028f-09ec-..."

        So we convert the stored UUID back to
        the user's identifier.
        """

        threads = get_threads_collection()

        rows = threads.query(
            expr=f'id == "{thread_id}"',
            output_fields=[
                "user_id"
            ],
            limit=1,
        )

        if not rows:
            return ""

        stored_user_id = rows[0]["user_id"]

        return get_user_identifier_by_id(
            stored_user_id
        )

    # ========================================================
    # GET THREAD
    # ========================================================

    async def get_thread(
        self,
        thread_id: str,
    ):

        threads = get_threads_collection()

        rows = threads.query(
            expr=f'id == "{thread_id}"',
            output_fields=[
                "id",
                "user_id",
                "name",
                "created_at",
            ],
            limit=1,
        )

        if not rows:
            return None

        thread_row = rows[0]

        # ----------------------------------------------------
        # Get messages from chat_history
        # ----------------------------------------------------

        history_collection = (
            get_chat_history_collection()
        )

        messages = history_collection.query(
            expr=f'thread_id == "{thread_id}"',
            output_fields=[
                "role",
                "content",
                "timestamp",
            ],
            limit=1000,
        )

        messages = sorted(
            messages,
            key=lambda m: m["timestamp"],
        )

        # ----------------------------------------------------
        # Convert messages into Chainlit steps
        # ----------------------------------------------------

        steps = []

        for i, message in enumerate(messages):

            steps.append(
                {
                    "id": f"{thread_id}-{i}",

                    "threadId": thread_id,

                    "type": (
                        "user_message"
                        if message["role"] == "user"
                        else "assistant_message"
                    ),

                    "output": message["content"],

                    "name": message["role"],

                    "createdAt": str(
                        message["timestamp"]
                    ),
                }
            )

        # ----------------------------------------------------
        # Convert UUID -> identifier
        # ----------------------------------------------------

        user_identifier = (
            get_user_identifier_by_id(
                thread_row["user_id"]
            )
        )

        # ----------------------------------------------------
        # Return Chainlit ThreadDict
        # ----------------------------------------------------

        return {
            "id": thread_row["id"],

            "name": thread_row["name"],

            # Persisted user UUID
            "userId": thread_row["user_id"],

            # Chainlit username / identifier
            "userIdentifier": user_identifier,

            "createdAt": str(
                thread_row["created_at"]
            ),

            "steps": steps,

            "elements": [],

            "tags": [],

            "metadata": {},
        }

    # ========================================================
    # UPDATE THREAD
    # ========================================================

    async def update_thread(
        self,
        thread_id: str,
        name: str = None,
        user_id: str = None,
        metadata: dict = None,
        tags: list = None,
    ):

        threads = get_threads_collection()

        existing = threads.query(
            expr=f'id == "{thread_id}"',
            output_fields=[
                "id",
                "user_id",
                "name",
                "created_at",
            ],
            limit=1,
        )

        existing_row = (
            existing[0]
            if existing
            else None
        )

        # ----------------------------------------------------
        # Preserve existing user ID if none was supplied
        # ----------------------------------------------------

        final_user_id = (
            user_id
            or (
                existing_row["user_id"]
                if existing_row
                else ""
            )
        )

        # ----------------------------------------------------
        # Preserve existing title
        # ----------------------------------------------------

        final_name = (
            name
            or (
                existing_row["name"]
                if existing_row
                else "New chat"
            )
        )

        # ----------------------------------------------------
        # Preserve creation timestamp
        # ----------------------------------------------------

        created_at = (
            existing_row["created_at"]
            if existing_row
            else int(time.time())
        )

        # ----------------------------------------------------
        # Store thread
        #
        # IMPORTANT:
        #
        # user_id = persisted UUID
        #
        # NOT:
        #
        # user_id = "momna"
        # ----------------------------------------------------

        row = {
            "id": thread_id,

            "user_id": final_user_id,

            "name": final_name,

            "created_at": created_at,

            "embedding": [0.0] * DIM_PLACEHOLDER,
        }

        threads.upsert(
            [row]
        )

        threads.flush()

    # ========================================================
    # DELETE THREAD
    # ========================================================

    async def delete_thread(
        self,
        thread_id: str,
    ):

        # ----------------------------------------------------
        # Delete thread metadata
        # ----------------------------------------------------

        threads = get_threads_collection()

        threads.delete(
            expr=f'id == "{thread_id}"'
        )

        # ----------------------------------------------------
        # Delete messages belonging to the thread
        # ----------------------------------------------------

        history_collection = (
            get_chat_history_collection()
        )

        history_collection.delete(
            expr=f'thread_id == "{thread_id}"'
        )

    # ========================================================
    # LIST THREADS
    # ========================================================

    async def list_threads(
        self,
        pagination,
        filters,
    ):

        threads = get_threads_collection()

        # ----------------------------------------------------
        # Chainlit supplies the persisted user UUID here.
        # ----------------------------------------------------

        user_id = getattr(
            filters,
            "userId",
            None,
        )

        if not user_id:

            return self._empty_thread_response()

        # ----------------------------------------------------
        # Get only this user's threads
        # ----------------------------------------------------

        rows = threads.query(
            expr=f'user_id == "{user_id}"',
            output_fields=[
                "id",
                "user_id",
                "name",
                "created_at",
            ],
            limit=1000,
        )

        # Newest first
        rows = sorted(
            rows,
            key=lambda r: r["created_at"],
            reverse=True,
        )

        thread_dicts = []

        for row in rows:

            user_identifier = (
                get_user_identifier_by_id(
                    row["user_id"]
                )
            )

            thread_dicts.append(
                {
                    "id": row["id"],

                    "name": row["name"],

                    # Persisted UUID
                    "userId": row["user_id"],

                    # Login identifier
                    "userIdentifier": user_identifier,

                    "createdAt": str(
                        row["created_at"]
                    ),

                    "steps": [],

                    "elements": [],

                    "tags": [],

                    "metadata": {},
                }
            )

        return self._make_thread_response(
            thread_dicts
        )

    # ========================================================
    # EMPTY THREAD RESPONSE
    # ========================================================

    @staticmethod
    def _empty_thread_response():

        from chainlit.types import (
            PaginatedResponse,
            PageInfo,
        )

        return PaginatedResponse(
            data=[],
            pageInfo=PageInfo(
                hasNextPage=False,
                startCursor=None,
                endCursor=None,
            ),
        )

    # ========================================================
    # THREAD RESPONSE
    # ========================================================

    @staticmethod
    def _make_thread_response(
        thread_dicts,
    ):

        from chainlit.types import (
            PaginatedResponse,
            PageInfo,
        )

        return PaginatedResponse(
            data=thread_dicts,
            pageInfo=PageInfo(
                hasNextPage=False,
                startCursor=None,
                endCursor=None,
            ),
        )

    # ========================================================
    # STEPS
    # ========================================================

    async def create_step(
        self,
        step_dict,
    ):
        pass

    async def update_step(
        self,
        step_dict,
    ):
        pass

    async def delete_step(
        self,
        step_id: str,
    ):
        pass

    # ========================================================
    # ELEMENTS
    # ========================================================

    async def create_element(
        self,
        element,
    ):
        pass

    async def get_element(
        self,
        thread_id: str,
        element_id: str,
    ):
        return None

    async def delete_element(
        self,
        element_id: str,
        thread_id: str = None,
    ):
        pass

    # ========================================================
    # FEEDBACK
    # ========================================================

    async def upsert_feedback(
        self,
        feedback,
    ) -> str:
        return ""

    async def delete_feedback(
        self,
        feedback_id: str,
    ) -> bool:
        return True

    # ========================================================
    # CLOSE
    # ========================================================

    async def close(self):
        pass

    # ========================================================
    # DEBUG URL
    # ========================================================

    async def build_debug_url(
        self,
    ) -> str:
        return ""

    # ========================================================
    # FAVORITE STEPS
    # ========================================================

    async def get_favorite_steps(
        self,
        *args,
        **kwargs,
    ):
        return []


# ============================================================
# REGISTER DATA LAYER
# ============================================================

cl_data._data_layer = MilvusDataLayer()