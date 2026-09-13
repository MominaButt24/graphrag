from src.logger_config import get_logger
from langchain.chat_models import init_chat_model
import os
from dotenv import load_dotenv

load_dotenv()

logger = get_logger(__name__)

llm = init_chat_model(
    model=os.getenv("LLM_MODEL"),
    model_provider="openai",
    api_key=os.getenv("LLM_API_KEY"),
    base_url=os.getenv("LLM_BASE_URL"),
    max_tokens=int(os.getenv("LLM_MAX_TOKENS")),
)

from threading import Lock


_retrieval_metadata = None
_retrieval_metadata_lock = Lock()


def get_retrieval_metadata():
    with _retrieval_metadata_lock:
        return _retrieval_metadata


def _set_retrieval_metadata(value):
    global _retrieval_metadata
    with _retrieval_metadata_lock:
        _retrieval_metadata = value
