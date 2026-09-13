from src.config.logging import get_logger

from src.generation.llm import get_llm

logger = get_logger(__name__)

llm = get_llm()

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
