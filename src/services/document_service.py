from src.config.logging import get_logger
from src.storage.milvus_data_layer import list_documents


logger = get_logger(__name__)


def get_documents():
    return {"documents": list_documents()}
