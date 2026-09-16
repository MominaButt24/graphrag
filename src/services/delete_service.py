from src.config.logging import get_logger
from src.services.errors import map_pipeline_error
from src.storage.milvus_client import delete_document_chunks
from src.storage.milvus_data_layer import delete_document as delete_metadata_document, list_documents
from src.storage.neo4j_client import delete_graph_document
from src.storage.storage import delete_document_files

logger = get_logger(__name__)


def delete_document(document_id: str):
    """Delete one uploaded file from the object store, vector wallet, metadata layer, and graph layer."""
    try:
        delete_graph_document(document_id)
        delete_document_files(document_id)
        delete_document_chunks(document_id)
        delete_metadata_document(document_id)

        logger.info(f"[/documents] deleted document_id='{document_id}'")
        return {"status": "deleted", "document_id": document_id}
    except Exception as e:
        raise map_pipeline_error(e)
