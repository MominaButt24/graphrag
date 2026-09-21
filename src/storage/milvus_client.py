from pymilvus import connections, utility, Collection, FieldSchema, CollectionSchema, DataType
from sentence_transformers import SentenceTransformer

from src.config.settings import settings

COLLECTION_NAME = "graphrag_documents"
EMBED_DIM = 384

_collection = None  
_embedder = None


def get_embedder():
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer("all-MiniLM-L6-v2")
    return _embedder


def get_collection():
    global _collection
    if _collection is None:
        connections.connect(
            alias="default",
            uri=settings.milvus_uri,
            token=settings.milvus_token,
        )
        if utility.has_collection(COLLECTION_NAME):
            _collection = Collection(COLLECTION_NAME)
        else:
            fields = [
                FieldSchema(
                    name="id",
                    dtype=DataType.INT64,
                    is_primary=True,
                    auto_id=True,
                ),

                FieldSchema(
                    name="document_id",
                    dtype=DataType.VARCHAR,
                    max_length=64,
                ),

                FieldSchema(
                    name="chunk_id",
                    dtype=DataType.VARCHAR,
                    max_length=128,
                ),

                FieldSchema(
                    name="filename",
                    dtype=DataType.VARCHAR,
                    max_length=512,
                ),

                FieldSchema(
                    name="source_key",
                    dtype=DataType.VARCHAR,
                    max_length=1024,
                ),

                FieldSchema(
                    name="page_number",
                    dtype=DataType.INT64,
                ),

                FieldSchema(
                    name="text",
                    dtype=DataType.VARCHAR,
                    max_length=8192,
                ),

                FieldSchema(
                    name="embedding",
                    dtype=DataType.FLOAT_VECTOR,
                    dim=EMBED_DIM,
                ),
            ]
            schema = CollectionSchema(fields, description="GraphRAG hybrid vector store")
            _collection = Collection(COLLECTION_NAME, schema)
            _collection.create_index(
                field_name="embedding",
                index_params={
                    "index_type": "HNSW",
                    "metric_type": "COSINE",
                    "params": {"M": 16, "efConstruction": 200},
                },
            )
    return _collection


def delete_document_chunks(document_id: str):
    """Delete every chunk row in the vector collection for one document id."""
    collection = get_collection()
    collection.delete(expr=f'document_id == "{document_id}"')
    collection.flush()


def close_collection():
    global _collection
    _collection = None