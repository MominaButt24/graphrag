from fastapi import HTTPException
from neo4j.exceptions import ServiceUnavailable as Neo4jUnavailable
from pymilvus import MilvusException
from openai import APIConnectionError, APIError

from src.config.logging import get_logger

logger = get_logger(__name__)


def map_pipeline_error(e: Exception) -> HTTPException:
    """Translate internal exceptions into clean HTTP responses."""
    if isinstance(e, (APIConnectionError, ConnectionError, TimeoutError)):
        logger.error(f"LLM backend unreachable: {e}")
        return HTTPException(
            status_code=503,
            detail="The LLM backend is currently unreachable. Please try again shortly.",
        )
    if isinstance(e, Neo4jUnavailable):
        logger.error(f"Neo4j unreachable: {e}")
        return HTTPException(status_code=503, detail="The graph database is currently unreachable.")
    if isinstance(e, MilvusException):
        logger.error(f"Milvus unreachable: {e}")
        return HTTPException(status_code=503, detail="The vector database is currently unreachable.")
    if isinstance(e, APIError):
        logger.error(f"LLM API error: {e}")
        return HTTPException(status_code=502, detail="The LLM backend returned an error.")
    logger.exception("Unexpected error in pipeline")
    return HTTPException(status_code=500, detail="Internal server error.")
