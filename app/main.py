import os
import shutil
from contextlib import asynccontextmanager
from dotenv import load_dotenv
import sentry_sdk
from sentry_sdk.integrations.fastapi import FastApiIntegration
from fastapi import FastAPI, UploadFile, File, APIRouter, HTTPException
from pydantic import BaseModel
from neo4j.exceptions import ServiceUnavailable as Neo4jUnavailable
from pymilvus import MilvusException
from openai import APIConnectionError, APIError

from src.neo4j_client import close_driver
from src.milvus_client import close_collection
from src.retrieval import smart_query, hybrid_answer
from src.agent import run_agent
from src.ingest import load_and_chunk, embed_and_ingest
from src.graph_builder import build_graph_from_chunks
from src.community import (
    load_graph_from_neo4j,
    run_leiden,
    write_communities_to_neo4j,
    build_all_community_summaries,
)
from src.logger_config import setup_logging, get_logger

load_dotenv()

setup_logging()
logger = get_logger(__name__)

# --- SAFE SENTRY INITIALIZATION ---
sentry_dsn = os.getenv("SENTRY_DSN", "").strip()
if sentry_dsn:
    sentry_sdk.init(
        dsn=sentry_dsn,
        environment=os.getenv("SENTRY_ENVIRONMENT", "development"),
        integrations=[FastApiIntegration()],
        traces_sample_rate=1.0,
    )


# --- MODERN LIFESPAN MANAGEMENT ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    close_driver()
    close_collection()


app = FastAPI(title="GraphRAG API", lifespan=lifespan)
router = APIRouter(prefix="/api/v1")


class Query(BaseModel):
    question: str


def _map_pipeline_error(e: Exception) -> HTTPException:
    """
    Translate internal exceptions into clean, specific HTTP responses
    instead of letting a raw 500 + stack trace reach the Gradio app (or
    any other caller). This is what actually matters when the LLM
    backend, Neo4j, or Milvus goes down mid-request. the caller gets a
    clear reason instead of a generic crash.
    """
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


@router.post("/query")
def query(payload: Query):
    """Original graph-only path — kept as-is so you can compare against /query/hybrid."""
    if not payload.question.strip():
        raise HTTPException(status_code=400, detail="question cannot be empty")
    logger.info(f"[/query] question='{payload.question}'")
    try:
        answer = smart_query(payload.question)
    except Exception as e:
        raise _map_pipeline_error(e)
    return {"answer": answer}


@router.post("/query/hybrid")
def query_hybrid(payload: Query):
    """Phase 1 + 2: graph + vector search in parallel, reranked, one synthesized answer."""
    if not payload.question.strip():
        raise HTTPException(status_code=400, detail="question cannot be empty")
    logger.info(f"[/query/hybrid] question='{payload.question}'")
    try:
        answer = hybrid_answer(payload.question)
    except Exception as e:
        raise _map_pipeline_error(e)
    return {"answer": answer}


@router.post("/query/agent")
def query_agent(payload: Query):
    """Phase 3: agent decides between the hybrid KB tool and Tavily, traced via Langfuse."""
    if not payload.question.strip():
        raise HTTPException(status_code=400, detail="question cannot be empty")
    logger.info(f"[/query/agent] question='{payload.question}'")
    try:
        answer = run_agent(payload.question)
    except Exception as e:
        raise _map_pipeline_error(e)
    return {"answer": answer}


@router.post("/upload")
def upload(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    logger.info(f"[/upload] filename='{file.filename}'")
    os.makedirs("data/raw", exist_ok=True)
    save_path = f"data/raw/{file.filename}"

    try:
        with open(save_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
    except Exception as e:
        logger.exception("Failed to save uploaded file")
        raise HTTPException(status_code=500, detail="Failed to save the uploaded file.")

    try:
        chunks = load_and_chunk(save_path)
    except Exception as e:
        raise _map_pipeline_error(e)

    try:
        failed = build_graph_from_chunks(chunks)
    except Exception as e:
        raise _map_pipeline_error(e)

    try:
        # Same chunks now also go into Milvus, so every upload keeps both
        # stores in sync instead of only the graph getting populated.
        embed_and_ingest(chunks, source=save_path)
    except Exception as e:
        raise _map_pipeline_error(e)

    try:
        G = load_graph_from_neo4j()
        community_map = run_leiden(G)
        write_communities_to_neo4j(community_map)
        build_all_community_summaries()
    except Exception as e:
        raise _map_pipeline_error(e)

    return {
        "status": "processed",
        "filename": file.filename,
        "chunks": len(chunks),
        "failed_chunks": len(failed),
    }


@router.get("/health")
def health():
    return {"status": "ok"}

# @app.get("/sentry-debug")
# async def trigger_error():
#     division_by_zero = 1 / 0
app.include_router(router)