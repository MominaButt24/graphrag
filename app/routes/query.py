from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from src.services.query_service import (
    query_graph,
    query_hybrid,
    query_agent,
)

router = APIRouter()


class Query(BaseModel):
    question: str


@router.post("/query")
def query(payload: Query):
    """Original graph-only path — kept as-is so you can compare against /query/hybrid."""
    if not payload.question.strip():
        raise HTTPException(status_code=400, detail="question cannot be empty")
    return {"answer": query_graph(payload.question)}


@router.post("/query/hybrid")
def query_hybrid(payload: Query):
    """Phase 1 + 2: graph + vector search in parallel, reranked, one synthesized answer."""
    if not payload.question.strip():
        raise HTTPException(status_code=400, detail="question cannot be empty")
    return {"answer": query_hybrid(payload.question)}


@router.post("/query/agent")
def query_agent(payload: Query):
    """Phase 3: agent decides between the hybrid KB tool and Tavily, traced via Langfuse."""
    if not payload.question.strip():
        raise HTTPException(status_code=400, detail="question cannot be empty")
    return {"answer": query_agent(payload.question)}
