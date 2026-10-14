from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from src.services.query_service import (
    query_graph,
    query_hybrid as run_hybrid_query,
    query_agent as run_agent_query,
)

router = APIRouter()


class Query(BaseModel):
    question: str


@router.post("/query")
def query(payload: Query):
    if not payload.question.strip():
        raise HTTPException(status_code=400, detail="question cannot be empty")
    return {"answer": query_graph(payload.question)}


@router.post("/query/hybrid")
def query_hybrid(payload: Query):
    if not payload.question.strip():
        raise HTTPException(status_code=400, detail="question cannot be empty")
    return {"answer": run_hybrid_query(payload.question)}


@router.post("/query/agent")
def query_agent(payload: Query):
    if not payload.question.strip():
        raise HTTPException(status_code=400, detail="question cannot be empty")
    return {"answer": run_agent_query(payload.question)}
