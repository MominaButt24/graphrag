from fastapi import FastAPI, UploadFile, File, APIRouter
from pydantic import BaseModel
from src.neo4j_client import close_driver
from src.retrieval import smart_query
from src.ingest import load_and_chunk
from src.graph_builder import build_graph_from_chunks
from src.community import load_graph_from_neo4j, run_leiden, write_communities_to_neo4j, build_all_community_summaries
import shutil

app = FastAPI(title="GraphRAG API")

router = APIRouter(prefix="/api/v1")

class Query(BaseModel):
    question: str

@router.post("/query")
def query(payload: Query):
    answer = smart_query(payload.question)
    return {"answer": answer}

@router.post("/upload")
def upload(file: UploadFile = File(...)):
    save_path = f"data/raw/{file.filename}"
    with open(save_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    chunks = load_and_chunk(save_path)
    failed = build_graph_from_chunks(chunks)

    G = load_graph_from_neo4j()
    community_map = run_leiden(G)
    write_communities_to_neo4j(community_map)
    build_all_community_summaries()

    return {
        "status": "processed",
        "filename": file.filename,
        "chunks": len(chunks),
        "failed_chunks": len(failed),
    }

@router.get("/health")
def health():
    return {"status": "ok"}




@app.on_event("shutdown")
def shutdown():
    close_driver()
app.include_router(router)