from contextlib import asynccontextmanager

import sentry_sdk
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sentry_sdk.integrations.fastapi import FastApiIntegration

from src.config.settings import settings
from src.storage.neo4j_client import close_driver
from src.storage.milvus_client import close_collection
from src.config.my_logging import setup_logging, get_logger

from app.routes import router


setup_logging()
logger = get_logger(__name__)

if settings.sentry_dsn:
    sentry_sdk.init(
        dsn=settings.sentry_dsn,
        environment=settings.sentry_environment,
        integrations=[FastApiIntegration()],
        traces_sample_rate=1.0,
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        yield
    finally:
        close_driver()
        close_collection()


def create_app() -> FastAPI:
    api = FastAPI(
        title="GraphRAG API",
        lifespan=lifespan,
    )

    api.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://localhost:7860",
            "http://127.0.0.1:7860",
            "http://localhost:8001",
            "http://localhost:8002",
            "http://127.0.0.1:8002",
        ],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Keep all existing GraphRAG API endpoints.
    api.include_router(router)

    logger.info("GraphRAG API configured")
    return api


app = create_app()

