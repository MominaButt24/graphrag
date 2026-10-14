
from contextlib import asynccontextmanager

import sentry_sdk
import gradio as gr
from sentry_sdk.integrations.fastapi import FastApiIntegration
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.config.settings import settings
from src.storage.neo4j_client import close_driver
from src.storage.milvus_client import close_collection
from src.config.logging import setup_logging, get_logger

from app.routes import router
from app.gradio_ui import demo


setup_logging()
logger = get_logger(__name__)


# Sentry initialization
if settings.sentry_dsn:
    sentry_sdk.init(
        dsn=settings.sentry_dsn,
        environment=settings.sentry_environment,
        integrations=[FastApiIntegration()],
        traces_sample_rate=1.0,
    )


# Application lifespan
@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    close_driver()
    close_collection()


app = FastAPI(
    title="GraphRAG API",
    lifespan=lifespan,
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8001",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Existing API endpoints
app.include_router(router)


# Mount Gradio at /gradio
app = gr.mount_gradio_app(
    app,
    demo,
    path="/gradio",
)