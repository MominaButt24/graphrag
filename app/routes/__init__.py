from fastapi import APIRouter

from app.routes.query import router as query_router
from app.routes.upload import router as upload_router
from app.routes.documents import router as documents_router
from app.routes.health import router as health_router
from app.routes.view import router as view_router
from app.routes.delete import router as delete_router

router = APIRouter(prefix="/api/v1")

router.include_router(query_router)
router.include_router(upload_router)
router.include_router(documents_router)
router.include_router(health_router)
router.include_router(view_router)
router.include_router(delete_router)