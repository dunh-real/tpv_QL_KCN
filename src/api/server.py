import os

from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware

from src.api.routes.v1.chat import router as chat_router_v1
from src.api.routes.v1.ocr import router as ocr_router_v1
from src.api.routes.v1.ingest import router as ingest_router_v1
from src.core.logger import get_logger
from src.api.security import verify_api_key, rate_limiter

logger = get_logger(__name__)


def create_app() -> FastAPI:
    app = FastAPI(
        title="Industrial Park Management API",
        description="API chuẩn production cho ứng dụng hỏi đáp quản lý khu công nghiệp",
        version="1.0.0",
    )

    allowed_origins = os.getenv("CORS_ORIGINS", "*").split(",")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(
        chat_router_v1, 
        prefix="/api/v1/chat", 
        tags=["Chat"],
        dependencies=[Depends(verify_api_key), Depends(rate_limiter)]
    )
    app.include_router(
        ocr_router_v1, 
        prefix="/api/v1/ocr", 
        tags=["OCR"],
        dependencies=[Depends(verify_api_key), Depends(rate_limiter)]
    )
    app.include_router(
        ingest_router_v1,
        prefix="/api/v1/ingest",
        tags=["Ingest"],
        dependencies=[Depends(verify_api_key), Depends(rate_limiter)]
    )

    @app.get("/health", tags=["Health"])
    async def health_check():
        return {"status": "ok"}

    logger.info("[server] FastAPI app đã được khởi tạo.")
    return app


app = create_app()
