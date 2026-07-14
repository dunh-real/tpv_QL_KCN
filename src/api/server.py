import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.routes.v1.chat import router as chat_router_v1
from src.core.logger import get_logger

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

    app.include_router(chat_router_v1, prefix="/api/v1/chat", tags=["Chat"])

    @app.get("/health", tags=["Health"])
    async def health_check():
        return {"status": "ok"}

    logger.info("[server] FastAPI app đã được khởi tạo.")
    return app


app = create_app()
