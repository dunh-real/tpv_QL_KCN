from datetime import timedelta
import asyncio

from celery.result import AsyncResult
from fastapi import APIRouter

from src.api.schemas.ocr import OCRQueueResponse, OCRRequest, OCRStatusResponse
from src.core.config import settings
from src.core.logger import get_logger
from src.core.ocr_pipeline import get_minio_client
from src.workers.celery_app import celery_app
from src.workers.tasks import process_document

router = APIRouter()
logger = get_logger(__name__)


@router.post("/", response_model=OCRQueueResponse, status_code=202)
async def start_ocr(request: OCRRequest):
    """Submit a document for background OCR processing."""
    task = process_document.delay(str(request.file_url), request.filename)
    logger.info(f"OCR task queued. Task ID: {task.id}")
    return OCRQueueResponse(task_id=task.id)


@router.get("/{task_id}", response_model=OCRStatusResponse)
async def get_task_status(task_id: str):
    """Check the current status of an OCR task."""
    task_result = AsyncResult(task_id, app=celery_app)

    response = OCRStatusResponse(task_id=task_id, status=task_result.status)

    if task_result.status == "PROCESSING":
        meta = task_result.info or {}
        response.progress = meta.get("progress")
        response.step = meta.get("step")

    elif task_result.status == "SUCCESS":
        task_data = task_result.result or {}
        result_filename = task_data.get("result_filename")
        response.result_file = result_filename

        if result_filename:
            try:
                minio_client = get_minio_client()
                download_url = await asyncio.to_thread(
                    minio_client.presigned_get_object,
                    settings.MINIO_BUCKET,
                    result_filename,
                    timedelta(hours=1),
                )
                response.download_url = download_url
            except Exception as e:
                logger.error(f"Failed to generate MinIO presigned URL: {e}")
                response.error = "Failed to generate download link."

    elif task_result.status == "FAILURE":
        response.error = str(task_result.result)

    return response