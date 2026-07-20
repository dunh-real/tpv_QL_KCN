from celery.result import AsyncResult
from fastapi import APIRouter

from src.api.schemas.ingest import (
    IngestQueueResponse,
    IngestRequest,
    IngestStatusResponse,
)
from src.core.logger import get_logger
from src.workers.celery_app import celery_app
from src.workers.tasks import ingest_document

router = APIRouter()
logger = get_logger(__name__)


@router.post("/", response_model=IngestQueueResponse, status_code=202)
async def start_ingest(request: IngestRequest):
    """Submit a document for background ingestion (chunking + VectorDB)."""
    task = ingest_document.delay(str(request.file_url), request.filename)
    logger.info(f"Ingest task queued. Task ID: {task.id}")
    return IngestQueueResponse(task_id=task.id)


@router.get("/{task_id}", response_model=IngestStatusResponse)
async def get_ingest_status(task_id: str):
    """Check the current status of an ingestion task."""
    task_result = AsyncResult(task_id, app=celery_app)

    response = IngestStatusResponse(task_id=task_id, status=task_result.status)

    if task_result.status == "PROCESSING":
        meta = task_result.info or {}
        response.progress = meta.get("progress")
        response.step = meta.get("step")

    elif task_result.status == "SUCCESS":
        task_data = task_result.result or {}
        response.result = {
            "parent_chunks": task_data.get("parent_chunks"),
            "children_chunks": task_data.get("children_chunks"),
        }

    elif task_result.status == "FAILURE":
        response.error = str(task_result.result)

    return response
