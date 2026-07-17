from src.core.logger import get_logger
from src.workers.celery_app import celery_app
from src.core.ocr_pipeline import OCRPipeline

logger = get_logger(__name__)

_pipeline_instance = None


def get_pipeline() -> OCRPipeline:
    """Lazy singleton — initializes OCRPipeline once per worker process."""
    global _pipeline_instance
    if _pipeline_instance is None:
        logger.info("Loading VLM model into VRAM/RAM...")
        _pipeline_instance = OCRPipeline()
    return _pipeline_instance


@celery_app.task(
    bind=True,
    name="tasks.process_document",
    autoretry_for=(ConnectionError, TimeoutError),
    retry_backoff=True,
    retry_kwargs={"max_retries": 3},
)
def process_document(self, file_url: str, filename: str):
    """
    Celery task: download a document, run OCR, and upload the result.

    Returns a dict with status and result_filename on success.
    Raises on failure so Celery marks the task as FAILURE with traceback.
    """
    try:
        pipeline = get_pipeline()

        def _on_progress(step: str, progress: int) -> None:
            self.update_state(
                state="PROCESSING",
                meta={"step": step, "progress": progress},
            )

        result_filename = pipeline.run_ocr_pipeline(
            file_url=file_url,
            file_name=filename,
            progress_callback=_on_progress,
        )

        logger.info(f"OCR task succeeded. Result: {result_filename}")
        return {
            "status": "success",
            "result_filename": result_filename,
        }

    except Exception as e:
        logger.error(f"OCR task failed: {e}", exc_info=True)
        raise