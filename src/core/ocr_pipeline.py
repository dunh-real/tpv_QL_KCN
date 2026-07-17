import io
from typing import Callable, Optional

import requests

from minio import Minio
from minio.error import S3Error

from src.core.config import settings
from src.core.logger import get_logger
from src.services.ocr_service import get_ocr_service

logger = get_logger(__name__)

_minio_client = None


def get_minio_client() -> Minio:
    """Lazy singleton for MinIO client — reused across pipeline and API."""
    global _minio_client
    if _minio_client is None:
        _minio_client = Minio(
            settings.MINIO_ENDPOINT,
            access_key=settings.MINIO_ACCESS_KEY,
            secret_key=settings.MINIO_SECRET_KEY,
            secure=settings.MINIO_SECURE,
        )
    return _minio_client


class OCRPipeline:
    def __init__(self):
        self.minio_client = get_minio_client()
        self.ocr_service = get_ocr_service()
        self.dest_bucket = settings.MINIO_BUCKET

    def _ensure_bucket_exists(self):
        """Create the destination bucket if it doesn't already exist."""
        if not self.minio_client.bucket_exists(bucket_name=self.dest_bucket):
            self.minio_client.make_bucket(bucket_name=self.dest_bucket)
            logger.info(f"Bucket '{self.dest_bucket}' created.")

    @staticmethod
    def _derive_output_name(file_name: str) -> str:
        """Convert the original filename to a .md extension."""
        if "." in file_name:
            return file_name.rsplit(".", 1)[0] + ".md"
        return file_name + ".md"

    def run_ocr_pipeline(
        self,
        file_url: str,
        file_name: str,
        progress_callback: Optional[Callable[[str, int], None]] = None,
    ) -> str:
        """
        Run the full OCR pipeline: download file → OCR → upload result to MinIO.

        Args:
            file_url: Presigned URL of the source file.
            file_name: Original filename.
            progress_callback: Optional callback(step, progress_percent)
                               for reporting pipeline progress.

        Returns:
            The object name of the uploaded markdown file in MinIO.

        Raises:
            requests.exceptions.RequestException: If file download fails.
            S3Error: If MinIO operation fails.
        """
        logger.info(f"[OCR Pipeline] Starting OCR for file: {file_name}")

        # --- Step 1: Download the source file ---
        try:
            http_response = requests.get(file_url, timeout=60)
            http_response.raise_for_status()
        except requests.exceptions.RequestException as req_err:
            logger.error(f"Failed to download file from presigned URL: {req_err}")
            raise

        pdf_bytes = http_response.content
        logger.info(f"Downloaded {len(pdf_bytes) / (1024 * 1024):.2f} MB into memory.")

        # --- Step 2: Run OCR ---
        if progress_callback:
            progress_callback("Running OCR", 0)

        # Bridge: convert per-page callback into pipeline-level progress
        def _page_progress(completed: int, total: int) -> None:
            if progress_callback:
                pct = int((completed / total) * 90)  # 0–90% for OCR phase
                progress_callback(f"OCR page {completed}/{total}", pct)

        md_content = self.ocr_service.process_file(
            pdf_input=pdf_bytes,
            progress_callback=_page_progress,
        )

        # --- Step 3: Upload result to MinIO ---
        if progress_callback:
            progress_callback("Uploading result", 95)

        self._ensure_bucket_exists()

        dest_object_name = self._derive_output_name(file_name)
        md_bytes = md_content.encode("utf-8")
        md_stream = io.BytesIO(md_bytes)

        try:
            self.minio_client.put_object(
                bucket_name=self.dest_bucket,
                object_name=dest_object_name,
                data=md_stream,
                length=len(md_bytes),
                content_type="text/markdown",
            )
        except S3Error as s3_err:
            logger.error(f"MinIO upload failed: {s3_err}")
            raise

        logger.info(f"Uploaded result to {self.dest_bucket}/{dest_object_name}")
        return dest_object_name