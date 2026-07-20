import asyncio
from typing import Callable, Optional

import requests
from langchain_core.documents import Document
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from src.core.logger import get_logger
from src.services.chunking_service import get_chunking_service
from src.services.ingestion_service import get_ingestion_service

logger = get_logger(__name__)


def _build_http_session() -> requests.Session:
    """Tạo HTTP session với retry tự động (3 lần, backoff 1s)."""
    session = requests.Session()
    retry = Retry(
        total=3,
        backoff_factor=1,
        status_forcelist=[502, 503, 504],
    )
    session.mount("http://", HTTPAdapter(max_retries=retry))
    session.mount("https://", HTTPAdapter(max_retries=retry))
    return session


def ingest_file(
    file_url: str,
    file_name: str,
    progress_callback: Optional[Callable[[str, int], None]] = None,
) -> dict:
    """
    Pipeline đồng bộ: Download → Xoá dữ liệu cũ → Chunking → Ingest.
    Được thiết kế để chạy tối ưu trong Celery worker.

    Args:
        file_url:  Presigned URL của file markdown trên MinIO.
        file_name: Tên file.
        progress_callback: Optional callback(step, progress_percent)
                           cho Celery update_state().

    Returns:
        dict: Thống kê kết quả ingest (parent_chunks, children_chunks).
    """
    logger.info(f"[Ingest Pipeline] Bắt đầu xử lý file: {file_name}")

    #Bước 1: Download file từ MinIO 
    if progress_callback:
        progress_callback("Downloading", 5)

    session = _build_http_session()
    try:
        response = session.get(file_url, timeout=60)
        response.raise_for_status()
    except requests.exceptions.RequestException as req_err:
        logger.error(f"Failed to download file from presigned URL: {req_err}")
        raise

    content = response.text
    logger.info(
        f"[Ingest Pipeline] Downloaded {len(response.content) / 1024:.1f} KB."
    )

    doc = Document(
        page_content=content,
        metadata={"source": file_url, "filename": file_name},
    )

    # Bước 2: Xoá dữ liệu cũ của file 
    if progress_callback:
        progress_callback("Cleaning old data", 15)

    ingestion_service = get_ingestion_service()
    asyncio.run(ingestion_service.delete_by_filename(file_name))
    logger.info(f"[Ingest Pipeline] Đã xoá dữ liệu cũ của '{file_name}'.")

    # Bước 3: Chunking 
    if progress_callback:
        progress_callback("Chunking", 25)

    chunking_service = get_chunking_service()
    parent_chunks, children_chunks = chunking_service.process_and_split(doc)

    logger.info(
        f"[Ingest Pipeline] Chunking xong: "
        f"{len(parent_chunks)} parents, {len(children_chunks)} children."
    )

    if not parent_chunks and not children_chunks:
        logger.warning("Không có chunk nào được tạo ra từ tài liệu này.")
        return {"parent_chunks": 0, "children_chunks": 0}

    # Bước 4: Ingest vào MongoDB + Qdrant 
    if progress_callback:
        progress_callback("Ingesting", 50)

    result = asyncio.run(
        ingestion_service.ingest(parent_chunks, children_chunks)
    )

    if progress_callback:
        progress_callback("Completed", 100)

    logger.info(f"[Ingest Pipeline] Hoàn tất ingest '{file_name}': {result}")
    return result