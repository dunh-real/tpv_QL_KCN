import os
from datetime import datetime
from pathlib import Path

from bson import ObjectId
from src.core.config import settings
from src.core.logger import get_logger
from src.db.mongodb import docs_collection
from src.services.chunking import ChunkingService
from src.services.extractor import StrictDocumentExtractor
from src.worker.celery_app import celery_app

logger = get_logger(__name__)

_services: dict = {}

def _get_services():
    """Trả về services đã khởi tạo cho process hiện tại. Tái khởi tạo nếu bị fork."""
    global _services
    current_pid = os.getpid()
    if _services.get("pid") != current_pid:
        logger.info(f"[PID {current_pid}] Khởi tạo services cho worker process...")
        from src.db.hyperspace import EnterpriseDocumentStore
        _services = {
            "pid": current_pid,
            "vector_store": EnterpriseDocumentStore(
                settings.HYPERSPACE_HOST,
                settings.HYPERSPACE_API_KEY,
                settings.HYPERSPACE_COLLECTION_NAME
            ),
            "doc_extractor": StrictDocumentExtractor(),
            "chunking_service": ChunkingService(),
        }
        logger.info(f"[PID {current_pid}] Tất cả services đã sẵn sàng.")
    return _services["vector_store"], _services["doc_extractor"], _services["chunking_service"]


@celery_app.task(name="extract_document", bind=True, max_retries=3)
def process_document_task(self, file_path_str: str, doc_id_str: str, original_filename: str):
    file_path = Path(file_path_str)

    # Lấy services đã được khởi tạo cho process hiện tại
    vector_store, doc_extractor, chunking_service = _get_services()

    try:
        logger.info(f"[{doc_id_str}] Bước 1/4: Extraction...")
        extracted_pages = doc_extractor.extract(file_path)
        if not extracted_pages:
            raise ValueError("Tài liệu rỗng hoặc không thể bóc tách chữ.")

        total_pages = len(extracted_pages)
        full_markdown_context = "\n\n".join([page.page_content for page in extracted_pages])

        logger.info(f"[{doc_id_str}] Bước 2/4: Chunking...")
        base_metadata = {
            "doc_id": doc_id_str,
            "source_file": original_filename
        }
        final_chunks = chunking_service.chunk_documents(
            pages=extracted_pages,
            base_metadata=base_metadata
        )
        total_chunks = len(final_chunks)
        if total_chunks == 0:
            raise ValueError("Không tạo được chunk nào từ tài liệu này.")

        logger.info(f"[{doc_id_str}] Bước 3/4: Lưu vào Vector DB...")
        vector_store.ingest_documents(final_chunks)

        logger.info(f"[{doc_id_str}] Bước 4/4: Lưu vào MongoDB...")
        docs_collection.update_one(
            {"_id": ObjectId(doc_id_str)},
            {
                "$set": {
                    "status": "COMPLETED",
                    "context": full_markdown_context,
                    "total_pages": total_pages,
                    "total_chunks": total_chunks,
                    "completed_at": datetime.now()
                }
            }
        )
        logger.info(f"[{doc_id_str}] HOÀN TẤT! (Trang: {total_pages}, Chunks: {total_chunks})")

        # Dọn dẹp file sau khi thành công
        if file_path.exists():
            try:
                os.remove(file_path)
                logger.info(f"[{doc_id_str}] Đã dọn dẹp file tạm: {file_path.name}")
            except Exception as cleanup_error:
                logger.warning(f"[{doc_id_str}] Không thể xóa file tạm: {cleanup_error}")

        return {"status": "success", "doc_id": doc_id_str, "chunks": total_chunks}

    except Exception as e:
        logger.error(f"[{doc_id_str}] THẤT BẠI: {e}")
        docs_collection.update_one(
            {"_id": ObjectId(doc_id_str)},
            {"$set": {"status": "FAILED", "error_message": str(e), "completed_at": datetime.now()}}
        )
        is_last_attempt = self.request.retries >= self.max_retries
        if is_last_attempt and file_path.exists():
            try:
                os.remove(file_path)
                logger.info(f"[{doc_id_str}] Dọn dẹp file tạm sau retry cuối: {file_path.name}")
            except Exception as cleanup_error:
                logger.warning(f"[{doc_id_str}] Không thể xóa file tạm: {cleanup_error}")

        raise self.retry(exc=e, countdown=60)