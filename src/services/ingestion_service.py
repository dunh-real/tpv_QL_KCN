from __future__ import annotations

import asyncio
from typing import List

from langchain_core.documents import Document

from src.core.logger import get_logger
from src.db.mongodb import get_docs_collection
from src.db.qdrant import QdrantConfig, QdrantDocumentStore
from src.services.embedding_service import EmbeddingService

logger = get_logger(__name__)


class IngestionService:
    """
    Pipeline ingestion:
      1. Lưu parent_chunks vào MongoDB
      2. Upsert children_chunks vào Qdrant
    """

    def __init__(self):
        from src.services.embedding_service import get_embedding_service
        self.docs_collection = get_docs_collection()

        qdrant_config = QdrantConfig()
        embedding_service = get_embedding_service()
        self.document_store = QdrantDocumentStore(
            config=qdrant_config,
            embedding_service=embedding_service,
        )

    async def ingest(
        self,
        parent_chunks: List[Document],
        children_chunks: List[Document],
    ) -> dict:
        """
        Lưu parent_chunks vào MongoDB và upsert children_chunks vào Qdrant.

        Args:
            parent_chunks:  Danh sách Document nội dung đầy đủ, mỗi doc có metadata['parent_id'].
            children_chunks: Danh sách Document để upsert vào Qdrant.
        Returns:
            dict: Thống kê số parent và children đã lưu.
        """
        # Bước 1: Lưu parent_chunks vào MongoDB (dùng async motor)
        logger.info(f"Bước 1 — Lưu {len(parent_chunks)} parent chunk(s) vào MongoDB...")
        saved_parents = await self._save_parents_to_mongo(parent_chunks)
        logger.info(f"  -> Đã lưu {saved_parents} parent chunk(s).")

        # Bước 2: Upsert children_chunks vào Qdrant (async)
        logger.info(f"Bước 2 — Upsert {len(children_chunks)} children chunk(s) vào Qdrant...")
        await self.document_store.upsert_documents(children_chunks)
        logger.info(f"  -> Đã upsert {len(children_chunks)} children chunk(s).")

        return {
            "parent_chunks": saved_parents,
            "children_chunks": len(children_chunks),
        }

    async def _save_parents_to_mongo(self, parent_chunks: List[Document]) -> int:
        """
        Lưu danh sách parent_chunks vào MongoDB.
        Dùng bulk_write với UpdateOne để tối ưu I/O.
        """
        from pymongo import UpdateOne
        
        operations = []
        for parent in parent_chunks:
            parent_id = parent.metadata.get("parent_id")
            if not parent_id:
                logger.warning("Parent chunk không có parent_id, bỏ qua.")
                continue
                
            metadata = dict(parent.metadata)
            metadata.pop("parent_id", None)

            operations.append(
                UpdateOne(
                    {"_id": parent_id},
                    {"$set": {
                        "content": parent.page_content,
                        "metadata": metadata,
                    }},
                    upsert=True,
                )
            )
            
        if operations:
            await self.docs_collection.bulk_write(operations)
            
        return len(operations)

    async def delete_by_filename(self, filename: str) -> None:
        """
        Xoá tất cả dữ liệu cũ của file theo filename (cả MongoDB và Qdrant).
        Dùng để đảm bảo idempotency khi ingest lại cùng một file.
        """
        # Xoá parent chunks từ MongoDB
        result = await self.docs_collection.delete_many(
            {"metadata.filename": filename}
        )
        logger.info(f"Đã xoá {result.deleted_count} parent chunk(s) cũ từ MongoDB.")

        # Xoá children chunks từ Qdrant
        from qdrant_client.http.models import FieldCondition, Filter, MatchValue

        await self.document_store.delete_by_filter(
            Filter(
                must=[
                    FieldCondition(
                        key="filename", match=MatchValue(value=filename)
                    )
                ]
            )
        )
        logger.info(f"Đã xoá children chunk(s) cũ từ Qdrant theo filename='{filename}'.")


_ingestion_instance: IngestionService | None = None


def get_ingestion_service() -> IngestionService:
    """Lazy singleton — tạo IngestionService 1 lần duy nhất."""
    global _ingestion_instance
    if _ingestion_instance is None:
        logger.info("[singleton] Khởi tạo IngestionService instance...")
        _ingestion_instance = IngestionService()
    return _ingestion_instance
