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
        self.docs_collection = get_docs_collection()

        qdrant_config = QdrantConfig()
        embedding_service = EmbeddingService()
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
        # Bước 1: Lưu parent_chunks vào MongoDB
        logger.info(f"Bước 1 — Lưu {len(parent_chunks)} parent chunk(s) vào MongoDB...")
        saved_parents = await asyncio.to_thread(self._save_parents_to_mongo, parent_chunks)
        logger.info(f"  -> Đã lưu {saved_parents} parent chunk(s).")

        # Bước 2: Upsert children_chunks vào Qdrant
        logger.info(f"Bước 2 — Upsert {len(children_chunks)} children chunk(s) vào Qdrant...")
        await self.document_store.upsert_documents(children_chunks)
        logger.info(f"  -> Đã upsert {len(children_chunks)} children chunk(s).")

        return {
            "parent_chunks": saved_parents,
            "children_chunks": len(children_chunks),
        }

    def _save_parents_to_mongo(self, parent_chunks: List[Document]) -> int:
        """
        Lưu danh sách parent_chunks vào MongoDB.
        Dùng upsert theo parent_id để tránh trùng lặp.
        """
        count = 0
        for parent in parent_chunks:
            parent_id = parent.metadata.get("parent_id")
            if not parent_id:
                logger.warning("Parent chunk không có parent_id, bỏ qua.")
                continue
            self.docs_collection.update_one(
                {"parent_id": parent_id},
                {"$set": {
                    "parent_id": parent_id,
                    "content": parent.page_content,
                    "metadata": parent.metadata,
                }},
                upsert=True,
            )
            count += 1
        return count
