from __future__ import annotations

import asyncio
from typing import List, Optional

from langchain_core.documents import Document

from src.core.logger import get_logger
from src.db.mongodb import get_docs_collection
from src.db.qdrant import QdrantDocumentStore, QdrantConfig
from src.services.embedding_service import EmbeddingService
from src.services.rerank_service import get_reranker_service

logger = get_logger(__name__)


class RetrieverService:
    """
    Pipeline retrieve:
      1. Hybrid search Qdrant top_k=10
      2. Rerank bằng CrossEncoder top_k=5
      3. Lấy parent_id fetch từ MongoDB trả về List[Document]
    """

    def __init__(self):
        from src.services.embedding_service import get_embedding_service
        qdrant_config = QdrantConfig()
        embedding_service = get_embedding_service()

        self.document_store = QdrantDocumentStore(
            config=qdrant_config,
            embedding_service=embedding_service,
        )
        self.reranker = get_reranker_service()

    async def retrieve(
        self,
        query: str,
        qdrant_top_k: int = 10,
        rerank_top_k: int = 3,
        filter_dict: Optional[dict] = None,
        rerank_threshold: Optional[float] = 0.5,
    ) -> List[Document]:
        """
        Thực hiện toàn bộ pipeline retrieve.

        Args:
            query:            Câu hỏi của người dùng
            qdrant_top_k:     Số kết quả lấy từ Qdrant trước khi rerank
            rerank_top_k:     Số kết quả giữ lại sau khi rerank
            filter_dict:      Bộ lọc metadata cho Qdrant
            rerank_threshold: Ngưỡng điểm số sau khi rerank

        Returns:
            List[Document]: Parent documents lấy từ MongoDB.
        """

        logger.info(f"Bước 1 — Qdrant hybrid search")
        qdrant_docs: List[Document] = await self.document_store.hybrid_search(
            query=query,
            limit=qdrant_top_k,
            filter_dict=filter_dict,
        )

        if not qdrant_docs:
            logger.warning("[retriever] Qdrant không trả về kết quả nào.")
            return []

        logger.info(f"[retriever] Qdrant trả về {len(qdrant_docs)} documents.")

        logger.info(f"[retriever] Bước 2 — Rerank")
        texts = [doc.page_content for doc in qdrant_docs]

        # Rerank là CPU-bound → chạy trong thread pool
        reranked_pairs: list[tuple[str, float]] = await asyncio.to_thread(
            self.reranker.rerank, query, texts, rerank_top_k, rerank_threshold
        )

        # Map reranked text back to original Document via index
        # (avoids collision when 2 chunks have identical page_content)
        text_to_indices: dict[str, list[int]] = {}
        for i, text in enumerate(texts):
            text_to_indices.setdefault(text, []).append(i)

        seen_indices: set[int] = set()
        top_docs: List[Document] = []
        for text, _ in reranked_pairs:
            for idx in text_to_indices.get(text, []):
                if idx not in seen_indices:
                    seen_indices.add(idx)
                    top_docs.append(qdrant_docs[idx])
                    break

        logger.info(f"Sau rerank còn {len(top_docs)} documents.")

        logger.info("Bước 3 — Fetch parent content từ MongoDB...")

        seen: set[str] = set()
        parent_ids: List[str] = []
        for doc in top_docs:
            pid = doc.metadata.get("parent_id", "")
            if pid and pid not in seen:
                seen.add(pid)
                parent_ids.append(pid)

        if not parent_ids:
            logger.warning("Không có parent_id nào hợp lệ.")
            return []

        async def _fetch_parents() -> dict[str, dict]:
            collection = get_docs_collection()
            cursor = collection.find(
                {"_id": {"$in": parent_ids}},
                {"content": 1, "metadata": 1, "_id": 1},
            )
            rows = await cursor.to_list(length=None)
            return {
                row["_id"]: {
                    "content": row.get("content", ""),
                    "metadata": row.get("metadata", {}),
                }
                for row in rows
            }

        # MongoDB motor là async
        parent_map = await _fetch_parents()
        logger.info(f"Fetch được {len(parent_map)} parent documents từ MongoDB.")

        results: List[Document] = []
        for pid in parent_ids:
            data = parent_map.get(pid)
            if not data:
                logger.warning(f"Không tìm thấy parent_id='{pid}' trong MongoDB.")
                continue
            results.append(
                Document(
                    page_content=data["content"],
                    metadata=data["metadata"],
                )
            )

        logger.info(f"Pipeline hoàn thành. Trả về {len(results)} documents.")
        return results


_retriever_instance: RetrieverService | None = None


def get_retriever_service() -> RetrieverService:
    """Lazy singleton — tạo RetrieverService 1 lần duy nhất."""
    global _retriever_instance
    if _retriever_instance is None:
        logger.info("[singleton] Khởi tạo RetrieverService instance...")
        _retriever_instance = RetrieverService()
    return _retriever_instance