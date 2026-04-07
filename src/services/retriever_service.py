from __future__ import annotations

import asyncio
from typing import List, Optional

from langchain_core.documents import Document

from src.core.logger import get_logger
from src.db.mongodb import get_docs_collection
from src.db.qdrant import QdrantDocumentStore, QdrantConfig
from src.services.embedding_service import EmbeddingService
from src.services.rerank_service import RerankerService

logger = get_logger(__name__)


class RetrieverService:
    """
    Pipeline retrieve:
      1. Hybrid search Qdrant top_k=10
      2. Rerank bằng CrossEncoder top_k=5
      3. Lấy parent_id fetch từ MongoDB trả về List[Document]
    """

    def __init__(self):
        qdrant_config = QdrantConfig()
        embedding_service = EmbeddingService()

        self.document_store = QdrantDocumentStore(
            config=qdrant_config,
            embedding_service=embedding_service,
        )
        self.reranker = RerankerService()

    async def retrieve(
        self,
        query: str,
        qdrant_top_k: int = 10,
        rerank_top_k: int = 5,
        filter_dict: Optional[dict] = None,
        rerank_threshold: Optional[float] = None,
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

        reranked_pairs: list[tuple[str, float]] = await asyncio.to_thread(
            self.reranker.rerank, query, texts, rerank_top_k, rerank_threshold
        )

        # Map text → Document gốc để lấy lại metadata (parent_id)
        text_to_doc = {doc.page_content: doc for doc in qdrant_docs}
        top_docs: List[Document] = []
        for text, _ in reranked_pairs:
            doc = text_to_doc.get(text)
            if doc:
                top_docs.append(doc)

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

        def _fetch_parents() -> dict[str, dict]:
            collection = get_docs_collection()
            rows = collection.find(
                {"parent_id": {"$in": parent_ids}},
                {"parent_id": 1, "content": 1, "metadata": 1, "_id": 0},
            )
            return {
                row["parent_id"]: {
                    "content": row.get("content", ""),
                    "metadata": row.get("metadata", {}),
                }
                for row in rows
            }

        parent_map = await asyncio.to_thread(_fetch_parents)
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


retriever_service = RetrieverService()