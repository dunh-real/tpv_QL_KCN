import asyncio
from src.core.logger import get_logger
from langchain_core.documents import Document
from src.db.qdrant import QdrantDocumentStore, QdrantConfig
from src.services.embedding_service import EmbeddingService
from src.services.rerank_service import get_reranker_service

logger = get_logger(__name__)


class SqlSchemaService:
    def __init__(self, collection_name: str = "sql_schemas"):
        from src.services.embedding_service import get_embedding_service
        qdrant_config = QdrantConfig(collection_name=collection_name)
        embedding_service = get_embedding_service()
        self.document_store = QdrantDocumentStore(
            config=qdrant_config,
            embedding_service=embedding_service,
        )
        self.reranker = get_reranker_service()

    async def ingest_schemas(self, schemas_data: list[dict]):
        """
        schemas_data: List các dictionary 
        """
        documents = []
        for item in schemas_data:
            documents.append(
                Document(
                    page_content=item["search_content"],
                    metadata=item["payload"]
                )
            )
        
        await self.document_store.upsert_documents(documents)
        logger.info(f"Đã ingest {len(documents)} schemas vào Qdrant")
        return {"status": "success", "total_ingested": len(documents)}

    async def search_tables(self, query: str, top_k: int = 5, rerank_threshold: float = 0.8) -> list[str]:
        """
        Tìm kiếm các bảng liên quan đến câu hỏi người dùng bằng vector search (hybrid) và Rerank.
        Trả về danh sách tên bảng (table_name).
        """
        # Bước 1: Qdrant hybrid search mở rộng
        candidate_limit = max(20, top_k * 3)
        results = await self.document_store.hybrid_search(query=query, limit=candidate_limit)
        
        if not results:
            return []

        # Bước 2: Rerank 
        texts = [doc.page_content for doc in results]
        reranked_pairs = await asyncio.to_thread(
            self.reranker.rerank, query, texts, top_k, rerank_threshold
        )
        
        # Bước 3: Map lại để lấy table_name (index-based to avoid collision)
        table_names = []
        seen_tables: set[str] = set()
        text_to_indices: dict[str, list[int]] = {}
        for i, text in enumerate(texts):
            text_to_indices.setdefault(text, []).append(i)

        for text, score in reranked_pairs:
            for idx in text_to_indices.get(text, []):
                t_name = results[idx].metadata.get("table_name")
                if t_name and t_name not in seen_tables:
                    seen_tables.add(t_name)
                    table_names.append(t_name)
                break
                
        logger.info(f"[schema] search_tables → {len(table_names)} bảng: {table_names}")
        return table_names

    async def get_schemas(self, table_names: list[str]) -> dict[str, str]:
        """
        Lấy DDL của một danh sách các bảng bằng MatchAny filter trên trường table_name.
        Trả về dictionary mapping giữa table_name và DDL.
        """
        if not table_names:
            return {}
            
        from qdrant_client.http import models
        
        filter_conditions = models.Filter(
            must=[
                models.FieldCondition(
                    key="table_name",
                    match=models.MatchAny(any=table_names)
                )
            ]
        )
        
        # Fix: limit phải đủ lớn để lấy hết tất cả documents của các bảng
        # Mỗi bảng có thể có nhiều hơn 1 chunk document
        fetch_limit = len(table_names) * 10
        
        response = await self.document_store.client.scroll(
            collection_name=self.document_store.config.collection_name,
            scroll_filter=filter_conditions,
            limit=fetch_limit,
            with_payload=["table_name", "ddl"]
        )
        
        points = response[0]
        result = {}
        for point in points:
            t_name = point.payload.get("table_name")
            ddl = point.payload.get("ddl")
            if t_name and ddl and t_name not in result:
                # Lấy DDL của điểm đầu tiên tìm thấy cho mỗi bảng
                result[t_name] = ddl
                
        logger.info(f"[schema] get_schemas → lấy được DDL cho {len(result)}/{len(table_names)} bảng.")
        return result


# ─── Singleton ───────────────────────────────────────────
_schema_service_instance: SqlSchemaService | None = None


def get_schema_service() -> SqlSchemaService:
    """Lazy singleton — tạo SqlSchemaService 1 lần duy nhất."""
    global _schema_service_instance
    if _schema_service_instance is None:
        logger.info("[singleton] Khởi tạo SqlSchemaService instance...")
        _schema_service_instance = SqlSchemaService()
    return _schema_service_instance
