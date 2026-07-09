import asyncio
from src.core.logger import get_logger
from langchain_core.documents import Document
from src.db.qdrant import QdrantDocumentStore, QdrantConfig
from src.services.embedding_service import EmbeddingService
from src.services.rerank_service import RerankerService

logger = get_logger(__name__)

class SqlSchemaService:
    def __init__(self, collection_name: str = "sql_schemas"):
        qdrant_config = QdrantConfig(collection_name=collection_name)
        embedding_service = EmbeddingService()
        self.document_store = QdrantDocumentStore(
            config=qdrant_config,
            embedding_service=embedding_service,
        )
        self.reranker = RerankerService()

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

    async def search_tables(self, query: str, top_k: int = 5, rerank_threshold: float = 0.0) -> list[str]:
        """
        Tìm kiếm các bảng liên quan đến câu hỏi người dùng bằng vector search (hybrid) và Rerank.
        Trả về danh sách tên bảng (table_name).
        """
        # Bước 1: Qdrant hybrid search mở rộng
        results = await self.document_store.hybrid_search(query=query, limit=max(20, top_k * 3))
        
        if not results:
            return []

        # Bước 2: Rerank
        texts = [doc.page_content for doc in results]
        reranked_pairs = await asyncio.to_thread(
            self.reranker.rerank, query, texts, top_k, rerank_threshold
        )
        
        # Bước 3: Map lại để lấy table_name
        text_to_table = {doc.page_content: doc.metadata.get("table_name") for doc in results}
        
        table_names = []
        for text, score in reranked_pairs:
            t_name = text_to_table.get(text)
            if t_name and t_name not in table_names:
                table_names.append(t_name)
                
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
        
        response = await self.document_store.client.scroll(
            collection_name=self.document_store.config.collection_name,
            scroll_filter=filter_conditions,
            limit=len(table_names),
            with_payload=True
        )
        
        points = response[0]
        result = {}
        for point in points:
            t_name = point.payload.get("table_name")
            ddl = point.payload.get("ddl")
            if t_name and ddl:
                result[t_name] = ddl
                
        return result
