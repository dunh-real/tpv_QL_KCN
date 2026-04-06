from __future__ import annotations
import uuid
from dataclasses import dataclass, field
from typing import Any, List, Optional
from langchain_core.documents import Document
from qdrant_client import QdrantClient, models
from qdrant_client.http.models import (
    Distance,
    HnswConfigDiff,
    OptimizersConfigDiff,
    PointStruct,
    QuantizationConfig,
    ScalarQuantization,
    ScalarQuantizationConfig,
    ScalarType,
    SparseIndexParams,
    SparseVectorParams,
    VectorParams,
)
from src.core.config import settings
from src.core.logger import get_logger
from src.services.embedding import EmbeddingService
logger = get_logger(__name__)

@dataclass
class QdrantConfig:
    """Cau hinh cho Qdrant"""
    url: str = f"http://{settings.QDRANT_HOST}:{settings.QDRANT_PORT}"
    api_key: str = settings.QDRANT_API_KEY
    collection_name: str = settings.QDRANT_COLLECTION_NAME

    dense_size: int = settings.VECTOR_DIMENSION  
    dense_distance: Distance = Distance.COSINE
    dense_vector_name: str = "dense"

    sparse_model: str = "Qdrant/bm25"            
    sparse_vector_name: str = "sparse"

    hnsw_m: int = 16
    hnsw_ef_construct: int = 100
    hnsw_on_disk: bool = False

    use_quantization: bool = True
    quantization_always_ram: bool = True

    indexing_threshold: int = 10000

    prefetch_limit: int = 100          
    fusion_limit: int = 20

    
class QdrantDocumentStore:
    """
    Quản lý toàn bộ vòng đời của một Qdrant hybrid collection:
      - Tạo / kiểm tra collection
      - Upsert documents
      - Hybrid search (dense + sparse + RRF fusion)
      - Filtered search
      - Scroll / delete / recreate
    """
    def __init__(self, config: QdrantConfig, embedding_service: EmbeddingService):
        self.config = config
        self.embedding_service = embedding_service
        self.client = QdrantClient(
            url=self.config.url,
            api_key=self.config.api_key,
            timeout=60,
        )
    def create_collection(self):
        """
        Tạo collection với:
          - 1 dense named vector 
          - 1 sparse named vector 
        """
        name = self.config.collection_name
        exists = self.client.collection_exists(name)
        if exists:
            collection_info = self.client.get_collection(name)
            existing_size = collection_info.config.params.vectors[self.config.dense_vector_name].size
            if existing_size != self.config.dense_size:
                print(f"[qdrant] Dimension mismatch: existing={existing_size}, config={self.config.dense_size}. Recreating collection '{name}'...")
                self.client.delete_collection(name)
            else:
                print(f"[qdrant] Collection '{name}' đã tồn tại và khớp dimension, bỏ qua tạo mới.")
                return
        quantization: QuantizationConfig | None = None
        if self.config.use_quantization:
            quantization = ScalarQuantization(
                scalar=ScalarQuantizationConfig(
                    type=ScalarType.INT8,
                    quantile=0.99,
                    always_ram=self.config.quantization_always_ram,
                )
            )
        self.client.create_collection(
            collection_name=name,
            vectors_config={
                self.config.dense_vector_name: VectorParams(
                    size=self.config.dense_size,
                    distance=self.config.dense_distance,
                    hnsw_config=HnswConfigDiff(
                        m=self.config.hnsw_m,
                        ef_construct=self.config.hnsw_ef_construct,
                        on_disk=self.config.hnsw_on_disk,
                    ),
                    quantization_config=quantization,
                    on_disk=False,
                )
            },
            sparse_vectors_config={
                self.config.sparse_vector_name: SparseVectorParams(
                    index=SparseIndexParams(
                        on_disk=False,
                        full_scan_threshold=5_000,
                    )
                )
            },
            optimizers_config=OptimizersConfigDiff(
                indexing_threshold=self.config.indexing_threshold,
            ),
        )

    def collection_info(self) -> dict[str, Any]:
        info = self.client.get_collection(self.config.collection_name)
        return {
            "status": info.status,
            "vectors_count": info.vectors_count,
            "points_count": info.points_count,
            "indexed_vectors_count": info.indexed_vectors_count,
        }
    def upsert_documents(self, documents: List[Document], batch_size: int = 64):
        """Upsert documents vào Qdrant collection.
        Args:
            documents: List of Document objects to upsert.
            batch_size: Number of documents to upsert in each batch.
        """
        self.create_collection()
        total_points = 0
        for i in range(0, len(documents), batch_size):
            batch_docs = documents[i : i + batch_size]
            texts = [doc.page_content for doc in batch_docs]
            
            # Dense vectors from Ollama
            dense_vectors = self.embedding_service.embed_documents(texts)
            # Sparse vectors from FastEmbed
            sparse_vectors = self._encode_sparse(texts, batch_size=batch_size)
            
            points = []
            for doc, dense, sparse in zip(batch_docs, dense_vectors, sparse_vectors):
                # Dùng thuộc tính id từ metadata nếu có, nếu không thì tự tạo
                point_id = doc.metadata.get("id") or str(uuid.uuid4())
                payload = {
                    **doc.metadata,
                    "page_content": doc.page_content,
                }
                
                point = models.PointStruct(
                    id=point_id,
                    vector={
                        self.config.dense_vector_name: dense,
                        self.config.sparse_vector_name: sparse,
                    },
                    payload=payload
                )
                points.append(point)
                
            self.client.upsert(
                collection_name=self.config.collection_name,
                points=points
            )
            total_points += len(points)
            logger.info(f"Đã upsert {len(points)} points. Tổng số: {total_points}/{len(documents)}")
            
        logger.info(f"Hoàn thành upsert tổng cộng {total_points} documents vào collection '{self.config.collection_name}'.")

    def hybrid_search(self, query: str, limit: int, filter_dict: Optional[dict] = None) -> List[Document]:
        """Tìm kiếm kết hợp dense và sparse sử dụng RRF.
        Args:
            query: Query string.
            limit: Number of results to return.
            filter_dict: Dictionary of filters.
        Returns:
            List of Document objects.
        """

        limit = limit or self.config.fusion_limit
        dense_query = self.embedding_service.embed_query(query)
        sparse_query_list = self._encode_sparse([query], batch_size=1)
        sparse_query = sparse_query_list[0]
        
        # Tạo query filter nếu cần thiết
        query_filter = None
        if filter_dict:
            # chuyển đổi dictionary filter thành Qdrant filter
            must_conditions = []
            for key, value in filter_dict.items():
                must_conditions.append(
                    models.FieldCondition(
                        key=key,
                        match=models.MatchValue(value=value)
                    )
                )
            if must_conditions:
                query_filter = models.Filter(must=must_conditions)

        prefetch = [
            # Branch 1: Dense
            models.Prefetch(
                query=dense_query,
                using=self.config.dense_vector_name,
                limit=limit * 2,
                filter=query_filter,
            ),
            # Branch 2: Sparse
            models.Prefetch(
                query=models.SparseVector(
                    indices=sparse_query.indices,
                    values=sparse_query.values,
                ),
                using=self.config.sparse_vector_name,
                limit=limit * 2,
                filter=query_filter,
            ),
        ]
        
        results = self.client.query_points(
            collection_name=self.config.collection_name,
            prefetch=prefetch,
            query=models.FusionQuery(
                fusion=models.Fusion.RRF
            ),
            limit=limit,
            with_payload=True
        )
        
        documents = []
        for point in results.points:
            payload = point.payload or {}
            payload["score"] = point.score
            page_content = payload.pop("page_content", "")
            doc = Document(page_content=page_content, metadata=payload)
            documents.append(doc)
            
        return documents

    def _encode_sparse(self, texts: list[str], batch_size: int) -> list[models.SparseVector]:
        """Dùng fastembed sparse (BM25) để encode sparse vectors."""
        from fastembed.sparse.bm25 import Bm25
        if not hasattr(self, "_sparse_encoder"):
            self._sparse_encoder = Bm25(self.config.sparse_model)
        results = []
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            for sv in self._sparse_encoder.embed(batch):
                results.append(
                    models.SparseVector(
                        indices=sv.indices.tolist(),
                        values=sv.values.tolist(),
                    )
                )
        return results
    def drop_collection(self) -> None:
        self.client.delete_collection(self.config.collection_name)
        print(f"[qdrant] Collection '{self.config.collection_name}' đã bị xoá.")
    
    def delete_by_filter(self, filter_conditions: models.Filter) -> None:
        self.client.delete(
            collection_name=self.config.collection_name,
            points_selector=models.FilterSelector(filter=filter_conditions),
            wait=True,
        )
        print("[qdrant] Đã xoá points theo filter.")
    def delete_by_ids(self, ids: list[str | int]) -> None:
        self.client.delete(
            collection_name=self.config.collection_name,
            points_selector=models.PointIdsList(points=ids),
            wait=True,
        )
        print(f"[qdrant] Đã xoá {len(ids)} points.")


    