from src.core.config import settings
from src.core.logger import get_logger
from src.services.embedding import EmbeddingService
from typing import List, Dict, Any, Optional
from hyperspace import HyperspaceClient
import uuid
import sys

logger = get_logger(__name__)

class EnterpriseDocumentStore:
    def __init__(self, host: str, api_key: str, collection_name: str):
        """Khởi tạo kết nối với Timeout và Error Handling"""
        clean_host = host.replace("http://", "").replace("https://", "")
        self.host = clean_host if ":" in clean_host else f"{clean_host}:50051"
        self.collection_name = collection_name
        try:
            self.client = HyperspaceClient(host=self.host, api_key=api_key)
            logger.info(f"Kết nối thành công tới HyperspaceDB tại {self.host}")
        except Exception as e:
            logger.error(f"Lỗi kết nối cơ sở dữ liệu: {e}")
            raise RuntimeError("Không thể khởi động Vector Store.") from e
    
        self.init_collection(self.collection_name, settings.HYPERSPACE_VECTOR_DIMENSION)
        self.embedding_service = EmbeddingService()
    def init_collection(self, collection_name: str, dimension: int = 384):
        """Thiết lập không gian vector"""
        try:
            collections = self.client.list_collections()
            if collection_name not in collections:
                success = self.client.create_collection(
                    name=collection_name,
                    dimension=dimension,
                    metric="cosine"  
                )
                if success:
                    logger.info(f"Đã tạo collection mới: '{collection_name}' ({dimension}d)")
                else:
                    logger.error(f"Lỗi khi tạo collection {collection_name}")
            else:
                logger.info(f"Collection '{collection_name}' đã tồn tại.")
        except Exception as e:
            logger.error(f"Lỗi cấu hình collection {collection_name}: {e}")
            raise



    def ingest_documents(self, documents: List[Dict[str, Any]], batch_size: int = 200):
        """Ingest dữ liệu theo batch"""
        text_to_embed = [doc.get("content", "") for doc in documents]
        embeddings = self.embedding_service.embed_texts(text_to_embed)
        logger.info("Dang luu du lieu vao HyperSpaceDB ...")
        
        for i in range(0, len(documents), batch_size):
            batch_vecs = embeddings[i:i+batch_size]
            batch_docs = documents[i:i+batch_size]
            
            ids = []
            metadatas = []
            
            import hashlib
            for index, doc in enumerate(batch_docs):
                doc_id_string = doc.get("doc_id", uuid.uuid4().hex)
                doc_id_hash = int(hashlib.md5(doc_id_string.encode('utf-8')).hexdigest(), 16) % (2**31 - 1)
                
                ids.append(doc_id_hash)
                
                metadatas.append(doc)
                
            success = self.client.batch_insert(
                vectors=batch_vecs,
                ids=ids,
                metadatas=metadatas,
                collection=self.collection_name
            )
            if success:
                logger.info(f"Đã ingest batch {i//batch_size + 1} / {len(documents)//batch_size + 1}")
            else:
                logger.error(f"Lỗi khi ingest batch {i//batch_size + 1}")

    def search_hybrid(self, query: str, top_k: int = 5, alpha: float = 0.5, metadata_filters: Dict[str, Any] = None, score_threshold: float = 0.50) -> List[Dict[str, Any]]:
        try:
            query_vector = self.embedding_service.embed_query(query)
            
            api_filters = []
            if metadata_filters:
                for k, v in metadata_filters.items():
                    api_filters.append({"type": "match", "key": k, "value": str(v)})
                    
            raw_results = self.client.search(
                vector=query_vector,
                top_k=top_k,
                hybrid_query=query,
                hybrid_alpha=alpha,
                filters=api_filters if api_filters else None,
                collection=self.collection_name
            )

            results = []
            for res in raw_results:
                if res.get("distance", 1.0) <= (1.0 - score_threshold):
                    payload = res.get("metadata", {})
                    results.append({
                        "doc_id": payload.get("doc_id", str(res.get("id"))),
                        "score": 1.0 - res.get("distance", 0.0),
                        "content": payload.get("content", ""),
                        "metadata": {k: v for k, v in payload.items() if k not in ["doc_id", "content"]}
                    })
            return results
        except Exception as e:
            logger.error(f"Lỗi Hybrid Search: {e}")
            return []
