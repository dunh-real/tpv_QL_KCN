from src.core.config import settings
from src.core.logger import get_logger
from src.services.embedding import EmbeddingService
from typing import List, Dict, Any, Optional
from langchain_core.documents import Document
from hyperspace import HyperspaceClient
import hashlib
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
    
        self.init_collection(self.collection_name, settings.VECTOR_DIMENSION)
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



    def ingest_documents(self, documents: List[Document], batch_size: int = 200):
        """Ingest dữ liệu theo batch, chống ghi đè ID"""
        text_to_embed = [doc.page_content for doc in documents]
        embeddings = self.embedding_service.embed_texts(text_to_embed)
        logger.info("Đang lưu dữ liệu vào HyperSpaceDB ...")
        
        for i in range(0, len(documents), batch_size):
            batch_vecs = embeddings[i:i+batch_size]
            batch_docs = documents[i:i+batch_size]
            
            ids = []
            metadatas = []
            
            for index, doc in enumerate(batch_docs):
                base_doc_id = doc.metadata.get("doc_id", uuid.uuid4().hex)
                chunk_index = doc.metadata.get("chunk_index", index)
                
                unique_chunk_string = f"{base_doc_id}_chunk_{chunk_index}"
                doc_id_hash = int(hashlib.md5(unique_chunk_string.encode('utf-8')).hexdigest(), 16) % (2**31 - 1)
                
                ids.append(doc_id_hash)
                
                raw = {"content": doc.page_content, **doc.metadata}
                clean_metadata = {str(k): str(v) for k, v in raw.items()}
                metadatas.append(clean_metadata)

            safe_vecs = [[float(x) for x in vec] for vec in batch_vecs]
                
            success = self.client.batch_insert(
                vectors=safe_vecs,
                ids=ids,
                metadatas=metadatas,
                collection=self.collection_name
            )
            if success:
                logger.info(f"Đã ingest batch {i//batch_size + 1} / {(len(documents)-1)//batch_size + 1}")
            else:
                raise RuntimeError(f"HyperspaceDB từ chối ingest batch {i//batch_size + 1} — kiểm tra kết nối và collection.")

    def search_hybrid(self, query: str, top_k: int = 5, alpha: float = 0.5, metadata_filters: Dict[str, Any] = None, score_threshold: float = 0.50) -> List[Dict[str, Any]]:
        """
        Search for documents using hybrid search.
        Args:
            query (str): The query to search for.
            top_k (int): The number of results to return.
            alpha (float): The weight of the hybrid search.
            metadata_filters (Dict[str, Any]): The metadata filters to apply.
            score_threshold (float): The score threshold to apply.
        Returns:
            List[Dict[str, Any]]: The list of documents.
        """
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
