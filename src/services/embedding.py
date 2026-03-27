from src.core.config import settings
from src.core.logger import get_logger
from langchain_ollama import OllamaEmbeddings
from typing import List
logger = get_logger(__name__)

class EmbeddingService:
    def __init__(self):
        self.embedding = OllamaEmbeddings(
            model=settings.EMBEDDING_MODEL
        )
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Nhúng văn bản thành vector"""
        formatted_texts = [f"passage: {text}" for text in texts]
        return self.embedding.embed_documents(formatted_texts)
        
    def embed_query(self, query: str) -> List[float]:
        formatted_query = f"query: {query}"
        return self.embedding.embed_query(formatted_query)