from src.core.config import settings
from src.core.logger import get_logger
from langchain_ollama import OllamaEmbeddings
from typing import List

logger = get_logger(__name__)

class EmbeddingService:
    def __init__(self):
        logger.info("[singleton] Khởi tạo OllamaEmbeddings instance...")
        self.embedding = OllamaEmbeddings(
            model=settings.EMBEDDING_MODEL,
            base_url=settings.OLLAMA_BASE_URL
        )

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Embed a list of texts into vectors.
        """
        formatted_texts = [f"passage: {text}" for text in texts]
        return self.embedding.embed_documents(formatted_texts)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embed a list of documents into vectors.
        """
        return self.embed_texts(texts)
        
    def embed_query(self, query: str) -> List[float]:
        """
        Embed a query into a vector.
        Args:
            query (str): The query to embed.
        Returns:
            List[float]: The vector representation of the query."""
        formatted_query = f"query: {query}"
        return self.embedding.embed_query(formatted_query)

    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Async version: Embed a list of documents into vectors.
        """
        formatted_texts = [f"passage: {text}" for text in texts]
        return await self.embedding.aembed_documents(formatted_texts)

    async def aembed_query(self, query: str) -> List[float]:
        """
        Async version: Embed a query into a vector.
        """
        formatted_query = f"query: {query}"
        return await self.embedding.aembed_query(formatted_query)


_embedding_service_instance: EmbeddingService | None = None

def get_embedding_service() -> EmbeddingService:
    """Lazy singleton — tạo EmbeddingService 1 lần duy nhất."""
    global _embedding_service_instance
    if _embedding_service_instance is None:
        _embedding_service_instance = EmbeddingService()
    return _embedding_service_instance