from src.core.config import settings
from src.core.logger import get_logger
from langchain_ollama import OllamaEmbeddings
from typing import List

logger = get_logger(__name__)

class EmbeddingService:
    def __init__(self):
        self.embedding = OllamaEmbeddings(
            model=settings.EMBEDDING_MODEL,
            base_url=settings.OLLAMA_BASE_URL
        )

    async def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Embed a list of texts into vectors.
        """
        formatted_texts = [f"passage: {text}" for text in texts]
        return await self.embedding.aembed_documents(formatted_texts)

    async def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embed a list of documents into vectors.
        """
        return await self.embed_texts(texts)
        
    async def embed_query(self, query: str) -> List[float]:
        """
        Embed a query into a vector.
        Args:
            query (str): The query to embed.
        Returns:
            List[float]: The vector representation of the query."""
        formatted_query = f"query: {query}"
        return await self.embedding.aembed_query(formatted_query)