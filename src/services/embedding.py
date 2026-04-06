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
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Embed a list of texts into vectors.
        Args:
            texts (List[str]): The list of texts to embed.
        Returns:
            List[List[float]]: The vector representation of the texts.
        """
        formatted_texts = [f"passage: {text}" for text in texts]
        return self.embedding.embed_documents(formatted_texts)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embed a list of documents into vectors.
        Args:
            texts (List[str]): The list of documents to embed.
        Returns:
            List[List[float]]: The vector representation of the documents.
        """
        return self.embed_texts(texts)
        
    def embed_query(self, query: str) -> List[float]:
        """
        Embed a query into a vector.
        Args:
            query (str): The query to embed.
        Returns:
            List[float]: The vector representation of the query.
        """
        formatted_query = f"query: {query}"
        return self.embedding.embed_query(formatted_query)

embedding_service = EmbeddingService()