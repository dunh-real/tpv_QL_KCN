
from __future__ import annotations

import uuid
from typing import Optional

from qdrant_client import AsyncQdrantClient, models
from qdrant_client.http.models import Distance, VectorParams

from src.core.config import settings
from src.core.logger import get_logger
from src.services.embedding_service import EmbeddingService
from src.agent.sql_agent.constants import CACHE_SCORE_THRESHOLD, CACHE_TOP_K

logger = get_logger(__name__)

COLLECTION = settings.SQL_COLLECTION_NAME
DIMENSION  = settings.VECTOR_DIMENSION
URL        = f"http://{settings.QDRANT_HOST}:{settings.QDRANT_PORT}"
API_KEY    = settings.QDRANT_API_KEY

client: AsyncQdrantClient = AsyncQdrantClient(url=URL, api_key=API_KEY, timeout=30)
embedding: EmbeddingService = EmbeddingService()
collection_ready: bool = False


async def _ensure_collection() -> None:
    """Tạo collection nếu chưa tồn tại (lazy init, chỉ chạy 1 lần)."""
    global collection_ready
    if collection_ready:
        return
    exists = await client.collection_exists(COLLECTION)
    if not exists:
        await client.create_collection(
            collection_name=COLLECTION,
            vectors_config=VectorParams(size=DIMENSION, distance=Distance.COSINE),
        )
        logger.info(f"[semantic_cache] Đã tạo Qdrant collection '{COLLECTION}'.")
    else:
        logger.debug(f"[semantic_cache] Collection '{COLLECTION}' đã tồn tại.")
    collection_ready = True


async def get_cached_sql(
    question: str,
    top_k: int = CACHE_TOP_K,
    score_threshold: float = CACHE_SCORE_THRESHOLD,
) -> Optional[str]:
    """
    Tìm SQL đã cache gần nhất với câu hỏi.

    Args:
        question: Câu hỏi của người dùng.
        top_k: Số kết quả trả về từ Qdrant.
        score_threshold: Ngưỡng cosine similarity tối thiểu (0–1).

    Returns:
        SQL string nếu cache HIT, None nếu MISS.
    """
    try:
        await _ensure_collection()
        query_vector = await embedding.embed_query(question)

        results = await client.search(
            collection_name=COLLECTION,
            query_vector=query_vector,
            limit=top_k,
            with_payload=True,
            score_threshold=score_threshold,
        )

        if results:
            best = results[0]
            sql_query = best.payload.get("sql_query")
            logger.info(
                f"[semantic_cache] Cache HIT (score={best.score:.3f}) "
                f"cho câu hỏi: '{question}'"
            )
            return sql_query

    except Exception as e:
        logger.error(f"[semantic_cache] Lỗi get_cached_sql: {e}")

    logger.info(f"[semantic_cache] Cache MISS cho câu hỏi: '{question}'")
    return None


async def cache_sql(question: str, sql_query: str) -> None:
    """
    Lưu câu hỏi + SQL vào Qdrant cache.

    Args:
        question: Câu hỏi của người dùng.
        sql_query: Câu lệnh SQL tương ứng.
    """
    try:
        await _ensure_collection()
        vector = await embedding.embed_query(question)

        point = models.PointStruct(
            id=str(uuid.uuid4()),
            vector=vector,
            payload={"question": question, "sql_query": sql_query},
        )
        await client.upsert(collection_name=COLLECTION, points=[point])
        logger.info(f"[semantic_cache] Cache STORED cho câu hỏi: '{question}'")

    except Exception as e:
        logger.error(f"[semantic_cache] Lỗi cache_sql: {e}")
