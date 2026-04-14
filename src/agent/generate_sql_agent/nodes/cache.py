

from src.agent.generate_sql_agent.state import GenerateSQLState
from src.core.semantic_cache import get_cached_sql, cache_sql
from src.core.logger import get_logger

logger = get_logger(__name__)


async def check_cache_node(state: GenerateSQLState) -> dict:
    """
    Kiểm tra semantic cache trước khi gọi LLM để sinh SQL.

    Returns:
        Nếu cache hit: {'sql_query': <sql>, 'is_cache_hit': True, 'error': None}
        Nếu cache miss: {'is_cache_hit': False}
    """
    question = state["question"]
    logger.info(f"[check_cache] Kiểm tra cache cho câu hỏi: '{question}'")

    cached_sql = await get_cached_sql(question)
    if cached_sql:
        logger.info(f"[check_cache] Cache HIT → SQL: {cached_sql[:80]}...")
        return {
            "sql_query": cached_sql,
            "is_cache_hit": True,
            "error": None,
        }

    logger.info("[check_cache] Cache MISS → sẽ generate SQL mới.")
    return {"is_cache_hit": False}


async def update_cache_node(state: GenerateSQLState) -> dict:
    """
    Lưu câu hỏi + SQL vào cache sau khi thực thi thành công.
    Chỉ lưu khi: không có lỗi VÀ đây là SQL mới (không phải cache hit).

    Returns:
        {} — không thay đổi state
    """
    if state.get("error") or state.get("is_cache_hit"):
        logger.debug("[update_cache] Bỏ qua cache update (lỗi hoặc cache hit).")
        return {}

    question = state.get("question", "")
    sql_query = state.get("sql_query", "")

    if question and sql_query and sql_query.strip():
        await cache_sql(question, sql_query)
        logger.info(f"[update_cache] Đã lưu cache cho câu hỏi: '{question}'")

    return {}