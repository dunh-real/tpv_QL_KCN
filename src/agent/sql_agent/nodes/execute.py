import asyncio
import time

from src.agent.sql_agent.state import SQLState
from src.agent.sql_agent.constants import MAX_RESULT_LENGTH
from src.db.mysql import get_db_manager
from src.core.logger import get_logger

logger = get_logger(__name__)


async def execute_sql_node(state: SQLState) -> dict:
    """
    Thực thi câu lệnh SQL đã validate xuống database.

    - Chạy async qua asyncio.to_thread để không block event loop.
    - Truncate kết quả nếu vượt MAX_RESULT_LENGTH, tránh quá token cho LLM downstream.

    Returns:
        {'result': <str>, 'error': None} khi thành công.
        {'result': <error_msg>, 'error': <str>} khi lỗi.
    """
    sql_query = state.get("sql_query", "").strip()
    if not sql_query:
        logger.error("[execute_sql] Không có câu lệnh SQL để thực thi.")
        return {
            "result": "",
            "error": "Không có câu lệnh SQL để thực thi.",
        }

    try:
        logger.info(f"[execute_sql] Đang gửi truy vấn DB: {sql_query}")
        start_ms = time.perf_counter() * 1000
        result = await asyncio.to_thread(get_db_manager().run_query, sql_query)
        elapsed_ms = time.perf_counter() * 1000 - start_ms

        if not result or str(result) == "[]":
            result_str = "Không tìm thấy dữ liệu nào thỏa mãn."
        else:
            result_str = str(result)

        if len(result_str) > MAX_RESULT_LENGTH:
            logger.warning(
                f"[execute_sql] Kết quả quá dài ({len(result_str)} chars), "
                f"truncate xuống {MAX_RESULT_LENGTH} chars."
            )
            result_str = result_str[:MAX_RESULT_LENGTH] + "\n... (kết quả đã bị cắt bớt)"

        logger.info(
            f"[execute_sql] Truy vấn thành công ({elapsed_ms:.1f}ms). "
            f"Result length: {len(result_str)} chars."
        )

        return {
            "result": result_str,
            "error": None,
        }

    except Exception as e:
        retries = state.get("retries", 0)
        logger.error(f"[execute_sql] Lỗi khi thực thi SQL (retry {retries}): {e}")
        return {
            "result": "",
            "error": str(e),
            "retries": retries + 1,
        }