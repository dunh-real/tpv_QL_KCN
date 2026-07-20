from __future__ import annotations

from typing import Optional
from dataclasses import dataclass

from langchain_core.messages import HumanMessage

from src.agent.sql_agent.graph import get_graph
from src.core.logger import get_logger

logger = get_logger(__name__)


@dataclass
class SQLAgentResult:
    """Kết quả trả về từ SQLAgentService."""
    sql_query: str
    result: str
    is_valid: bool
    is_cache_hit: bool
    validation_error_type: Optional[str]
    error: Optional[str]


class SQLAgentService:
    """
    Service layer bọc sql_agent subgraph.

    Nhận câu hỏi tự nhiên và trả về sql_query đã được validate + execute,
    kèm metadata (cache hit, trạng thái validate, lỗi nếu có).

    Ví dụ sử dụng:
        service = SQLAgentService()
        result = await service.generate("Có bao nhiêu nhân viên nữ?")
        if result.is_valid:
            print(result.result)
    """

    def __init__(self) -> None:
        self._graph = get_graph()
        logger.info("[SQLAgentService] Khởi tạo thành công.")

    async def run_sql_agent(
        self,
        question: str,
    ) -> SQLAgentResult:
        """
        Sinh, validate và execute câu lệnh SQL từ câu hỏi tự nhiên.

        Args:
            question:  Câu hỏi của người dùng (tiếng Việt hoặc tiếng Anh).

        Returns:
            SQLAgentResult với đầy đủ thông tin về câu SQL, kết quả và trạng thái.
        """
        logger.info(f"[SQLAgentService] Đang xử lý câu hỏi: '{question}'")

        initial_state = {
            "messages": [HumanMessage(content=question)],
            "question": question,
            "retries": 0,
        }

        try:
            result_state = await self._graph.ainvoke(initial_state)
        except Exception as e:
            logger.error(f"[SQLAgentService] Lỗi khi chạy graph: {e}")
            return SQLAgentResult(
                sql_query="",
                result="",
                is_valid=False,
                is_cache_hit=False,
                validation_error_type="RUNTIME_ERROR",
                error=str(e),
            )

        sql_query = result_state.get("sql_query", "")
        result = result_state.get("result", "")
        is_valid = result_state.get("is_valid", False)
        is_cache_hit = result_state.get("is_cache_hit", False)
        err_type = result_state.get("validation_error_type")
        error = result_state.get("error")

        logger.info(
            f"[SQLAgentService] Kết quả — valid={is_valid}, "
            f"cache_hit={is_cache_hit}, err_type={err_type}"
        )

        return SQLAgentResult(
            sql_query=sql_query,
            result=result,
            is_valid=is_valid,
            is_cache_hit=is_cache_hit,
            validation_error_type=err_type,
            error=error,
        )


_sql_service_instance: SQLAgentService | None = None


def get_sql_service() -> SQLAgentService:
    """Lazy singleton — tạo SQLAgentService 1 lần duy nhất."""
    global _sql_service_instance
    if _sql_service_instance is None:
        logger.info("[singleton] Khởi tạo SQLAgentService instance...")
        _sql_service_instance = SQLAgentService()
    return _sql_service_instance
