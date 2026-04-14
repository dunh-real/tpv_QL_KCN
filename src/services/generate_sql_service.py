from __future__ import annotations

from typing import Optional
from dataclasses import dataclass

from langchain_core.messages import HumanMessage

from src.agent.generate_sql_agent.graph import get_generate_graph
from src.core.logger import get_logger

logger = get_logger(__name__)


@dataclass
class GenerateSQLResult:
    """Kết quả trả về từ GenerateSQLService."""
    sql_query: str
    is_valid: bool
    is_cache_hit: bool
    validation_error_type: Optional[str]
    error: Optional[str]


class GenerateSQLService:
    """
    Service layer bọc generate_graph subgraph.

    Nhận câu hỏi tự nhiên và trả về sql_query đã được validate,
    kèm metadata (cache hit, trạng thái validate, lỗi nếu có).

    Ví dụ sử dụng:
        service = GenerateSQLService()
        result = await service.generate("Có bao nhiêu nhân viên nữ?")
        if result.is_valid:
            print(result.sql_query)
    """

    def __init__(self) -> None:
        self._graph = get_generate_graph()
        logger.info("[GenerateSQLService] Khởi tạo thành công.")

    async def generate(
        self,
        question: str,
    ) -> GenerateSQLResult:
        """
        Sinh và validate câu lệnh SQL từ câu hỏi tự nhiên.

        Args:
            question:  Câu hỏi của người dùng (tiếng Việt hoặc tiếng Anh).

        Returns:
            GenerateSQLResult với đầy đủ thông tin về câu SQL và trạng thái.
        """
        logger.info(f"[GenerateSQLService] Đang xử lý câu hỏi: '{question}'")

        initial_state = {
            "messages": [HumanMessage(content=question)],
            "question": question,
            "retries": 0,
        }

        try:
            result_state = await self._graph.ainvoke(initial_state)
        except Exception as e:
            logger.error(f"[GenerateSQLService] Lỗi khi chạy graph: {e}")
            return GenerateSQLResult(
                sql_query="",
                is_valid=False,
                is_cache_hit=False,
                validation_error_type="RUNTIME_ERROR",
                error=str(e),
            )

        sql_query = result_state.get("sql_query", "")
        is_valid = result_state.get("is_valid", False)
        is_cache_hit = result_state.get("is_cache_hit", False)
        err_type = result_state.get("validation_error_type")
        error = result_state.get("error")

        logger.info(
            f"[GenerateSQLService] Kết quả — valid={is_valid}, "
            f"cache_hit={is_cache_hit}, err_type={err_type}"
        )

        return GenerateSQLResult(
            sql_query=sql_query,
            is_valid=is_valid,
            is_cache_hit=is_cache_hit,
            validation_error_type=err_type,
            error=error,
        )
