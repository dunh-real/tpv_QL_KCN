from __future__ import annotations

from typing import Optional
from dataclasses import dataclass

from src.agent.recommend_agent.graph import get_recommend_graph
from src.core.logger import get_logger

logger = get_logger(__name__)


@dataclass
class RecommendResult:
    """Kết quả trả về từ RecommendService."""

    response: str
    error: Optional[str]


class RecommendService:
    """
    Service gọi recommend_graph → trả về state.
    """

    def __init__(self) -> None:
        self._graph = get_recommend_graph()
        logger.info("[RecommendService] Khởi tạo thành công.")

    async def recommend(
        self,
        question: str,
    ) -> RecommendResult:
        """
        Retrieve context → gọi recommend agent → trả về kết quả.
        """
        logger.info(f"[RecommendService] Đang xử lý câu hỏi: '{question}'")

        try:
            result_state = await self._graph.ainvoke({
                "question": question,
            })
        except Exception as e:
            logger.error(f"[RecommendService] Lỗi khi chạy graph: {e}", exc_info=True)
            return RecommendResult(
                response="Xin lỗi, đã xảy ra lỗi hệ thống. Vui lòng thử lại sau.",
                error=str(e),
            )

        response = result_state.get("response", "")
        logger.info(f"[RecommendService] Hoàn thành — response dài {len(response)} ký tự.")

        return RecommendResult(
            response=response,
            error=None,
        )
