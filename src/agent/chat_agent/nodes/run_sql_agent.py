from src.agent.chat_agent.state import AgentState
from src.services.sql_service import get_sql_service
from src.core.logger import get_logger

logger = get_logger(__name__)


async def run_sql_agent_node(state: AgentState):
    try:
        sql_service = get_sql_service()
        result = await sql_service.run_sql_agent(state["question"])
        if result.error or not result.is_valid:
            error_msg = result.error if result.error else "Không thể thực thi câu truy vấn SQL."
            return {"sql_result": {"error": error_msg}}
        return {"sql_result": {"result": result.result, "query": result.sql_query}}
    except Exception as e:
        logger.error(f"[run_sql_agent_node] Lỗi khi chạy SQL agent: {e}", exc_info=True)
        return {"sql_result": {"error": str(e)}}