from src.services.retriever_service import get_retriever_service
from src.agent.chat_agent.state import AgentState
from src.core.logger import get_logger

logger = get_logger(__name__)


async def retrieve_node(state: AgentState):
    try:
        question = state["question"]
        retriever_service = get_retriever_service()
        docs = await retriever_service.retrieve(query=question)
        return {"vectordb_result": docs}
    except Exception as e:
        logger.error(f"[retrieve_node] Lỗi khi truy xuất VectorDB: {e}", exc_info=True)
        return {"vectordb_result": []}