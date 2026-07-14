from src.agent.chat_agent.state import AgentState
from src.agent.chat_agent.utils import get_llm
from src.agent.chat_agent.schema import RouteDecision
from src.agent.chat_agent.prompts import ROUTE_PROMPT
from src.core.logger import get_logger

logger = get_logger(__name__)


async def route_node(state: AgentState):
    """LLM phân tích và quyết định hướng đi"""
    try:
        question = state["question"]
        llm = get_llm()
        router_llm = llm.with_structured_output(RouteDecision)
        chain = ROUTE_PROMPT | router_llm
        decision: RouteDecision = await chain.ainvoke({"question": question})
        return {"routes": decision.routes}
    except Exception as e:
        logger.error(f"[route_node] Lỗi khi phân tích route: {e}", exc_info=True)
        return {"routes": []}
