from src.agent.chat_agent.state import AgentState
from src.agent.chat_agent.utils import get_llm
from src.agent.chat_agent.schema import RouteDecision
from src.agent.chat_agent.prompts import ROUTE_PROMPT
from src.core.logger import get_logger

logger = get_logger(__name__)

_router_chain = None


def _get_router_chain():
    """Cache the structured-output chain — avoids recreating it every request."""
    global _router_chain
    if _router_chain is None:
        llm = get_llm()
        router_llm = llm.with_structured_output(RouteDecision)
        _router_chain = ROUTE_PROMPT | router_llm
    return _router_chain


async def route_node(state: AgentState):
    """LLM phân tích và quyết định hướng đi"""
    try:
        question = state["question"]
        chain = _get_router_chain()
        decision: RouteDecision = await chain.ainvoke({"question": question})
        return {"routes": decision.routes}
    except Exception as e:
        logger.error(f"[route_node] Lỗi khi phân tích route: {e}", exc_info=True)
        return {"routes": []}
