import threading

from langgraph.graph import StateGraph, END
from langgraph.graph.state import CompiledStateGraph
from src.agent.recommend_agent.nodes.answer import answer_node
from src.core.logger import get_logger
from src.agent.recommend_agent.state import RecommendAgentState

logger = get_logger(__name__)


def build_recommend_graph() -> CompiledStateGraph:
    """Xay dung recommend agent workflow"""
    workflow = StateGraph(RecommendAgentState)
    workflow.add_node("answer", answer_node)
    workflow.add_edge("answer", END)
    workflow.set_entry_point("answer")
    return workflow.compile()


_recommend_graph = None
_graph_lock = threading.Lock()


def get_recommend_graph() -> CompiledStateGraph:
    """Tra ve instance recommend agent (thread-safe)."""
    global _recommend_graph
    if _recommend_graph is None:
        with _graph_lock:
            if _recommend_graph is None:
                logger.info("[Recommend Agent] Khoi tao recommend agent graph...")
                _recommend_graph = build_recommend_graph()
    return _recommend_graph