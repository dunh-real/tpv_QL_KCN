
from langgraph.graph import StateGraph, END

from src.agent.generate_sql_agent.state import GenerateSQLState
from src.agent.generate_sql_agent.constants import MAX_RETRIES
from src.core.logger import get_logger

from src.agent.generate_sql_agent.nodes.fetch_schema import fetch_schema_node
from src.agent.generate_sql_agent.nodes.cache import check_cache_node, update_cache_node
from src.agent.generate_sql_agent.nodes.generate import generate_sql_node
from src.agent.generate_sql_agent.nodes.validate import validate_sql_node
from src.agent.generate_sql_agent.nodes.fallback import fallback_node

logger = get_logger(__name__)


def _route_after_cache(state: GenerateSQLState) -> str:
    """Cache hit → validate ngay; miss → generate mới."""
    if state.get("is_cache_hit"):
        logger.info("[generate_graph] Cache HIT → validate_sql")
        return "validate_sql"
    logger.info("[generate_graph] Cache MISS → generate_sql")
    return "generate_sql"


def _route_after_validation(state: GenerateSQLState) -> str:
    """
    SQL hợp lệ → kết thúc subgraph (trả về state cho main graph).
    SQL không hợp lệ:
      - Nếu EMPTY hoặc hết retry → kết thúc subgraph (với is_valid=False).
      - Còn retry → fallback để LLM tự sửa.
    """
    if state.get("is_valid"):
        logger.info("[generate_graph] SQL hợp lệ → update_cache")
        return "update_cache"

    retries = state.get("retries", 0)
    err_type = state.get("validation_error_type")

    if err_type == "EMPTY":
        logger.warning("[generate_graph] SQL rỗng (EMPTY) → END subgraph")
        return END

    if retries >= MAX_RETRIES:
        logger.error(f"[generate_graph] Hết {MAX_RETRIES} lần retry → END subgraph")
        return END

    logger.warning(
        f"[generate_graph] SQL không hợp lệ ({err_type}), "
        f"thử fallback lần {retries + 1}/{MAX_RETRIES}"
    )
    return "fallback"




def build_generate_graph() -> StateGraph:
    """
    Tạo và biên dịch subgraph generate_graph.

    Returns:
        CompiledStateGraph — có thể được nhúng trực tiếp vào main graph
        thông qua `workflow.add_node("generate_graph", build_generate_graph())`.
    """
    sg = StateGraph(GenerateSQLState)

    # Nodes
    sg.add_node("fetch_schema", fetch_schema_node)
    sg.add_node("check_cache", check_cache_node)
    sg.add_node("generate_sql", generate_sql_node)
    sg.add_node("validate_sql", validate_sql_node)
    sg.add_node("fallback", fallback_node)
    sg.add_node("update_cache", update_cache_node)

    # Entrypoint
    sg.set_entry_point("fetch_schema")

    # Edges
    sg.add_edge("fetch_schema", "check_cache")
    sg.add_conditional_edges("check_cache", _route_after_cache)
    sg.add_edge("generate_sql", "validate_sql")
    sg.add_conditional_edges("validate_sql", _route_after_validation)
    sg.add_edge("fallback", "validate_sql")
    sg.add_edge("update_cache", END)

    return sg.compile()




_generate_graph_instance = None

def get_generate_graph():
    global _generate_graph_instance
    if _generate_graph_instance is None:
        _generate_graph_instance = build_generate_graph()
    return _generate_graph_instance