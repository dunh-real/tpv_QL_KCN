import threading

from langgraph.graph import StateGraph, END

from src.agent.sql_agent.state import SQLState
from src.agent.sql_agent.constants import MAX_RETRIES
from src.core.logger import get_logger

from src.agent.sql_agent.nodes.fetch_schema import fetch_schema_node
from src.agent.sql_agent.nodes.cache import check_cache_node, update_cache_node
from src.agent.sql_agent.nodes.generate import generate_sql_node
from src.agent.sql_agent.nodes.validate import validate_sql_node
from src.agent.sql_agent.nodes.fallback import fallback_node
from src.agent.sql_agent.nodes.execute import execute_sql_node

logger = get_logger(__name__)



def _route_after_cache(state: SQLState) -> str:
    """Cache hit → validate ngay; miss → generate mới."""
    if state.get("is_cache_hit"):
        logger.info("[graph] Cache HIT → validate_sql")
        return "validate_sql"
    logger.info("[graph] Cache MISS → sql")
    return "sql"


def _route_after_validation(state: SQLState) -> str:
    """
    SQL hợp lệ → update_cache → execute.
    SQL không hợp lệ:
      - Nếu EMPTY hoặc hết retry → kết thúc subgraph (với is_valid=False).
      - Còn retry → fallback để LLM tự sửa.
    """
    if state.get("is_valid"):
        logger.info("[graph] SQL hợp lệ → update_cache")
        return "update_cache"

    retries = state.get("retries", 0)
    err_type = state.get("validation_error_type")

    if err_type == "EMPTY":
        logger.warning("[graph] SQL rỗng (EMPTY) → END subgraph")
        return END

    if retries >= MAX_RETRIES:
        logger.error(f"[graph] Hết {MAX_RETRIES} lần retry → END subgraph")
        return END

    logger.warning(
        f"[graph] SQL không hợp lệ ({err_type}), "
        f"thử fallback lần {retries + 1}/{MAX_RETRIES}"
    )
    return "fallback"


def _route_after_execute(state: SQLState) -> str:
    """
    Execute thành công (không error) → END.
    Execute thất bại:
      - Còn retry → fallback để LLM sửa dựa trên DB error.
      - Hết retry → END (giữ error trong state).
    """
    if not state.get("error"):
        logger.info("[graph] Execute thành công → END")
        return END

    retries = state.get("retries", 0)
    if retries >= MAX_RETRIES:
        logger.error(
            f"[graph] Execute thất bại và đã hết {MAX_RETRIES} lần retry → END"
        )
        return END

    logger.warning(
        f"[graph] Execute thất bại, chuyển fallback lần {retries + 1}/{MAX_RETRIES}. "
        f"Lỗi: {state.get('error', '')[:100]}"
    )
    return "fallback"



def build_graph() -> StateGraph:
    """
    Tạo và biên dịch sql_agent subgraph.

    Luồng:
        fetch_schema → check_cache
            ├─ (HIT)  → validate_sql
            └─ (MISS) → sql → validate_sql
                              ├─ (valid)   → update_cache → execute_sql
                              │                              ├─ (ok)    → END
                              │                              └─ (error) → fallback → validate_sql (loop)
                              ├─ (invalid) → fallback → validate_sql  (loop)
                              └─ (EMPTY / hết retry) → END

    Returns:
        CompiledStateGraph — có thể nhúng vào main graph
        thông qua `workflow.add_node("sql_agent", build_graph())`.
    """
    graph = StateGraph(SQLState)

    # --- Nodes ---
    graph.add_node("fetch_schema", fetch_schema_node)
    graph.add_node("check_cache", check_cache_node)
    graph.add_node("sql", generate_sql_node)
    graph.add_node("validate_sql", validate_sql_node)
    graph.add_node("fallback", fallback_node)
    graph.add_node("update_cache", update_cache_node)
    graph.add_node("execute_sql", execute_sql_node)

    # --- Entry ---
    graph.set_entry_point("fetch_schema")

    # --- Edges ---
    graph.add_edge("fetch_schema", "check_cache")
    graph.add_conditional_edges("check_cache", _route_after_cache)
    graph.add_edge("sql", "validate_sql")
    graph.add_conditional_edges("validate_sql", _route_after_validation)
    graph.add_edge("fallback", "validate_sql")
    graph.add_edge("update_cache", "execute_sql")
    graph.add_conditional_edges("execute_sql", _route_after_execute)

    return graph.compile()

_graph_instance = None
_graph_lock = threading.Lock()


def get_graph():
    """Trả về compiled graph singleton (thread-safe)."""
    global _graph_instance
    if _graph_instance is None:
        with _graph_lock:
            if _graph_instance is None:
                _graph_instance = build_graph()
                logger.info("[graph] SQL Agent graph đã được compile.")
    return _graph_instance