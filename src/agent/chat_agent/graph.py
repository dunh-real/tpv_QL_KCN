from langgraph.graph import StateGraph, END
from src.agent.chat_agent.state import AgentState
from src.agent.chat_agent.nodes.route import route_node
from src.agent.chat_agent.nodes.retrieve_vectordb import retrieve_node
from src.agent.chat_agent.nodes.run_sql_agent import run_sql_agent_node
from src.agent.chat_agent.nodes.format_answer import format_answer_node

_VALID_ROUTES = {"retrieve_vectordb", "run_sql_agent"}


def route_condition(state: AgentState):
    routes = state.get("routes", [])
    if not isinstance(routes, list) or not routes:
        return "format_answer"
    next_nodes = [r for r in routes if r in _VALID_ROUTES]
    return next_nodes if next_nodes else "format_answer"


def build_graph():
    workflow = StateGraph(AgentState)

    # Add nodes
    workflow.add_node("route", route_node)
    workflow.add_node("retrieve_vectordb", retrieve_node)
    workflow.add_node("run_sql_agent", run_sql_agent_node)
    workflow.add_node("format_answer", format_answer_node)

    # Set entry point
    workflow.set_entry_point("route")

    # Add conditional edges
    workflow.add_conditional_edges(
        "route",
        route_condition,
        {
            "retrieve_vectordb": "retrieve_vectordb",
            "run_sql_agent": "run_sql_agent",
            "format_answer": "format_answer"
        }
    )

    # Add edges to format_answer
    workflow.add_edge("retrieve_vectordb", "format_answer")
    workflow.add_edge("run_sql_agent", "format_answer")
    
    # End workflow
    workflow.add_edge("format_answer", END)

    return workflow.compile()


_compiled_graph = None


def get_chat_agent():
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_graph()
    return _compiled_graph
