import operator
from typing import TypedDict, List, Optional, Dict, Any, Annotated


class AgentState(TypedDict, total=False):
    question: str
    routes: list[str]
    vectordb_result: Annotated[list[Any], operator.add]
    sql_result: dict[str, Any]
    final_answer: str
