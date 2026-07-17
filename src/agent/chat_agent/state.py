import operator
from typing import TypedDict, List, Optional, Dict, Any, Annotated


def _merge_dict(old: dict, new: dict) -> dict:
    merged = {**old}
    merged.update(new)
    return merged


class AgentState(TypedDict, total=False):
    question: str
    routes: list[str]
    vectordb_result: Annotated[list[Any], operator.add]
    sql_result: Annotated[dict[str, Any], _merge_dict]
    final_answer: str

