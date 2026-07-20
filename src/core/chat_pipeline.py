import asyncio
from typing import Dict, Any, AsyncGenerator, Optional
from langchain_core.runnables import RunnableConfig

from src.agent.chat_agent.state import AgentState
from src.agent.chat_agent.schema import RouteDecision
from src.agent.chat_agent.utils import get_llm
from src.agent.chat_agent.prompts import ROUTE_PROMPT, FORMAT_ANSWER_PROMPT
from src.services.retriever_service import get_retriever_service
from src.services.sql_service import get_sql_service
from src.core.logger import get_logger

logger = get_logger(__name__)

async def _route_question(question: str) -> list[str]:
    """Sử dụng LLM để phân tích và quyết định hướng đi (route)"""
    try:
        llm = get_llm()
        router_llm = llm.with_structured_output(RouteDecision)
        chain = ROUTE_PROMPT | router_llm
        decision: RouteDecision = await chain.ainvoke({"question": question})
        return decision.routes
    except Exception as e:
        logger.error(f"[_route_question] Lỗi khi phân tích route: {e}", exc_info=True)
        return []

async def _retrieve_vectordb(question: str) -> list[Any]:
    """Gọi trực tiếp retriever_service"""
    try:
        retriever_service = get_retriever_service()
        docs = await retriever_service.retrieve(query=question)
        return docs
    except Exception as e:
        logger.error(f"[_retrieve_vectordb] Lỗi khi truy xuất VectorDB: {e}", exc_info=True)
        return []

async def _run_sql(question: str) -> dict[str, Any]:
    """Gọi trực tiếp sql_service"""
    try:
        sql_service = get_sql_service()
        result = await sql_service.run_sql_agent(question)
        if result.error or not result.is_valid:
            error_msg = result.error if result.error else "Không thể thực thi câu truy vấn SQL."
            return {"error": error_msg}
        return {"result": result.result, "query": result.sql_query}
    except Exception as e:
        logger.error(f"[_run_sql] Lỗi khi chạy SQL agent: {e}", exc_info=True)
        return {"error": str(e)}

def _format_vectordb_text(vectordb_result: list[Any]) -> str:
    vectordb_text = ""
    if vectordb_result:
        for idx, doc in enumerate(vectordb_result, 1):
            vectordb_text += f"--- Tài liệu {idx} ---\n"
            vectordb_text += f"Nội dung: {doc.page_content}\n"
            vectordb_text += f"Metadata: {doc.metadata}\n\n"
    else:
        vectordb_text = "Không có thông tin từ tài liệu văn bản."
    return vectordb_text

def _format_sql_text(sql_result: dict[str, Any]) -> str:
    sql_text = ""
    if sql_result:
        if "error" in sql_result:
            sql_text = f"Lỗi khi truy xuất dữ liệu: {sql_result['error']}"
        else:
            sql_text = f"Kết quả truy vấn: {sql_result.get('result', '')}\n(Câu SQL đã dùng: {sql_result.get('query', '')})"
    else:
        sql_text = "Không có thông tin từ cơ sở dữ liệu."
    return sql_text

async def run_chat_pipeline(question: str, config: Optional[RunnableConfig] = None) -> Dict[str, Any]:
    """
    Luồng xử lý câu hỏi gọi trực tiếp các service.
    Trả về toàn bộ trạng thái cuối cùng.
    """
    state: AgentState = {"question": question, "routes": [], "vectordb_result": [], "sql_result": {}}
    
    # 1. Routing
    routes = await _route_question(question)
    state["routes"] = routes
    valid_routes = {"retrieve_vectordb", "run_sql_agent"}
    next_routes = [r for r in routes if r in valid_routes]
    
    # 2. Thực thi song song các service dựa trên routes
    tasks = []
    task_names = []
    
    if "retrieve_vectordb" in next_routes:
        tasks.append(_retrieve_vectordb(question))
        task_names.append("retrieve_vectordb")
    if "run_sql_agent" in next_routes:
        tasks.append(_run_sql(question))
        task_names.append("run_sql_agent")
        
    if tasks:
        results = await asyncio.gather(*tasks)
        for name, res in zip(task_names, results):
            if name == "retrieve_vectordb":
                state["vectordb_result"] = res
            elif name == "run_sql_agent":
                state["sql_result"] = res

    # 3. Format Answer
    vectordb_text = _format_vectordb_text(state.get("vectordb_result", []))
    sql_text = _format_sql_text(state.get("sql_result", {}))

    try:
        llm = get_llm()
        chain = (FORMAT_ANSWER_PROMPT | llm).with_config({"tags": ["format_answer"]})
        
        response_content = ""
        async for chunk in chain.astream(
            {
                "question": question,
                "vectordb_result": vectordb_text,
                "sql_result": sql_text
            },
            config=config
        ):
            response_content += chunk.content
        
        state["final_answer"] = response_content
    except Exception as e:
        logger.error(f"[run_chat_pipeline] Lỗi khi sinh câu trả lời: {e}", exc_info=True)
        state["final_answer"] = "Xin lỗi, đã có lỗi xảy ra khi xử lý câu trả lời. Vui lòng thử lại."

    return state


async def stream_chat_pipeline(question: str, config: Optional[RunnableConfig] = None) -> AsyncGenerator[str, None]:
    """
    Luồng xử lý câu hỏi gọi trực tiếp các service, không dùng Node của LangGraph, stream kết quả trả về.
    """
    # 1. Routing
    routes = await _route_question(question)
    valid_routes = {"retrieve_vectordb", "run_sql_agent"}
    next_routes = [r for r in routes if r in valid_routes]
    
    # 2. Thực thi song song các service
    tasks = []
    task_names = []
    vectordb_result = []
    sql_result = {}
    
    if "retrieve_vectordb" in next_routes:
        tasks.append(_retrieve_vectordb(question))
        task_names.append("retrieve_vectordb")
    if "run_sql_agent" in next_routes:
        tasks.append(_run_sql(question))
        task_names.append("run_sql_agent")
        
    if tasks:
        results = await asyncio.gather(*tasks)
        for name, res in zip(task_names, results):
            if name == "retrieve_vectordb":
                vectordb_result = res
            elif name == "run_sql_agent":
                sql_result = res

    # 3. Format answer và stream kết quả
    vectordb_text = _format_vectordb_text(vectordb_result)
    sql_text = _format_sql_text(sql_result)

    try:
        llm = get_llm()
        chain = (FORMAT_ANSWER_PROMPT | llm).with_config({"tags": ["format_answer"]})
        
        async for chunk in chain.astream(
            {
                "question": question,
                "vectordb_result": vectordb_text,
                "sql_result": sql_text
            },
            config=config
        ):
            if chunk.content:
                yield chunk.content
    except Exception as e:
        logger.error(f"[stream_chat_pipeline] Lỗi khi sinh câu trả lời: {e}", exc_info=True)
        yield "Xin lỗi, đã có lỗi xảy ra khi xử lý câu trả lời. Vui lòng thử lại."
