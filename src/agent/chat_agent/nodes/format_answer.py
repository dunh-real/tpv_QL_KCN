from src.agent.chat_agent.state import AgentState
from src.agent.chat_agent.utils import get_llm
from src.agent.chat_agent.prompts import FORMAT_ANSWER_PROMPT
from langchain_core.runnables import RunnableConfig
from src.core.logger import get_logger

logger = get_logger(__name__)

_format_chain = None


def _get_format_chain():
    """Cache the format-answer chain — avoids recreating it every request."""
    global _format_chain
    if _format_chain is None:
        llm = get_llm()
        _format_chain = (FORMAT_ANSWER_PROMPT | llm).with_config({"tags": ["format_answer"]})
    return _format_chain


async def format_answer_node(state: AgentState, config: RunnableConfig = None):
    question = state["question"]
    vectordb_result = state.get("vectordb_result", [])
    sql_result = state.get("sql_result", {})
    
    vectordb_text = ""
    if vectordb_result:
        for idx, doc in enumerate(vectordb_result, 1):
            vectordb_text += f"--- Tài liệu {idx} ---\n"
            vectordb_text += f"Nội dung: {doc.page_content}\n"
            vectordb_text += f"Metadata: {doc.metadata}\n\n"
    else:
        vectordb_text = "Không có thông tin từ tài liệu văn bản."

    sql_text = ""
    if sql_result:
        if "error" in sql_result:
            sql_text = f"Lỗi khi truy xuất dữ liệu: {sql_result['error']}"
        else:
            sql_text = f"Kết quả truy vấn: {sql_result.get('result', '')}\n(Câu SQL đã dùng: {sql_result.get('query', '')})"
    else:
        sql_text = "Không có thông tin từ cơ sở dữ liệu."

    try:
        chain = _get_format_chain()
        
        parts = []
        async for chunk in chain.astream(
            {
                "question": question,
                "vectordb_result": vectordb_text,
                "sql_result": sql_text
            },
            config=config
        ):
            if chunk.content:
                parts.append(chunk.content)
        
        return {"final_answer": "".join(parts)}
    except Exception as e:
        logger.error(f"[format_answer_node] Lỗi khi sinh câu trả lời: {e}", exc_info=True)
        return {"final_answer": "Xin lỗi, đã có lỗi xảy ra khi xử lý câu trả lời. Vui lòng thử lại."}
