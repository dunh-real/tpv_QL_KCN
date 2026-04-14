
from src.agent.generate_sql_agent.state import GenerateSQLState
from src.agent.generate_sql_agent.prompts import FIX_SQL_PROMPT
from src.agent.generate_sql_agent.utils import get_llm, clean_output
from src.core.logger import get_logger

logger = get_logger(__name__)


async def fallback_node(state: GenerateSQLState) -> dict:
    """
    Yêu cầu LLM sửa lại câu SQL khi SQL bị lỗi (cú pháp, chạy thất bại, hoặc bị chặn bởi security).
    """
    retries = state.get("retries", 0)
    logger.info(f"[fallback] Đang yêu cầu LLM tự sửa lỗi (Lần {retries + 1})...")

    schema = state.get("db_schema", "")
    question = state.get("question", "")
    sql_query = state.get("sql_query", "")
    error_msg = state.get("error", "Không rõ lỗi")

    llm = get_llm()
    chain = FIX_SQL_PROMPT | llm

    try:
        response = await chain.ainvoke({
            "schema": schema,
            "question": question,
            "sql_query": sql_query,
            "error": error_msg
        })

        new_sql = clean_output(response.content)

        logger.info(f"[fallback] SQL sau khi sửa: {new_sql[:100]}...")

        return {
            "sql_query": new_sql,
            "retries": retries + 1,
            "error": None,
            "is_valid": False
        }
    except Exception as e:
        logger.error(f"[fallback] Lỗi LLM khi cố gắng sửa SQL: {e}")
        return {
            "retries": retries + 1,
            "error": f"Lỗi nghiêm trậng khi fallback: {str(e)}",
            "is_valid": False
        }