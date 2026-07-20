
from src.agent.sql_agent.state import SQLState
from src.agent.sql_agent.prompts import GENERATE_SQL_PROMPT
from src.agent.sql_agent.utils import get_llm, clean_output
from src.core.logger import get_logger

logger = get_logger(__name__)

_generate_chain = None


def _get_generate_chain():
    global _generate_chain
    if _generate_chain is None:
        _generate_chain = GENERATE_SQL_PROMPT | get_llm()
    return _generate_chain


async def generate_sql_node(state: SQLState) -> dict:
    """
    Gọi LLM để sinh câu lệnh SQL từ câu hỏi tự nhiên của người dùng.

    Input state cần có:
        - state["question"]: câu hỏi của user
        - state["db_schema"]: schema DB (được fetch bởi fetch_schema_node)

    Returns:
        {'sql_query': <sql>, 'retries': 0, 'error': None}
    """
    question = state["question"]
    schema = state.get("db_schema", "")
    if not schema:
        logger.warning("[generate_sql] db_schema trống trong state! LLM sẽ thiếu context.")

    logger.info(f"[generate_sql] Đang sinh SQL cho câu hỏi: '{question}'")

    chain = _get_generate_chain()

    try:
        response = await chain.ainvoke({
            "schema": schema,
            "question": question,
        })
        sql_query = clean_output(response.content)
        logger.info(f"[generate_sql] SQL được sinh: {sql_query[:120]}...")
        return {
            "sql_query": sql_query,
            "retries": 0,
            "error": None,
        }
    except Exception as e:
        logger.error(f"[generate_sql] LLM gặp lỗi: {e}")
        return {
            "sql_query": "",
            "retries": 0,
            "error": str(e),
        }

