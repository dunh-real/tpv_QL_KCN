import asyncio
from src.agent.generate_sql_agent.state import GenerateSQLState
from src.db.postgresql import get_db_schema
from src.core.logger import get_logger

logger = get_logger(__name__)


async def fetch_schema_node(state: GenerateSQLState) -> dict:
    """
    Fetch DDL schema từ PostgreSQL và lưu vào state.

    Returns:
        {'db_schema': <schema_string>}

    Raises:
        Nếu DB chưa kết nối, raise ConnectionError (sẽ kết thúc graph).
    """
    if state.get("db_schema"):
        logger.debug("[fetch_schema] Schema đã có trong state, bỏ qua fetch.")
        return {}

    logger.info("[fetch_schema] Đang fetch DB schema...")
    try:
        schema = await asyncio.to_thread(get_db_schema)
        logger.info(f"[fetch_schema] Fetch thành công. Schema size: {len(schema)} chars.")
        return {"db_schema": schema}
    except ConnectionError as e:
        logger.error(f"[fetch_schema] Không thể kết nối DB: {e}")
        raise
    except Exception as e:
        logger.error(f"[fetch_schema] Lỗi không mong đợi khi fetch schema: {e}")
        raise
