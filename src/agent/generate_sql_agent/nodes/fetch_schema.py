import asyncio
from src.agent.generate_sql_agent.state import GenerateSQLState
from src.db.mysql import db_manager
from src.core.logger import get_logger
from src.agent.generate_sql_agent.utils import get_schema_service

logger = get_logger(__name__)


async def fetch_schema_node(state: GenerateSQLState) -> dict:
    """
    Fetch DDL schema từ MySQL và lưu vào state thông qua quy trình RAG + GraphRAG:
      1. Retrieve vector DB để lấy list table liên quan đến câu hỏi.
      2. Dùng Graph (get_related_tables) để tìm các bảng kết nối (đủ để join).
      3. Trích xuất DDL thực tế (get_schema_ddl).
    """
    schema_service = get_schema_service()
    question = state.get("question")
    
    logger.info(f"[fetch_schema] 1. Truy vấn Vector DB lấy bảng liên quan: '{question}'")
    
    target_tables = await schema_service.search_tables(question, top_k=3)
    logger.info(f"[fetch_schema] Các bảng từ Vector DB: {target_tables}")
    
    if target_tables:
        logger.info("[fetch_schema] 2. GraphRAG: Mở rộng các bảng qua khóa ngoại...")
        expanded_tables = db_manager.get_related_tables(target_tables)
        logger.info(f"[fetch_schema] Các bảng sau mở rộng: {expanded_tables}")
    else:
        logger.warning("[fetch_schema] Không tìm thấy bảng nào từ Vector DB.")
        expanded_tables = []
        
    logger.info("[fetch_schema] 3. Lấy DDL từ database...")
    schema_ddl = ""
    if expanded_tables:
        # Lấy DDL từ database thực tế
        schema_ddl = await asyncio.to_thread(db_manager.get_schema_ddl, expanded_tables)
        
    logger.info(f"[fetch_schema] Hoàn tất fetch schema (kích thước {len(schema_ddl)} chars).")
    
    return {"db_schema": schema_ddl}
