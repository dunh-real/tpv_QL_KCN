import re
from src.core.logger import get_logger

logger = get_logger(__name__)

_schema_service_instance = None

def get_llm():
    """
    Trả về instance LLM duy nhất
    """
    from src.models.llm_qwen25 import get_qwen25_model
    return get_qwen25_model().get_llm()

def get_schema_service():
    """
    Trả về instance SqlSchemaService duy nhất
    """
    global _schema_service_instance
    if _schema_service_instance is None:
        from src.services.schema_service import SqlSchemaService
        logger.info("[utils] Khởi tạo SqlSchemaService instance cho SQL Agent...")
        _schema_service_instance = SqlSchemaService()

    return _schema_service_instance


def clean_output(raw: str) -> str:
    """
    Loại bỏ phần "thinking" của Qwen3 (<think>...</think>)
    và markdown code fences (```sql ... ```) mà LLM hay thêm vào.
    """
    cleaned = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL)
    cleaned = re.sub(r"<think>.*", "", cleaned, flags=re.DOTALL)
    cleaned = re.sub(r"```[a-zA-Z]*\s*", "", cleaned)
    cleaned = cleaned.replace("```", "")
    if cleaned in ["", '""', "''", "NULL", "None", "NO_SQL"]:
        return ""
    return cleaned.strip()
