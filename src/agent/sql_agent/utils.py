import re
from src.core.logger import get_logger

logger = get_logger(__name__)


def get_llm():
    """
    Trả về instance LLM duy nhất
    """
    from src.models.llm_qwen25 import get_qwen25_model
    return get_qwen25_model().get_llm()

def get_schema_service():
    """
    Trả về instance SqlSchemaService duy nhất (delegates to existing singleton)
    """
    from src.services.schema_service import get_schema_service as _get_schema_service
    return _get_schema_service()


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
