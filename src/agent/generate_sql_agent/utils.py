import re
from src.core.logger import get_logger

logger = get_logger(__name__)

_llm_instance = None


def get_llm():
    """
    Trả về instance LLM duy nhất
    """
    global _llm_instance
    if _llm_instance is None:
        from src.models.llm_qwen25 import Qwen25Model
        logger.info("[utils] Khởi tạo LLM instance cho SQL Agent...")
        _llm_instance = Qwen25Model()

    return _llm_instance.get_llm()


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
