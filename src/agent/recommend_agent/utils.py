from src.core.logger import get_logger

logger = get_logger(__name__)


def get_llm():
    """
    Trả về instance LLM duy nhất (thread-safe thông qua hàm singleton của model).
    """
    from src.models.llm_qwen25 import get_qwen25_model
    return get_qwen25_model().get_llm()

