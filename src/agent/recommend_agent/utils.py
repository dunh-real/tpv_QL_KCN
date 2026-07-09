import threading

from src.core.logger import get_logger

logger = get_logger(__name__)

_llm_instance = None
_llm_lock = threading.Lock()


def get_llm():
    """
    Trả về instance LLM duy nhất (thread-safe).
    """
    global _llm_instance
    if _llm_instance is None:
        with _llm_lock:
            if _llm_instance is None:
                from src.models.llm_qwen25 import Qwen25Model
                logger.info("[utils] Khởi tạo LLM instance cho Recommend Agent...")
                _llm_instance = Qwen25Model()

    return _llm_instance.get_llm()

