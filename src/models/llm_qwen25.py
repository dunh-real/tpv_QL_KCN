from src.core.config import settings
from langchain_openai import ChatOpenAI
from src.services.cache import cache_llm_response

class Qwen25Model:
    def __init__(self):
        self.model = ChatOpenAI(
            base_url=settings.LLM_BASE_URL,
            api_key=settings.LLM_API_KEY,
            model=settings.LLM_MODEL_QWEN25,
            temperature=0.7,
            max_tokens=settings.LLM_MAX_TOKENS)
            
    def get_llm(self):
        return self.model

    @cache_llm_response(ttl_seconds=86400)
    async def generate(self,prompt:str) -> str :
        """
        Generate a response from the LLM model.
        Args:
            prompt (str): The prompt to send to the LLM model.
        Returns:
            str: The response from the LLM model.
        """
        try:
            response = await self.model.ainvoke(prompt)
            return response.content
        except Exception as e:
            return f"Error: {e}"
    async def generate_stream(self,prompt:str) -> str :
        """
        Generate a response from the LLM model in a streaming fashion.
        Args:
            prompt (str): The prompt to send to the LLM model.
        Returns:
            str: The response from the LLM model.
        """
        try:
            async for chunk in self.model.astream(prompt):
                yield chunk.content
        except Exception as e:
            yield f"Error: {str(e)}"

_qwen25_instance: Qwen25Model | None = None

def get_qwen25_model() -> Qwen25Model:
    """Lazy singleton — tạo Qwen25Model 1 lần duy nhất."""
    global _qwen25_instance
    if _qwen25_instance is None:
        _qwen25_instance = Qwen25Model()
    return _qwen25_instance