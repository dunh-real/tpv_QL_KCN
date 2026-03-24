from src.core.config import settings
from langchain_openai import ChatOpenAI

class Qwen25Model:
    def __init__(self):
        self.model = ChatOpenAI(
            base_url=settings.LLM_BASE_URL,
            api_key=settings.LLM_API_KEY,
            model=settings.LLM_MODEL_QWEN25,
            temperature=0.7)
    async def generate(self,prompt:str) -> str :
        try:
            response = await self.model.ainvoke(prompt)
            return response.content
        except Exception as e:
            return f"Error: {e}"
    async def generate_stream(self,prompt:str) -> str :
        try:
            async for chunk in self.model.astream(prompt):
                yield chunk.content
        except Exception as e:
            yield f"Error: {str(e)}"