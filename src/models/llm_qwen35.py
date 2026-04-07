from src.core.config import settings
from langchain_openai import ChatOpenAI
from src.services.cache import cache_llm_response
import httpx
import json
class Qwen35Model:
    def __init__(self):
        self.model = ChatOpenAI(
            base_url=settings.LLM_BASE_URL,
            api_key=settings.LLM_API_KEY,
            model=settings.LLM_MODEL_QWEN35,
            temperature=0.7,
            max_tokens=settings.LLM_MAX_TOKENS
        )
    
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
        payload = {
            "model": settings.LLM_MODEL_QWEN35,
            "messages": [{"role": "user", "content": prompt}],
            "stream": True,
            "temperature": 0.0,
            "chat_template_kwargs": {
                "enable_thinking": True,
            }
        }
        
        url = f"{settings.LLM_BASE_URL.rstrip('/')}/chat/completions"
        headers = {"Authorization": f"Bearer {settings.LLM_API_KEY}"} if settings.LLM_API_KEY else {}
        try:
            async with httpx.AsyncClient() as client:
                async with client.stream("POST", url, json=payload, headers=headers, timeout=120.0) as response:
                    async for line in response.aiter_lines():
                        if line.startswith("data: ") and line != "data: [DONE]":
                            data_str = line[6:]
                            try:
                                data = json.loads(data_str)
                                delta = data.get("choices", [{}])[0].get("delta", {})
                                
                                content = delta.get("content", "")
                                reasoning = delta.get("reasoning", "")
                                if reasoning:
                                    yield reasoning
                                if content:
                                    yield content
                            except json.JSONDecodeError:
                                pass
        except Exception as e:
            yield f"Error: {str(e)}"



        
    
    
    