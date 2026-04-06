import json
import redis
from typing import Optional, Any
from functools import wraps
from src.core.config import settings
from src.core.logger import get_logger

logger = get_logger(__name__)

class RedisCache:
    def __init__(self):
        try:
            self.client = redis.Redis(
                host=settings.REDIS_HOST,
                port=settings.REDIS_PORT,
                db=settings.REDIS_DB,
                decode_responses=True
            )
            self.client.ping()
            logger.info(f"Kết nối Redis thành công tại {settings.REDIS_HOST}:{settings.REDIS_PORT}")
        except Exception as e:
            logger.error(f"Lỗi kết nối Redis: {e}")
            self.client = None

    def get(self, key: str) -> Optional[Any]:
        if not self.client:
            return None
        value = self.client.get(key)
        if value:
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                return value
        return None

    def set(self, key: str, value: Any, ttl_seconds: int = 86400) -> bool:
        if not self.client:
            return False
        try:
            if isinstance(value, (dict, list)):
                value = json.dumps(value)
            self.client.setex(key, ttl_seconds, value)
            return True
        except Exception as e:
            logger.error(f"Lỗi khi set cache Redis: {e}")
            return False

redis_cache = RedisCache()

def cache_llm_response(ttl_seconds: int = 86400):
    """
    Decorator để tự động cache kết quả trả về của các hàm asyncio gọi LLM.
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            if not redis_cache.client:
                return await func(*args, **kwargs)
                
            import hashlib
            prompt_str = str(args) + str(kwargs)
            prompt_hash = hashlib.md5(prompt_str.encode('utf-8')).hexdigest()
            cache_key = f"llm_cache:{func.__name__}:{prompt_hash}"
            
            cached_result = redis_cache.get(cache_key)
            if cached_result is not None:
                logger.info(f"Cache HIT cho LLM: {cache_key}")
                return cached_result
                
            logger.info(f"Cache MISS cho LLM. Đang gọi AI...")
            result = await func(*args, **kwargs)
            
            if result and not str(result).startswith("Error:"):
                redis_cache.set(cache_key, result, ttl_seconds)
                
            return result
        return wrapper
    return decorator
