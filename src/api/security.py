import asyncio
import time

from fastapi import HTTPException, Security, Request, status
from fastapi.security import APIKeyHeader
from src.core.config import settings
from src.core.logger import get_logger
from src.services.cache import get_redis_cache

logger = get_logger(__name__)

API_KEY_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=True)

# Parse valid keys from config
VALID_API_KEYS = set([k.strip() for k in settings.API_KEYS.split(",") if k.strip()])

async def verify_api_key(api_key: str = Security(api_key_header)):
    """
    Xác thực API Key truyền vào header `X-API-Key`.
    """
    if not VALID_API_KEYS:
        logger.warning("Chưa cấu hình API_KEYS trong biến môi trường. Bỏ qua xác thực.")
        return api_key

    if api_key not in VALID_API_KEYS:
        logger.warning(f"Từ chối truy cập: API Key không hợp lệ '***{api_key[-4:]}'")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Could not validate API KEY",
        )
    return api_key

async def rate_limiter(request: Request):
    """
    Redis-based Fixed Window Rate Limiter (async-safe).
    Giới hạn theo IP của client (số request tối đa trong 1 phút).
    Redis I/O chạy trong thread pool để không block event loop.
    """
    if settings.RATE_LIMIT_PER_MINUTE <= 0:
        return True

    user_id = request.headers.get("X-User-ID")
    forwarded_for = request.headers.get("X-Forwarded-For")

    if user_id:
        identifier = f"user:{user_id}"
    elif forwarded_for:
        real_ip = forwarded_for.split(",")[0].strip()
        identifier = f"ip:{real_ip}"
    else:
        client_ip = request.client.host if request.client else "unknown_ip"
        identifier = f"ip:{client_ip}"

    redis_cache = get_redis_cache()

    if not redis_cache.client:
        logger.warning("Redis không khả dụng, bỏ qua Rate Limiting.")
        return True

    current_minute = int(time.time() // 60)
    key = f"rate_limit:{identifier}:{current_minute}"

    def _check_rate_limit():
        """Chạy trong thread pool vì redis-py là sync blocking."""
        req_count = redis_cache.client.incr(key)
        if req_count == 1:
            redis_cache.client.expire(key, 60)
        return req_count

    try:
        req_count = await asyncio.to_thread(_check_rate_limit)

        if req_count > settings.RATE_LIMIT_PER_MINUTE:
            logger.warning(f"Rate limit exceeded cho {identifier} ({req_count} reqs/min)")
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=f"Rate limit exceeded. Maximum {settings.RATE_LIMIT_PER_MINUTE} requests per minute.",
            )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Lỗi Redis Rate Limiter: {e}")

    return True
