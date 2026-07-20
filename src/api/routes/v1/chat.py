from fastapi import APIRouter
from fastapi.responses import StreamingResponse
import asyncio
import json
import time
import uuid

from src.api.schemas.chat import ChatRequest
from src.agent.chat_agent.graph import get_chat_agent
from src.core.logger import get_logger

logger = get_logger(__name__)

STREAM_TIMEOUT_SECONDS = 120

router = APIRouter()


@router.post("/stream")
async def chat_stream_endpoint(request: ChatRequest):
    """
    API endpoint streaming response từ chat_agent.
    Sử dụng Server-Sent Events (SSE) để gửi từng chunk dữ liệu về client.
    """
    request_id = str(uuid.uuid4())[:8]
    logger.info(f"[{request_id}] Nhận câu hỏi: '{request.question[:100]}'")

    async def event_generator():
        start_time = time.time()
        try:
            chat_agent_app = get_chat_agent()
            initial_state = {"question": request.question}

            async with asyncio.timeout(STREAM_TIMEOUT_SECONDS):
                async for event in chat_agent_app.astream_events(initial_state, version="v2"):
                    if (event["event"] == "on_chat_model_stream"
                            and "format_answer" in event.get("tags", [])):
                        chunk = event["data"]["chunk"]
                        if chunk.content:
                            yield f"data: {json.dumps({'content': chunk.content}, ensure_ascii=False)}\n\n"

            elapsed = round(time.time() - start_time, 2)
            logger.info(f"[{request_id}] Hoàn thành trong {elapsed}s")
            yield "data: [DONE]\n\n"

        except TimeoutError:
            elapsed = round(time.time() - start_time, 2)
            logger.error(f"[{request_id}] Timeout sau {elapsed}s")
            yield f"data: {json.dumps({'error': f'Request timeout sau {STREAM_TIMEOUT_SECONDS} giây'}, ensure_ascii=False)}\n\n"

        except Exception as e:
            elapsed = round(time.time() - start_time, 2)
            logger.error(f"[{request_id}] Lỗi sau {elapsed}s: {e}", exc_info=True)
            yield f"data: {json.dumps({'error': str(e)}, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
            "X-Request-ID": request_id,
        }
    )
