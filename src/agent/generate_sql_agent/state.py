

from typing import TypedDict, Optional, Any, Annotated
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class GenerateSQLState(TypedDict, total=False):

    messages: Annotated[list[BaseMessage], add_messages]
    """Danh sách các tin nhắn trong hội thoại (User, AI, Tool)."""

    question: str
    """Câu hỏi từ phía người dùng."""


    db_schema: Optional[str]
    """DDL schema của database, dùng làm context cho LLM.
    Được fetch 1 lần trong node fetch_schema và tái sử dụng ở các node sau."""

    is_cache_hit: bool
    """True nếu đã tìm thấy SQL tương tự trong semantic cache."""
    sql_query: Optional[str]
    """Câu lệnh SQL được generate hoặc sửa bởi LLM."""

    is_valid: bool
    """True nếu SQL đã qua validate thành công."""

    validation_error_type: Optional[str]
    """Loại lỗi validation: 'EMPTY' | 'FORBIDDEN' | 'SYNTAX_ERROR' | None."""

    error: Optional[str]
    """Thông điệp lỗi từ database hoặc các bước xử lý."""

    retries: int
    """Số lần đã retry (fallback). Đếm từ 0."""
