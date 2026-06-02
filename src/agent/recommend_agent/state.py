from typing import Optional, TypedDict


class RecommendAgentState(TypedDict):
    question: str
    """Câu hỏi của người dùng"""
    context: Optional[str]
    """Thông tin truy xuất từ kho tri thức"""
    response: Optional[str]
    """Câu trả lời của mô hình"""