from pydantic import BaseModel, Field
from typing import List

class RouteDecision(BaseModel):
    """Schema trả về của Route Node"""
    routes: List[str] = Field(
        description="Chọn các luồng cần chạy:'retrieve_vectordb','run_sql_agent'. Nếu không có thông tin liên quan hoặc câu hỏi không rõ ràng, hãy trả về []")

