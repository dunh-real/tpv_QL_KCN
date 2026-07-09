from src.agent.recommend_agent.prompts import ANSWER_PROMPT
from src.agent.recommend_agent.utils import get_llm
from src.agent.recommend_agent.state import RecommendAgentState
from src.core.logger import get_logger

logger = get_logger(__name__)

async def answer_node(state:RecommendAgentState):
    """Goi LLM de tra loi cau hoi"""
    question = state.get("question", "")
    context = state.get("context", "")

    if not context:
        logger.warning("[answer] Không có context để trả lời.")
        return {
            "response": "Xin lỗi, tôi không có thông tin về vấn đề này. Vui lòng thử hỏi lại"
        }
    try:
        llm = get_llm()
        llm_with_config = llm.with_config({"run_name": "answer_llm"})
        chain = ANSWER_PROMPT | llm_with_config
        response = await chain.ainvoke({
            "question": question,
            "context": context
        })
        return {"response": response.content.strip()}
    except Exception as e:
        logger.error(f"[answer] Lỗi khi gọi LLM: {e}", exc_info=True)
        return {
            "response": "Xin lỗi, đã xảy ra lỗi hệ thống. Vui lòng thử lại sau."
        }
        