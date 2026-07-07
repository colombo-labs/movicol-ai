"""Agent router - conversational AI endpoint."""

from fastapi import APIRouter

from app.modules.agent.schemas import ChatRequest, ChatResponse
from app.modules.agent.service import AgentService

router = APIRouter()
service = AgentService()


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Send a message to the conversational agent."""
    return await service.chat(
        message=request.message,
        session_id=request.session_id,
        context=request.context,
    )


@router.delete("/chat/{session_id}")
async def clear_chat(session_id: str):
    """Clear chat session history."""
    service.clear_session(session_id)
    return {"status": "cleared", "session_id": session_id}
