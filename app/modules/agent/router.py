"""Agent router - conversational AI endpoint."""

from fastapi import APIRouter

from app.modules.agent.schemas import ChatRequest, ChatResponse
from app.modules.agent.service import AgentService

router = APIRouter()
service = AgentService()


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Send a message to the conversational agent."""
    return await service.chat(message=request.message, session_id=request.session_id)
