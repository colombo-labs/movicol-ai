"""Agent service - LLM + Graph RAG conversational agent."""

from app.config.settings import get_settings
from app.modules.agent.schemas import ChatResponse


class AgentService:
    """Conversational agent with Graph RAG capabilities."""

    def __init__(self) -> None:
        self._settings = get_settings()
        self._agent = None

    def initialize(self) -> None:
        """Initialize LangChain agent with tools."""
        # TODO: Initialize LangChain agent with:
        # - Graph query tools (stations, routes, neighbors)
        # - Prediction tools (congestion forecast)
        # - Data tools (statistics, anomalies)
        pass

    async def chat(self, message: str, session_id: str) -> ChatResponse:
        """Process a chat message and return agent response."""
        # TODO: Implement actual LangChain agent invocation
        return ChatResponse(
            response="Agent not yet initialized. Coming soon!",
            sources=[],
            session_id=session_id,
        )
