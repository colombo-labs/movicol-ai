"""Agent request/response schemas."""

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """Chat message from user."""

    message: str = Field(..., min_length=1, max_length=2000)
    session_id: str = Field(default="default")


class ChatResponse(BaseModel):
    """Agent response."""

    response: str
    sources: list[str] = Field(default_factory=list)
    session_id: str
