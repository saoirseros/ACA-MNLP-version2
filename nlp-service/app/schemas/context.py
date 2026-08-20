"""Pydantic schemas for the Adaptive Context Activation endpoint."""
from pydantic import BaseModel, Field


class ContextSelectionRequest(BaseModel):
    text: str = Field(..., min_length=1, description="The current message")
    history: list[str] = Field(
        default_factory=list,
        description="Preceding conversation messages, oldest first, current message excluded",
    )


class ContextProcessingInfo(BaseModel):
    latencyMs: float
    method: str


class ContextSelectionResponse(BaseModel):
    contextLevel: str
    score: float
    signals: dict[str, float]
    selectedContextMessages: list[str]
    processing: ContextProcessingInfo
