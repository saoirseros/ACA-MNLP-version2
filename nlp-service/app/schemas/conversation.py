"""
Pydantic request/response schemas for conversation-level endpoints
(summarization, topic extraction) - these operate on a list of messages
rather than a single message, unlike sentiment/emotion/toxicity.
"""
from typing import Optional

from pydantic import BaseModel, Field

from app.schemas.analysis import ProcessingInfo


class ConversationMessagesRequest(BaseModel):
    """Shared request shape: an ordered list of message texts (oldest first)."""
    messages: list[str] = Field(..., min_length=1, description="Ordered conversation messages, oldest first")


class SummaryResult(BaseModel):
    text: str


class SummarizeResponse(BaseModel):
    summary: SummaryResult
    processing: ProcessingInfo


class TopicResult(BaseModel):
    label: Optional[str] = None
    keywords: list[str] = []


class TopicResponse(BaseModel):
    topic: TopicResult
    processing: ProcessingInfo
