"""
/analyze/summarize and /analyze/topics - conversation-level endpoints.

Unlike sentiment/emotion/toxicity (per-message), these operate on an
ordered list of messages. They are not yet wired into the live chat
pipeline - that happens in Phase 8 (Conversation Intelligence), once
enough messages have accumulated to make a summary/topic meaningful.
"""
from fastapi import APIRouter, HTTPException

from app.models.summarization.model import summarize_conversation
from app.models.topic.model import extract_topics
from app.schemas.conversation import ConversationMessagesRequest, SummarizeResponse, TopicResponse
from app.utils.logging import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/analyze", tags=["conversation"])


@router.post("/summarize", response_model=SummarizeResponse)
def summarize_endpoint(payload: ConversationMessagesRequest) -> SummarizeResponse:
    try:
        return SummarizeResponse(**summarize_conversation(payload.messages))
    except Exception as exc:  # noqa: BLE001
        logger.exception("Conversation summarization failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/topics", response_model=TopicResponse)
def topics_endpoint(payload: ConversationMessagesRequest) -> TopicResponse:
    try:
        return TopicResponse(**extract_topics(payload.messages))
    except Exception as exc:  # noqa: BLE001
        logger.exception("Topic extraction failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
