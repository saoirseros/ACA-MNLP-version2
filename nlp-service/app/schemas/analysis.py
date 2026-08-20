"""Pydantic request/response schemas for the /analyze/* endpoints."""
from typing import Optional

from pydantic import BaseModel, Field


class TextAnalysisRequest(BaseModel):
    """Shared request shape for the single-task analyze endpoints."""
    text: str = Field(..., min_length=1, description="The message text to analyze")


class SentimentResult(BaseModel):
    label: str
    confidence: float


class EmotionResult(BaseModel):
    label: str
    confidence: float


class ToxicityResult(BaseModel):
    label: str
    confidence: float
    # Most likely specific toxicity category (e.g. "insult", "threat"),
    # only set when the message is flagged as toxic.
    category: Optional[str] = None


class ProcessingInfo(BaseModel):
    latencyMs: float
    model: str


class AnalyzeSentimentResponse(BaseModel):
    sentiment: SentimentResult
    processing: ProcessingInfo


class AnalyzeEmotionResponse(BaseModel):
    emotion: EmotionResult
    processing: ProcessingInfo


class AnalyzeToxicityResponse(BaseModel):
    toxicity: ToxicityResult
    processing: ProcessingInfo


class AnalyzeMessageRequest(BaseModel):
    """Request for the combined /analyze/message endpoint used by the chat pipeline."""
    text: str = Field(..., min_length=1, description="The message text to analyze")
    # Preceding conversation messages, oldest first, current message excluded.
    # Adaptive Context Activation decides how much (if any) of this is
    # actually used - see app/context/.
    history: list[str] = Field(default_factory=list)


class AnalyzeMessageResponse(BaseModel):
    """Runs every currently available NLP module on one message in a single call."""
    sentiment: SentimentResult
    emotion: EmotionResult
    toxicity: ToxicityResult
    processing: dict[str, ProcessingInfo]
    totalLatencyMs: float
    # Adaptive Context Activation outcome for this message.
    contextLevel: str
    contextScore: float
    selectedContextMessages: int
