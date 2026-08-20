"""
/context/select - the Adaptive Context Activation endpoint (Phase 6).

Given the current message and recent conversation history, decides how
much context (if any) is actually needed and returns the specific
selected messages. This is deliberately standalone for now - wiring it
into the live per-message chat pipeline (so the sentiment/emotion/
toxicity models actually receive this selected context) is Phase 7.
"""
import time

from fastapi import APIRouter, HTTPException

from app.context.scoring import score_context_requirement
from app.context.selector import select_context_messages
from app.schemas.context import ContextSelectionRequest, ContextSelectionResponse
from app.utils.logging import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/context", tags=["context"])


@router.post("/select", response_model=ContextSelectionResponse)
def select_context_endpoint(payload: ContextSelectionRequest) -> ContextSelectionResponse:
    try:
        start = time.perf_counter()

        result = score_context_requirement(payload.text, payload.history)
        selected = select_context_messages(result.considered_history, result.similarities, result.level)

        latency_ms = (time.perf_counter() - start) * 1000

        return ContextSelectionResponse(
            contextLevel=result.level,
            score=result.score,
            signals=result.signals,
            selectedContextMessages=selected,
            processing={"latencyMs": round(latency_ms, 2), "method": "aca-v1"},
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("Adaptive context activation failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
