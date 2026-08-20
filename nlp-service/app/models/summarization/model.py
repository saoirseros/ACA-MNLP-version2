"""
Lazy-loaded wrapper around a pretrained conversation summarization model.

Uses lidiya/bart-base-samsum: a BART-base checkpoint fine-tuned on
SAMSum, a dataset of chat/dialogue conversations paired with short human
summaries. This is a deliberately better fit than a generic news
summarizer (e.g. one trained on CNN/DailyMail) since our input is a
multi-turn chat conversation, not a news article, and BART-base keeps
memory usage modest for an 8GB RAM development machine.
"""
import time
from functools import lru_cache
from typing import Optional

from app.utils.logging import get_logger

logger = get_logger(__name__)

MODEL_CHECKPOINT = "lidiya/bart-base-samsum"
MAX_SUMMARY_TOKENS = 80


@lru_cache
def _get_pipeline():
    # Imported lazily so importing this module never pulls in torch/transformers
    # until a summarization request actually arrives.
    from transformers import pipeline

    logger.info("Loading summarization model (first use): %s", MODEL_CHECKPOINT)
    return pipeline("summarization", model=MODEL_CHECKPOINT)


def summarize_conversation(messages: list[str], context: Optional[dict] = None) -> dict:
    """
    Summarize an ordered list of conversation messages (oldest first).

    Messages are joined into a single dialogue-style transcript, which is
    the input format this checkpoint was fine-tuned on. max_length is a
    ceiling, not a target - the model naturally produces shorter output
    for short conversations, so we don't scale it down further (doing so
    risks truncating the summary mid-sentence).
    """
    summarizer = _get_pipeline()

    transcript = "\n".join(messages)

    start = time.perf_counter()
    result = summarizer(transcript, max_length=MAX_SUMMARY_TOKENS, truncation=True)[0]
    latency_ms = (time.perf_counter() - start) * 1000

    return {
        "summary": {"text": result["summary_text"].strip()},
        "processing": {"latencyMs": round(latency_ms, 2), "model": MODEL_CHECKPOINT},
    }
