"""
Selects which prior conversation messages to include as context, once
Adaptive Context Activation has decided a context LEVEL is needed.

Per the project spec: "Do NOT simply take the last N messages... rank
previous messages by semantic similarity, combine similarity with
recency, select top relevant messages, preserve chronological order
before sending to the Transformer."
"""
from typing import List, Tuple

# How many of the most relevant history messages to include per level.
MAX_MESSAGES_BY_LEVEL = {
    "low": 0,
    "medium": 2,
    "high": 5,
}

# Weight given to semantic similarity vs. recency when ranking history
# messages for selection. Must sum to 1.0.
SIMILARITY_WEIGHT = 0.7
RECENCY_WEIGHT = 0.3


def select_context_messages(history: List[str], similarities: List[float], level: str) -> List[str]:
    """
    Rank `history` messages by a blend of semantic similarity to the
    current message and recency, then return the top messages for the
    given context level, restored to chronological order.

    `history` and `similarities` must be aligned (same order, same
    length) - both should be the same trailing slice of conversation
    history that scoring.py considered.
    """
    limit = MAX_MESSAGES_BY_LEVEL.get(level, 0)
    if limit == 0 or not history:
        return []

    n = len(history)
    ranked: List[Tuple[int, float]] = []
    for index, similarity in enumerate(similarities):
        # More recent messages (higher index, since history is oldest-first)
        # get a higher recency score.
        recency = (index + 1) / n
        combined = SIMILARITY_WEIGHT * similarity + RECENCY_WEIGHT * recency
        ranked.append((index, combined))

    ranked.sort(key=lambda pair: pair[1], reverse=True)
    top_indices = sorted(index for index, _ in ranked[:limit])  # restore chronological order

    return [history[i] for i in top_indices]
