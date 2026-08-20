"""
Builds the actual text sent to the sentiment/emotion/toxicity models once
Adaptive Context Activation has chosen which prior messages (if any) are
relevant. This is the "Adaptive Model Routing" step from the project spec:
low context requirement -> the current message alone; medium/high -> the
current message together with its selected context.
"""
from typing import List


def build_effective_text(selected_context_messages: List[str], current_text: str) -> str:
    """
    Combine selected context messages (already in chronological order)
    with the current message into a single string for the classifiers.

    Kept deliberately simple (newline-joined transcript) rather than a
    special templating scheme - these classifiers were fine-tuned on
    single short texts, so how much context concatenation actually helps
    vs. hurts predictive quality is precisely what the Phase 9 experiments
    are meant to measure, not something to over-engineer here.
    """
    if not selected_context_messages:
        return current_text
    return "\n".join([*selected_context_messages, current_text])
