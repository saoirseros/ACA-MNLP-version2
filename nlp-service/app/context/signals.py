"""
Individual, interpretable signals used to estimate how much conversational
context a message needs (Adaptive Context Activation - Phase 6).

Each function returns a score in [0, 1]; higher means more likely the
message depends on prior conversation and can't be fully interpreted on
its own. These are intentionally simple/rule-based rather than learned,
per the project spec: "Do not make this mathematically unnecessarily
complicated. The first implementation should use an interpretable
scoring mechanism."
"""
import re

# Words/phrases that typically signal a message refers back to something
# said earlier (pronouns without a clear antecedent in the message itself,
# and common correction/continuation markers) and so can't be fully
# interpreted alone. Not exhaustive - a deliberately simple first pass.
REFERENCE_PATTERNS = [
    r"\bthat\b", r"\bthis\b", r"\bthese\b", r"\bthose\b",
    r"\bit\b", r"\bthey\b", r"\bthem\b", r"\bhe\b", r"\bshe\b", r"\bhim\b", r"\bher\b",
    r"\bagain\b", r"\bas i said\b", r"\bas mentioned\b", r"\bsame\b", r"\balso\b", r"\btoo\b",
    r"\bwhat about\b", r"\bwhat do you mean\b", r"\bwhich one\b",
    r"\bi meant\b", r"\bi mean\b", r"\bactually\b", r"\bthe other\b", r"\bnot that\b", r"\bnot this\b",
]
_REFERENCE_RE = re.compile("|".join(REFERENCE_PATTERNS), re.IGNORECASE)

# Messages at/above this word count get no brevity penalty at all.
MIN_WORDS_FOR_FULL_CONFIDENCE = 6


def reference_signal(text: str) -> float:
    """1.0 if the message contains a pronoun/back-reference/correction marker, else 0.0."""
    return 1.0 if _REFERENCE_RE.search(text) else 0.0


def brevity_signal(text: str) -> float:
    """
    Short messages carry a mild ambiguity boost, since there's less
    standalone content to interpret from the text alone. Linear ramp
    from 1.0 (empty) down to 0.0 at MIN_WORDS_FOR_FULL_CONFIDENCE words.
    """
    word_count = len(text.split())
    if word_count == 0:
        return 0.0
    return max(0.0, (MIN_WORDS_FOR_FULL_CONFIDENCE - word_count) / MIN_WORDS_FOR_FULL_CONFIDENCE)
