"""
EXPERIMENT A vs B: Full/recent-context strategy vs. Adaptive Context
Activation (ACA), evaluated on DailyDialog's real per-utterance emotion
labels. This directly targets the project's central research question:

  "Can adaptive context selection maintain or improve conversational
   analysis quality while reducing unnecessary contextual processing?"

For each dialogue, for each utterance (after the first, since earlier
ones have no history to use), two different inputs are built for the
SAME emotion classifier:

  - Full-context:     all preceding utterances in the dialogue (capped
                       at FULL_CONTEXT_WINDOW) + the current utterance,
                       concatenated - i.e. "just send recent history".
  - Adaptive-context:  only the messages ACA selects as relevant,
                       combined with the current utterance.

Both conditions use the exact same composer/model code path as the live
chat pipeline (app.context.scoring / app.context.selector /
app.context.composer / app.models.emotion.model), so this measures the
real behavior of the shipped algorithm, not a re-implementation of it.
"""
import time
from typing import Dict, List, Tuple

from common.pathutil import ensure_nlp_service_on_path

ensure_nlp_service_on_path()

from app.context.composer import build_effective_text  # noqa: E402
from app.context.scoring import MAX_HISTORY_CONSIDERED, score_context_requirement  # noqa: E402
from app.context.selector import select_context_messages  # noqa: E402
from app.models.emotion.model import analyze_emotion  # noqa: E402

# Full-context strategy uses the same history window ACA itself
# considers, so the comparison isolates "which messages were chosen",
# not "how far back either strategy is allowed to look".
FULL_CONTEXT_WINDOW = MAX_HISTORY_CONSIDERED


def run_context_experiment(dialogues: List[List[Tuple[str, str]]]) -> Dict[str, dict]:
    """
    dialogues: list of dialogues, each a list of (utterance_text,
    emotion_label) tuples in chronological order.

    Returns {"full": {...}, "adaptive": {...}}, each with aligned
    y_true/y_pred/latencies_ms/context_sizes (context size = total words
    across the selected/used context messages, excluding the current
    utterance itself).
    """
    results = {
        "full": {"y_true": [], "y_pred": [], "latencies_ms": [], "context_sizes": []},
        "adaptive": {"y_true": [], "y_pred": [], "latencies_ms": [], "context_sizes": []},
    }

    for dialogue in dialogues:
        history: List[str] = []
        for utterance, true_emotion in dialogue:
            if history:  # only evaluate turns that actually have prior context available
                # --- Experiment A: full/recent context ---
                full_context = history[-FULL_CONTEXT_WINDOW:]
                full_text = build_effective_text(full_context, utterance)

                start = time.perf_counter()
                full_result = analyze_emotion(full_text)
                full_latency = (time.perf_counter() - start) * 1000

                results["full"]["y_true"].append(true_emotion)
                results["full"]["y_pred"].append(full_result["emotion"]["label"])
                results["full"]["latencies_ms"].append(full_latency)
                results["full"]["context_sizes"].append(sum(len(m.split()) for m in full_context))

                # --- Experiment B: Adaptive Context Activation ---
                start = time.perf_counter()
                context_result = score_context_requirement(utterance, history)
                selected = select_context_messages(
                    context_result.considered_history, context_result.similarities, context_result.level
                )
                adaptive_text = build_effective_text(selected, utterance)
                adaptive_result = analyze_emotion(adaptive_text)
                adaptive_latency = (time.perf_counter() - start) * 1000

                results["adaptive"]["y_true"].append(true_emotion)
                results["adaptive"]["y_pred"].append(adaptive_result["emotion"]["label"])
                results["adaptive"]["latencies_ms"].append(adaptive_latency)
                results["adaptive"]["context_sizes"].append(sum(len(m.split()) for m in selected))

            history.append(utterance)

    return results
