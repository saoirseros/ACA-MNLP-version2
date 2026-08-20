"""
Adds the nlp-service app package to sys.path so experiment scripts can
reuse the exact same model wrappers used by the live NLP service (e.g.
app.models.sentiment.model.analyze_sentiment, app.context.scoring, ...)
instead of duplicating model-loading code here.
"""
import sys
from pathlib import Path

# experiments/common/pathutil.py -> parents[0]=common, [1]=experiments, [2]=repo root
REPO_ROOT = Path(__file__).resolve().parents[2]
NLP_SERVICE_DIR = REPO_ROOT / "nlp-service"


def ensure_nlp_service_on_path() -> None:
    path_str = str(NLP_SERVICE_DIR)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)
