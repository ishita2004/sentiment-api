from functools import lru_cache
from pathlib import Path
import joblib
import numpy as np

MODELS_DIR = Path("models")
DEFAULT_MODEL = "sentiment_ensemble"

# ── CONCEPT: @lru_cache = singleton model loader ──────────
# Loading a joblib model takes ~300-500ms.
# @lru_cache(maxsize=None) on a no-arg function = cached forever.
# First call loads the model; every subsequent call returns the cached object.
# In a web API serving 1000 req/s, this is non-negotiable.

@lru_cache(maxsize=None)
def load_model(name: str = DEFAULT_MODEL):
    path = MODELS_DIR / f"{name}.joblib"
    if not path.exists():
        raise FileNotFoundError(f"Model not found: {path}. Run `python -m src.train` first.")
    print(f"[INFO] Loading model: {path}")
    return joblib.load(path)


def predict_single(text: str, model_name: str = DEFAULT_MODEL, threshold: float = 0.5):
    pipeline = load_model(model_name)
    proba = pipeline.predict_proba([text])[0]   # [neg_prob, pos_prob]
    pos_prob = float(proba[1])
    neg_prob = float(proba[0])
    label = "positive" if pos_prob >= threshold else "negative"
    confidence = pos_prob if label == "positive" else neg_prob

    level = "high" if confidence >= 0.85 else ("medium" if confidence >= 0.65 else "low")

    return {
        "label": label,
        "confidence": round(confidence, 4),
        "confidence_level": level,
        "probabilities": {"positive": round(pos_prob, 4), "negative": round(neg_prob, 4)},
        "model": model_name,
        "threshold_used": threshold,
    }


def predict_batch(texts: list, model_name: str = DEFAULT_MODEL, threshold: float = 0.5):
    # ── CONCEPT: Always batch, never loop single predictions ──
    # Batch pushes the whole list through vectorized numpy ops.
    # 10-100x faster than calling predict_single() N times.
    if not texts:
        return []
    pipeline = load_model(model_name)
    probas = pipeline.predict_proba(texts)   # shape: (N, 2)
    results = []
    for proba in probas:
        pos_prob = float(proba[1])
        neg_prob = float(proba[0])
        label = "positive" if pos_prob >= threshold else "negative"
        confidence = pos_prob if label == "positive" else neg_prob
        level = "high" if confidence >= 0.85 else ("medium" if confidence >= 0.65 else "low")
        results.append({
            "label": label,
            "confidence": round(confidence, 4),
            "confidence_level": level,
            "probabilities": {"positive": round(pos_prob, 4), "negative": round(neg_prob, 4)},
        })
    return results


# ── Quick test ──────────────────────────────────────────────
if __name__ == "__main__":
    tests = [
        "Absolutely loved this film! One of the best I've seen.",
        "Complete waste of time. Terrible and boring.",
        "It was okay, not great but not terrible either.",
        "I can't believe how bad this was. Never again!",
    ]
    print("── Single prediction ────────────────────")
    print(predict_single(tests[0]))

    print("\n── Batch prediction ────────────────────")
    for text, result in zip(tests, predict_batch(tests)):
        print(f"  [{result['label']:8s} {result['confidence']:.3f} {result['confidence_level']:6s}] {text[:55]}")