import json
import time
from pathlib import Path

import joblib
import numpy as np
from datasets import load_dataset

from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from src.preprocess import TextPreprocessor
from src.features import SentimentFeaturePipeline


MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)


# =========================================================
# LOAD DATA
# =========================================================
def load_data(n_samples=None):
    print("Loading IMDB dataset...")

    ds = load_dataset("imdb")

    # Shuffle dataset first
    train_ds = ds["train"].shuffle(seed=42)
    test_ds = ds["test"].shuffle(seed=42)

    X_train = train_ds["text"]
    y_train = train_ds["label"]

    X_test = test_ds["text"]
    y_test = test_ds["label"]

    if n_samples:
        X_train = X_train[:n_samples]
        y_train = y_train[:n_samples]

        X_test = X_test[: n_samples // 5]
        y_test = y_test[: n_samples // 5]

    pos = sum(y_train)

    print(f"Train: {len(X_train)} | Test: {len(X_test)}")
    print(f"Class balance — pos: {pos} | neg: {len(y_train)-pos}")

    return X_train, y_train, X_test, y_test


# =========================================================
# BUILD PIPELINE
# =========================================================
def build_pipeline(model_type="lr"):

    models = {
        "lr": LogisticRegression(
            C=1.0,
            max_iter=3000,
            solver="lbfgs",
            n_jobs=-1,
        ),

        "svm": CalibratedClassifierCV(
            LinearSVC(C=1.0, max_iter=3000),
            cv=3,
        ),
    }

    return Pipeline([
        ("preprocessor", TextPreprocessor()),

        ("features", SentimentFeaturePipeline(
            ngram_range=(1, 2),
            max_features=30000,
        )),

        ("classifier", models[model_type]),
    ])


# =========================================================
# BUILD ENSEMBLE
# =========================================================
def build_ensemble():

    return Pipeline([
        ("preprocessor", TextPreprocessor()),

        ("features", SentimentFeaturePipeline(
            ngram_range=(1, 2),
            max_features=30000,
        )),

        ("classifier", VotingClassifier(
            estimators=[
                (
                    "lr",
                    LogisticRegression(
                        C=1.0,
                        max_iter=3000,
                        solver="lbfgs",
                    ),
                ),

                (
                    "svm",
                    CalibratedClassifierCV(
                        LinearSVC(C=1.0, max_iter=3000),
                        cv=3,
                    ),
                ),
            ],

            voting="soft",
            weights=[2, 1],
            n_jobs=-1,
        )),
    ])


# =========================================================
# EVALUATE MODEL
# =========================================================
def evaluate(pipeline, X_test, y_test, name="Model"):

    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)

    acc = np.mean(np.array(y_pred) == np.array(y_test))

    cm = confusion_matrix(y_test, y_pred)

    print(f"\n{'─'*50}")
    print(f" {name}")
    print(f"{'─'*50}")

    print(f"  ROC-AUC  : {auc:.4f}")
    print(f"  F1 Score : {f1:.4f}")
    print(f"  Accuracy : {acc:.4f}")

    print(f"\n  Confusion Matrix:")
    print(f"         Pred NEG  Pred POS")
    print(f"  Act NEG  {cm[0,0]:6d}    {cm[0,1]:6d}")
    print(f"  Act POS  {cm[1,0]:6d}    {cm[1,1]:6d}")

    print(
        classification_report(
            y_test,
            y_pred,
            target_names=["negative", "positive"],
        )
    )

    return {
        "f1": round(f1, 4),
        "roc_auc": round(auc, 4),
        "accuracy": round(acc, 4),
    }


# =========================================================
# TRAIN ALL
# =========================================================
def train_all(n_samples=None):

    X_train, y_train, X_test, y_test = load_data(
        n_samples=n_samples
    )

    all_results = {}

    # -----------------------------------------------------
    # Logistic Regression
    # -----------------------------------------------------
    print("\n>>> Training LR...")

    t0 = time.time()

    lr_pipeline = build_pipeline("lr")

    lr_pipeline.fit(X_train, y_train)

    print(f"    Done in {time.time()-t0:.1f}s")

    lr_metrics = evaluate(
        lr_pipeline,
        X_test,
        y_test,
        name="LOGISTIC REGRESSION",
    )

    joblib.dump(
        lr_pipeline,
        MODELS_DIR / "sentiment_lr.joblib",
        compress=3,
    )

    all_results["lr"] = lr_metrics

    # -----------------------------------------------------
    # Ensemble
    # -----------------------------------------------------
    print("\n>>> Training ENSEMBLE...")

    t0 = time.time()

    ensemble_pipeline = build_ensemble()

    ensemble_pipeline.fit(X_train, y_train)

    print(f"    Done in {time.time()-t0:.1f}s")

    ensemble_metrics = evaluate(
        ensemble_pipeline,
        X_test,
        y_test,
        name="ENSEMBLE",
    )

    joblib.dump(
        ensemble_pipeline,
        MODELS_DIR / "sentiment_ensemble.joblib",
        compress=3,
    )

    with open(
        MODELS_DIR / "sentiment_ensemble_metrics.json",
        "w",
    ) as f:
        json.dump(ensemble_metrics, f, indent=2)

    all_results["ensemble"] = ensemble_metrics

    # -----------------------------------------------------
    # FINAL RESULTS
    # -----------------------------------------------------
    print(f"\n{'═'*50}")
    print(" FINAL COMPARISON")
    print(f"{'═'*50}")

    print(f"{'Model':<15} {'AUC':>8} {'F1':>8} {'Acc':>8}")
    print(f"{'-'*45}")

    for name, r in all_results.items():
        print(
            f"{name:<15} "
            f"{r['roc_auc']:>8.4f} "
            f"{r['f1']:>8.4f} "
            f"{r['accuracy']:>8.4f}"
        )

    return all_results


# =========================================================
# MAIN
# =========================================================
if __name__ == "__main__":

    # Fast training for development
    train_all(n_samples=5000)

    # Full training:
    # train_all()