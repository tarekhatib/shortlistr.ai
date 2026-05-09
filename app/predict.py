"""predict.py — Shared model training and inference utilities for the resume screener."""

import os
import sys
from typing import Any

import joblib
import logging
import numpy as np
import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers, callbacks

logging.getLogger("absl").setLevel(logging.ERROR)
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight

sys.path.insert(0, os.path.dirname(__file__))
from gemini_extractor import extract_features_from_cv

DATA_PATH      = os.path.join(os.path.dirname(__file__), "..", "data", "ai_resume_screening.csv")
MODEL_CACHE_PATH = os.path.join(os.path.dirname(__file__), "models.joblib")
MLP_MODEL_PATH   = os.path.join(os.path.dirname(__file__), "mlp_model.keras")

FEATURE_NAMES = [
    "years_experience",
    "skills_match_score",
    "education_level",
    "project_count",
    "resume_length",
]

EDUCATION_ORDER = {"High School": 0, "Bachelors": 1, "Masters": 2, "PhD": 3}
EDUCATION_LABELS = {value: key for key, value in EDUCATION_ORDER.items()}

FEATURE_LABELS = {
    "years_experience":   "Years of Experience",
    "skills_match_score": "Skills Match Score",
    "education_level":    "Education Level",
    "project_count":      "Number of Projects",
    "resume_length":      "Resume Length",
}


def load_dataset() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    return df.drop(columns=["github_activity"]) if "github_activity" in df.columns else df


def compute_class_weight_dict(y_train: pd.Series) -> dict[int, float]:
    classes = np.array([0, 1])
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
    return {int(k): float(v) for k, v in zip(classes, weights)}


def prepare_training_data(df: pd.DataFrame) -> tuple[np.ndarray, pd.Series, dict[int, float], StandardScaler, pd.Series]:
    X = df.drop(columns=["shortlisted"])
    y = df["shortlisted"].map({"Yes": 1, "No": 0})

    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=42
    )

    X_train_enc = X_train.copy()
    X_train_enc["education_level"] = X_train_enc["education_level"].map(EDUCATION_ORDER)

    class_weight_dict = compute_class_weight_dict(y_train)
    scaler = StandardScaler().fit(X_train_enc[FEATURE_NAMES])
    X_train_s = scaler.transform(X_train_enc[FEATURE_NAMES])
    benchmarks = X_train_enc[y_train == 1][FEATURE_NAMES].median()

    return X_train_s, y_train, class_weight_dict, scaler, benchmarks


def _build_mlp(n_features: int) -> keras.Sequential:
    return keras.Sequential([
        layers.Input(shape=(n_features,)),
        layers.Dense(16, activation="relu"),
        layers.Dense(8, activation="relu"),
        layers.Dense(1, activation="sigmoid"),
    ], name="medium_mlp")


def train_models(X_train_s: np.ndarray, y_train: pd.Series, class_weight_dict: dict[int, float]) -> tuple[Any, Any]:
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_split=20,
        class_weight=class_weight_dict,
        random_state=42,
        n_jobs=-1,
    )
    rf.fit(X_train_s, y_train)

    mlp = _build_mlp(X_train_s.shape[1])
    mlp.compile(
        optimizer=keras.optimizers.legacy.Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    mlp.fit(
        X_train_s, y_train.values,
        epochs=100,
        batch_size=256,
        class_weight=class_weight_dict,
        validation_split=0.1,
        callbacks=[
            callbacks.EarlyStopping(
                monitor="val_loss",
                patience=10,
                restore_best_weights=True,
                verbose=0,
            ),
            callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.5,
                patience=5,
                min_lr=1e-6,
                verbose=0,
            ),
        ],
        verbose=0,
    )

    return rf, mlp


def save_model_cache(rf: Any, mlp: keras.Sequential, scaler: StandardScaler, benchmarks: pd.Series) -> None:
    mlp.save(MLP_MODEL_PATH)
    joblib.dump(
        {
            "rf":       rf,
            "mlp_path": MLP_MODEL_PATH,
            "scaler":   scaler,
            "benchmarks": benchmarks,
        },
        MODEL_CACHE_PATH,
    )


def load_model_cache() -> tuple[Any, Any, StandardScaler, pd.Series] | None:
    if not os.path.exists(MODEL_CACHE_PATH):
        return None
    data = joblib.load(MODEL_CACHE_PATH)
    if "mlp_path" not in data:
        return None
    mlp_path = data["mlp_path"]
    if not os.path.exists(mlp_path):
        return None
    mlp = keras.models.load_model(mlp_path)
    return data["rf"], mlp, data["scaler"], data["benchmarks"]


def load_or_train_models(force_retrain: bool = False) -> tuple[Any, Any, StandardScaler, pd.Series]:
    cached = None if force_retrain else load_model_cache()
    if cached is not None:
        return cached

    df = load_dataset()
    X_train_s, y_train, class_weight_dict, scaler, benchmarks = prepare_training_data(df)
    rf, mlp = train_models(X_train_s, y_train, class_weight_dict)
    save_model_cache(rf, mlp, scaler, benchmarks)
    return rf, mlp, scaler, benchmarks


def encode_candidate(features_raw: dict) -> dict:
    features_enc = features_raw.copy()
    features_enc["education_level"] = EDUCATION_ORDER[features_raw["education_level"]]
    return features_enc


def scale_candidate(features_enc: dict, scaler: StandardScaler) -> np.ndarray:
    return scaler.transform(pd.DataFrame([features_enc])[FEATURE_NAMES])


def generate_feedback(features_enc: dict, rf: Any, benchmarks: pd.Series, job_requirements: dict) -> list[dict]:
    importance_map = dict(zip(FEATURE_NAMES, rf.feature_importances_))
    items: list[dict] = []

    jd_targets: dict = {}
    jd_targets["years_experience"] = job_requirements.get("min_years_experience")
    jd_targets["project_count"]    = job_requirements.get("min_project_count")
    edu = job_requirements.get("min_education_level")
    jd_targets["education_level"]  = EDUCATION_ORDER[edu] if edu else EDUCATION_ORDER["High School"]

    for feat in sorted(FEATURE_NAMES, key=lambda f: importance_map[f], reverse=True):
        val   = features_enc[feat]
        label = FEATURE_LABELS[feat]

        # project_count and resume_length are only flagged when the JD explicitly requires them.
        # Their training medians are skewed by senior candidates and mislead on entry-level roles.
        jd_only_feats = {"project_count", "resume_length"}

        jd_val = jd_targets.get(feat)
        if jd_val is not None:
            if val >= jd_val:
                continue
            bench  = jd_val
            source = "job description"
        elif feat in jd_only_feats:
            continue
        else:
            bench = benchmarks[feat]
            if val >= bench:
                continue
            source = "shortlisted median"

        if feat == "education_level":
            items.append({
                "label":     label,
                "current":   EDUCATION_LABELS[int(val)],
                "benchmark": EDUCATION_LABELS[min(3, int(round(bench)))],
                "source":    source,
            })
        elif feat == "years_experience":
            items.append({"label": label, "current": f"{int(val)} yrs",    "benchmark": f"{int(bench)} yrs",   "source": source})
        elif feat == "skills_match_score":
            items.append({"label": label, "current": f"{val:.1f}%",        "benchmark": f"{bench:.1f}%",       "source": source})
        elif feat == "project_count":
            items.append({"label": label, "current": str(int(val)),        "benchmark": str(int(round(bench))), "source": source})
        elif feat == "resume_length":
            items.append({"label": label, "current": f"{int(val)} words",  "benchmark": f"{int(bench)} words", "source": source})
    return items


def predict(pdf_path: str, jd_path: str) -> None:
    rf, mlp, scaler, benchmarks = load_or_train_models()
    features_raw = extract_features_from_cv(pdf_path, jd_path)
    features_enc = encode_candidate(features_raw)
    row_scaled = scale_candidate(features_enc, scaler)

    rf_pred = int(rf.predict(row_scaled)[0])
    rf_prob = float(rf.predict_proba(row_scaled)[0][rf_pred])

    nn_prob_raw = float(mlp.predict(row_scaled, verbose=0)[0][0])
    nn_pred = 1 if nn_prob_raw >= 0.5 else 0
    nn_prob = nn_prob_raw if nn_pred == 1 else (1 - nn_prob_raw)

    print("=" * 55)
    print("  Extracted Features")
    print("=" * 55)
    for key in FEATURE_NAMES:
        print(f"  {key:<25} {features_raw[key]}")

    print("\n" + "=" * 55)
    print("  ML Model — Random Forest")
    print("=" * 55)
    print(f"  Result     : {'SHORTLISTED ✓' if rf_pred == 1 else 'NOT SHORTLISTED ✗'}")
    print(f"  Confidence : {rf_prob * 100:.1f}%")

    if rf_pred == 0:
        print("\n" + "=" * 55)
        print("  Suggested Improvements")
        print("=" * 55)
        for line in generate_feedback(features_enc, rf, benchmarks, {}):
            print(f"  • {line}")

    print("\n" + "=" * 55)
    print("  DL Model — Medium Neural Network (5→16→8→1)")
    print("=" * 55)
    print(f"  Result     : {'SHORTLISTED ✓' if nn_pred == 1 else 'NOT SHORTLISTED ✗'}")
    print(f"  Confidence : {nn_prob * 100:.1f}%")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python predict.py <cv.pdf> <job_description.txt>")
        sys.exit(1)

    pdf_path = sys.argv[1]
    jd_path = sys.argv[2]

    for path, label in [(pdf_path, "CV PDF"), (jd_path, "Job description")]:
        if not os.path.exists(path):
            print(f"Error: {label} not found at '{path}'")
            sys.exit(1)

    if not os.getenv("GEMINI_API_KEY"):
        print("Error: GEMINI_API_KEY environment variable is not set.")
        sys.exit(1)

    predict(pdf_path, jd_path)
