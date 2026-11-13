import io
import time
import urllib.request
import zipfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"
DATA_URL = "https://archive.ics.uci.edu/static/public/222/bank+marketing.zip"
RANDOM_STATE = 42
TEST_SIZE = 0.2


def ensure_dataset():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    target = DATA_DIR / "bank-additional-full.csv"
    if target.exists():
        return target
    with urllib.request.urlopen(DATA_URL) as response:
        payload = response.read()
    with zipfile.ZipFile(io.BytesIO(payload)) as outer:
        for outer_name in outer.namelist():
            if not outer_name.endswith("bank-additional.zip"):
                continue
            with zipfile.ZipFile(io.BytesIO(outer.read(outer_name))) as inner:
                for inner_name in inner.namelist():
                    if inner_name.endswith("bank-additional-full.csv"):
                        target.write_bytes(inner.read(inner_name))
                        return target
    raise FileNotFoundError("bank-additional-full.csv was not found in the archive")


def build_models(scale_pos_weight):
    return {
        "Decision Tree": DecisionTreeClassifier(
            max_depth=8,
            min_samples_leaf=20,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=3,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
        "AdaBoost": AdaBoostClassifier(
            n_estimators=200,
            learning_rate=0.5,
            random_state=RANDOM_STATE,
        ),
        "CatBoost": CatBoostClassifier(
            iterations=400,
            depth=6,
            learning_rate=0.05,
            auto_class_weights="Balanced",
            verbose=0,
            allow_writing_files=False,
            random_seed=RANDOM_STATE,
            thread_count=-1,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=400,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            eval_metric="logloss",
            tree_method="hist",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
    }


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    dataset_path = ensure_dataset()
    data = pd.read_csv(dataset_path, sep=";")
    data = data.drop(columns=["duration"])

    print(f"Dataset: {dataset_path.name}")
    print(f"Rows: {len(data)}")
    print(f"Features: {data.shape[1] - 1}")

    X = data.drop(columns=["y"])
    y = (data["y"] == "yes").astype(int)
    categorical_features = X.select_dtypes(include="object").columns.tolist()
    numeric_features = X.select_dtypes(exclude="object").columns.tolist()

    print(f"Categorical features: {len(categorical_features)}")
    print(f"Numeric features: {len(numeric_features)}")
    print(f"Positive class share: {y.mean():.4f}")
    print(f"Class counts: {y.value_counts().to_dict()}")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                categorical_features,
            ),
            ("numeric", "passthrough", numeric_features),
        ]
    )

    X_train_encoded = preprocessor.fit_transform(X_train)
    X_test_encoded = preprocessor.transform(X_test)
    feature_names = preprocessor.get_feature_names_out()

    print(f"Encoded feature count: {X_train_encoded.shape[1]}")
    print(f"Train size: {len(X_train)}")
    print(f"Test size: {len(X_test)}")

    scale_pos_weight = float((y_train == 0).sum() / (y_train == 1).sum())
    balanced_weights = np.where(
        y_train == 1,
        len(y_train) / (2 * (y_train == 1).sum()),
        len(y_train) / (2 * (y_train == 0).sum()),
    )
    models = build_models(scale_pos_weight)

    rows = []
    predictions = {}
    probabilities = {}

    for name, model in models.items():
        start = time.perf_counter()
        if name == "AdaBoost":
            model.fit(X_train_encoded, y_train, sample_weight=balanced_weights)
        else:
            model.fit(X_train_encoded, y_train)
        fit_time = time.perf_counter() - start

        start = time.perf_counter()
        y_pred = model.predict(X_test_encoded)
        inference_time = time.perf_counter() - start

        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_test_encoded)[:, 1]
        else:
            y_proba = y_pred

        predictions[name] = y_pred
        probabilities[name] = y_proba

        rows.append(
            {
                "model": name,
                "accuracy": accuracy_score(y_test, y_pred),
                "precision": precision_score(y_test, y_pred, zero_division=0),
                "recall": recall_score(y_test, y_pred, zero_division=0),
                "f1": f1_score(y_test, y_pred, zero_division=0),
                "roc_auc": roc_auc_score(y_test, y_proba),
                "fit_seconds": fit_time,
                "inference_seconds": inference_time,
            }
        )

    results = pd.DataFrame(rows).sort_values("f1", ascending=False)
    results.to_csv(RESULTS_DIR / "metrics.csv", index=False)

    print("\nModel comparison (sorted by F1-score):")
    print(results.to_string(index=False, float_format=lambda value: f"{value:.4f}"))

    best_model = results.iloc[0]["model"]
    best_f1 = results.iloc[0]["f1"]
    print(f"\nBest model by F1-score: {best_model} ({best_f1:.4f})")

    figure, axes = plt.subplots(2, 3, figsize=(15, 9))
    for axis, (name, y_pred) in zip(axes.ravel(), predictions.items()):
        matrix = confusion_matrix(y_test, y_pred)
        display = ConfusionMatrixDisplay(
            confusion_matrix=matrix,
            display_labels=["no", "yes"],
        )
        display.plot(ax=axis, colorbar=False, cmap="Blues")
        axis.set_title(name)
    axes.ravel()[-1].axis("off")
    figure.tight_layout()
    figure.savefig(RESULTS_DIR / "confusion_matrices.png", dpi=150)
    plt.close(figure)

    metrics_to_plot = ["precision", "recall", "f1"]
    plot_data = results.set_index("model")[metrics_to_plot]
    axis = plot_data.plot(kind="bar", figsize=(11, 6), rot=15)
    axis.set_title("Model comparison for the positive class")
    axis.set_ylabel("Score")
    axis.set_ylim(0, 1)
    axis.grid(axis="y", alpha=0.3)
    axis.legend(loc="lower right")
    axis.figure.tight_layout()
    axis.figure.savefig(RESULTS_DIR / "model_comparison.png", dpi=150)
    plt.close(axis.figure)

    importance_source = "Random Forest" if "Random Forest" in models else best_model
    importance_model = models[importance_source]
    importances = pd.Series(
        importance_model.feature_importances_,
        index=feature_names,
    ).sort_values(ascending=False)
    importances.head(15).to_csv(RESULTS_DIR / "top_features.csv", header=["importance"])
    print(f"\nTop 15 features by importance ({importance_source}):")
    print(importances.head(15).to_string())

    print(f"\nResults saved to: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
