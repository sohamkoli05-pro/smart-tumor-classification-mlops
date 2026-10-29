from pathlib import Path
import json

import joblib
import mlflow
import mlflow.sklearn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    ConfusionMatrixDisplay
)
from scipy.stats import loguniform

from src.data import split_dataset

ARTIFACT_DIR = Path("artifacts")
ARTIFACT_DIR.mkdir(exist_ok=True)

mlflow.set_experiment("student-smart-tumor-classification")


def make_pipeline():
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=3000, random_state=42))
    ])


def evaluate(model, X_test, y_test):
    predictions = model.predict(X_test)

    return {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(precision_score(y_test, predictions, zero_division=0)),
        "recall": float(recall_score(y_test, predictions, zero_division=0)),
        "f1": float(f1_score(y_test, predictions, zero_division=0))
    }


def log_run(name, model, X_train, y_train, X_test, y_test, params):
    with mlflow.start_run(run_name=name):
        mlflow.log_params(params)

        model.fit(X_train, y_train)
        metrics = evaluate(model, X_test, y_test)
        mlflow.log_metrics(metrics)

        fig, ax = plt.subplots()
        ConfusionMatrixDisplay.from_estimator(
            model, X_test, y_test, ax=ax
        )
        fig.tight_layout()

        image_path = ARTIFACT_DIR / f"{name}_confusion.png"
        fig.savefig(image_path)
        plt.close(fig)

        mlflow.log_artifact(str(image_path))
        mlflow.sklearn.log_model(model, artifact_path="model")

        print(f"\n{name}:")
        print(json.dumps(metrics, indent=2))

        return {
            "name": name,
            **metrics
        }


def main():
    X_train, X_test, y_train, y_test = split_dataset()
    results = []
    models = {}

    baseline = make_pipeline()
    models["baseline_logistic_regression"] = baseline

    results.append(log_run(
        "baseline_logistic_regression",
        baseline, X_train, y_train, X_test, y_test,
        {"model": "LogisticRegression", "max_iter": 3000}
    ))

    grid = GridSearchCV(
        make_pipeline(),
        {
            "model__C": [0.1, 1.0, 10.0],
            "model__solver": ["liblinear", "lbfgs"]
        },
        cv=5,
        scoring="accuracy",
        n_jobs=-1
    )

    grid.fit(X_train, y_train)
    models["grid_search_best"] = grid.best_estimator_

    results.append(log_run(
        "grid_search_best",
        grid.best_estimator_,
        X_train, y_train, X_test, y_test,
        {
            "search": "GridSearchCV",
            "best_C": grid.best_params_["model__C"],
            "best_solver": grid.best_params_["model__solver"],
            "best_cv_accuracy": float(grid.best_score_)
        }
    ))

    random_search = RandomizedSearchCV(
        make_pipeline(),
        {
            "model__C": loguniform(0.01, 100),
            "model__solver": ["liblinear", "lbfgs"]
        },
        n_iter=8,
        cv=5,
        scoring="accuracy",
        random_state=42,
        n_jobs=-1
    )

    random_search.fit(X_train, y_train)
    models["random_search_best"] = random_search.best_estimator_

    results.append(log_run(
        "random_search_best",
        random_search.best_estimator_,
        X_train, y_train, X_test, y_test,
        {
            "search": "RandomizedSearchCV",
            "best_C": random_search.best_params_["model__C"],
            "best_solver": random_search.best_params_["model__solver"],
            "best_cv_accuracy": float(random_search.best_score_)
        }
    ))

    best = max(results, key=lambda item: item["f1"])
    best_model = models[best["name"]]

    joblib.dump(best_model, ARTIFACT_DIR / "best_model.joblib")
    (ARTIFACT_DIR / "results.json").write_text(
        json.dumps(results, indent=2),
        encoding="utf-8"
    )

    print("\nEXPERIMENT COMPARISON")
    for item in sorted(results, key=lambda x: x["f1"], reverse=True):
        print(json.dumps(item, indent=2))

    print(f"\nSelected model: {best['name']}")
    print(f"F1 score: {best['f1']:.4f}")
    print("Saved: artifacts/best_model.joblib")
    print("Saved: artifacts/results.json")


if __name__ == "__main__":
    main()
