"""MLflow Project entry point used by the CI workflow."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score


SCRIPT_DIR = Path(__file__).resolve().parent
TARGET_COLUMN = "target"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=SCRIPT_DIR / "breast_cancer_preprocessing",
    )
    parser.add_argument("--experiment-name", default="breast-cancer-ci")
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def load_data(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    train = pd.read_csv(data_dir / "train.csv")
    test = pd.read_csv(data_dir / "test.csv")
    if TARGET_COLUMN not in train or TARGET_COLUMN not in test:
        raise ValueError(f"Kolom target '{TARGET_COLUMN}' tidak ditemukan.")
    return (
        train.drop(columns=TARGET_COLUMN),
        test.drop(columns=TARGET_COLUMN),
        train[TARGET_COLUMN].astype(int),
        test[TARGET_COLUMN].astype(int),
    )


def main() -> None:
    args = parse_args()
    X_train, X_test, y_train, y_test = load_data(args.data_dir)

    tracking_dir = SCRIPT_DIR / "mlruns"
    tracking_dir.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(tracking_dir.resolve().as_uri())
    mlflow.set_experiment(args.experiment_name)
    mlflow.sklearn.autolog(
        log_input_examples=True,
        log_model_signatures=True,
        silent=False,
    )

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        min_samples_split=2,
        class_weight="balanced",
        random_state=args.random_state,
        n_jobs=-1,
    )

    with mlflow.start_run(run_name="ci-random-forest") as run:
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        probabilities = model.predict_proba(X_test)
        metrics = {
            "test_accuracy": float(accuracy_score(y_test, predictions)),
            "test_precision_malignant": float(
                precision_score(y_test, predictions, pos_label=0, zero_division=0)
            ),
            "test_recall_malignant": float(
                recall_score(y_test, predictions, pos_label=0, zero_division=0)
            ),
            "test_f1_malignant": float(
                f1_score(y_test, predictions, pos_label=0, zero_division=0)
            ),
            "test_roc_auc": float(roc_auc_score(y_test, probabilities[:, 1])),
        }
        mlflow.log_metrics(metrics)
        run_id = run.info.run_id

    (SCRIPT_DIR / "run_id.txt").write_text(run_id, encoding="utf-8")
    (SCRIPT_DIR / "run_summary.json").write_text(
        json.dumps({"run_id": run_id, **metrics}, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({"run_id": run_id, **metrics}, indent=2))


if __name__ == "__main__":
    main()
