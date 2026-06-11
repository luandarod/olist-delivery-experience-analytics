from __future__ import annotations

from pathlib import Path

import pandas as pd

from olist_delivery_experience_analytics.pipeline import run_pipeline


REQUIRED_OUTPUTS = {
    "executive_summary.csv",
    "monthly_orders_revenue.csv",
    "payment_summary.csv",
    "same_state_vs_cross_state_delivery.csv",
    "top_categories_summary.csv",
    "model_metrics.csv",
    "feature_importance.csv",
    "data_quality_summary.csv",
    "calibration_by_model.csv",
    "confusion_matrix_summary.csv",
    "threshold_metrics.csv",
    "model_run_metadata.json",
}


def test_run_pipeline_writes_dashboard_contract_outputs(tmp_path: Path) -> None:
    output_dir = tmp_path / "outputs"
    run_pipeline(Path("data/raw"), output_dir, sample_orders=2500)

    produced = {path.name for path in output_dir.iterdir()}
    assert REQUIRED_OUTPUTS.issubset(produced)


def test_model_metrics_contains_expected_models(tmp_path: Path) -> None:
    output_dir = tmp_path / "outputs"
    run_pipeline(Path("data/raw"), output_dir, sample_orders=2500)

    metrics = pd.read_csv(output_dir / "model_metrics.csv")
    assert set(metrics["model"]) == {"Logistic Regression", "Random Forest"}


def test_new_outputs_have_expected_columns(tmp_path: Path) -> None:
    output_dir = tmp_path / "outputs"
    run_pipeline(Path("data/raw"), output_dir, sample_orders=2500)

    summary = pd.read_csv(output_dir / "data_quality_summary.csv")
    calibration = pd.read_csv(output_dir / "calibration_by_model.csv")
    confusion = pd.read_csv(output_dir / "confusion_matrix_summary.csv")
    thresholds = pd.read_csv(output_dir / "threshold_metrics.csv")

    assert {
        "orders",
        "items",
        "payments",
        "reviews",
        "products",
        "customers",
        "sellers",
        "missing_review_share",
        "undelivered_share",
    }.issubset(summary.columns)
    assert {
        "model",
        "bin",
        "mean_predicted_probability",
        "observed_rate",
    }.issubset(calibration.columns)
    assert {
        "model",
        "true_negative",
        "false_positive",
        "false_negative",
        "true_positive",
    }.issubset(confusion.columns)
    assert {
        "model",
        "threshold",
        "precision",
        "recall",
        "f1",
        "positive_predictions",
    }.issubset(thresholds.columns)
