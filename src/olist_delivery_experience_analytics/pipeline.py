from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


RAW_FILES = {
    "orders": "olist_orders_dataset.csv",
    "items": "olist_order_items_dataset.csv",
    "payments": "olist_order_payments_dataset.csv",
    "reviews": "olist_order_reviews_dataset.csv",
    "products": "olist_products_dataset.csv",
    "customers": "olist_customers_dataset.csv",
    "sellers": "olist_sellers_dataset.csv",
    "translation": "product_category_name_translation.csv",
}
THRESHOLDS = [0.20, 0.30, 0.40, 0.50, 0.60]


def read_raw_tables(raw_dir: Path) -> dict[str, pd.DataFrame]:
    if not raw_dir.exists():
        raise FileNotFoundError(f"Raw data directory not found: {raw_dir}")

    date_columns = {
        "orders": [
            "order_purchase_timestamp",
            "order_approved_at",
            "order_delivered_carrier_date",
            "order_delivered_customer_date",
            "order_estimated_delivery_date",
        ],
    }

    tables: dict[str, pd.DataFrame] = {}
    for key, filename in RAW_FILES.items():
        path = raw_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"Missing required file: {path}")
        tables[key] = pd.read_csv(
            path,
            parse_dates=date_columns.get(key),
            low_memory=False,
        )
    return tables


def sample_tables(tables: dict[str, pd.DataFrame], sample_orders: int | None) -> dict[str, pd.DataFrame]:
    if sample_orders is None:
        return tables

    orders = tables["orders"].sort_values("order_purchase_timestamp").head(sample_orders).copy()
    order_ids = set(orders["order_id"])

    items = tables["items"][tables["items"]["order_id"].isin(order_ids)].copy()
    payments = tables["payments"][tables["payments"]["order_id"].isin(order_ids)].copy()
    reviews = tables["reviews"][tables["reviews"]["order_id"].isin(order_ids)].copy()
    customer_ids = set(orders["customer_id"])
    product_ids = set(items["product_id"])
    seller_ids = set(items["seller_id"])

    return {
        "orders": orders,
        "items": items,
        "payments": payments,
        "reviews": reviews,
        "products": tables["products"][tables["products"]["product_id"].isin(product_ids)].copy(),
        "customers": tables["customers"][tables["customers"]["customer_id"].isin(customer_ids)].copy(),
        "sellers": tables["sellers"][tables["sellers"]["seller_id"].isin(seller_ids)].copy(),
        "translation": tables["translation"].copy(),
    }


def mode_or_unknown(series: pd.Series) -> str:
    non_null = series.dropna()
    if non_null.empty:
        return "unknown"
    return str(non_null.mode().iloc[0])


def build_order_mart(tables: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    products = tables["products"].merge(
        tables["translation"],
        on="product_category_name",
        how="left",
    )
    products["product_category_name_english"] = (
        products["product_category_name_english"]
        .fillna(products["product_category_name"])
        .fillna("unknown")
    )

    item_enriched = (
        tables["items"]
        .merge(
            products[["product_id", "product_category_name_english"]],
            on="product_id",
            how="left",
        )
        .merge(
            tables["sellers"][["seller_id", "seller_state"]],
            on="seller_id",
            how="left",
        )
    )

    item_agg = (
        item_enriched.groupby("order_id")
        .agg(
            items_count=("order_item_id", "count"),
            products_count=("product_id", "nunique"),
            sellers_count=("seller_id", "nunique"),
            item_revenue=("price", "sum"),
            freight_revenue=("freight_value", "sum"),
            primary_category=("product_category_name_english", mode_or_unknown),
            seller_state=("seller_state", mode_or_unknown),
        )
        .reset_index()
    )

    payment_agg = (
        tables["payments"]
        .groupby("order_id")
        .agg(
            payment_value=("payment_value", "sum"),
            payment_installments=("payment_installments", "max"),
            primary_payment_type=("payment_type", mode_or_unknown),
        )
        .reset_index()
    )

    review_agg = (
        tables["reviews"]
        .groupby("order_id")
        .agg(
            review_score=("review_score", "mean"),
            review_count=("review_id", "count"),
        )
        .reset_index()
    )

    mart = (
        tables["orders"]
        .merge(
            tables["customers"][["customer_id", "customer_unique_id", "customer_city", "customer_state"]],
            on="customer_id",
            how="left",
        )
        .merge(item_agg, on="order_id", how="left")
        .merge(payment_agg, on="order_id", how="left")
        .merge(review_agg, on="order_id", how="left")
    )

    mart["delivery_days"] = (
        mart["order_delivered_customer_date"] - mart["order_purchase_timestamp"]
    ).dt.total_seconds() / 86400
    mart["delay_days"] = (
        mart["order_delivered_customer_date"] - mart["order_estimated_delivery_date"]
    ).dt.total_seconds() / 86400
    mart["promised_days"] = (
        mart["order_estimated_delivery_date"] - mart["order_purchase_timestamp"]
    ).dt.total_seconds() / 86400
    mart["delivered_flag"] = mart["order_status"].eq("delivered")
    mart["late_flag"] = (
        mart["delivered_flag"]
        & mart["order_delivered_customer_date"].notna()
        & (mart["delay_days"] > 0)
    )
    mart["low_review_flag"] = mart["review_score"].le(2)
    mart["same_state"] = np.where(
        mart["customer_state"].notna() & mart["seller_state"].notna(),
        mart["customer_state"] == mart["seller_state"],
        np.nan,
    )
    mart["cross_state"] = np.where(
        mart["customer_state"].notna() & mart["seller_state"].notna(),
        mart["customer_state"] != mart["seller_state"],
        np.nan,
    )
    mart["total_gmv"] = mart["item_revenue"].fillna(0) + mart["freight_revenue"].fillna(0)
    mart["purchase_month"] = mart["order_purchase_timestamp"].dt.to_period("M").astype("string")

    return mart, item_enriched


def build_executive_summary(tables: dict[str, pd.DataFrame], mart: pd.DataFrame) -> pd.DataFrame:
    delivered = mart[mart["delivered_flag"]].copy()
    reviewed = delivered[delivered["review_score"].notna()].copy()

    summary = pd.DataFrame(
        [
            {
                "orders": len(tables["orders"]),
                "delivered_orders": int(delivered["order_id"].nunique()),
                "items": len(tables["items"]),
                "customers": tables["customers"]["customer_unique_id"].nunique(),
                "sellers": tables["sellers"]["seller_id"].nunique(),
                "products": tables["products"]["product_id"].nunique(),
                "customer_states": tables["customers"]["customer_state"].nunique(),
                "seller_states": tables["sellers"]["seller_state"].nunique(),
                "total_gmv": tables["items"]["price"].sum() + tables["items"]["freight_value"].sum(),
                "item_revenue": tables["items"]["price"].sum(),
                "freight_revenue": tables["items"]["freight_value"].sum(),
                "avg_order_value": delivered["payment_value"].mean(),
                "avg_review": reviewed["review_score"].mean(),
                "low_review_rate_pct": reviewed["low_review_flag"].mean() * 100,
                "delay_rate_pct": delivered["late_flag"].mean() * 100,
                "avg_delivery_days": delivered["delivery_days"].mean(),
                "median_delivery_days": delivered["delivery_days"].median(),
            }
        ]
    )
    return summary.round(4)


def build_monthly_orders_revenue(mart: pd.DataFrame) -> pd.DataFrame:
    delivered = mart[mart["delivered_flag"] & mart["purchase_month"].notna()].copy()
    monthly = (
        delivered.groupby("purchase_month")
        .agg(
            orders=("order_id", "nunique"),
            revenue=("total_gmv", "sum"),
            avg_review=("review_score", "mean"),
            delay_rate=("late_flag", "mean"),
            avg_delivery_days=("delivery_days", "mean"),
        )
        .reset_index()
        .sort_values("purchase_month")
    )
    return monthly.round(4)


def build_payment_summary(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    payments = (
        tables["payments"]
        .groupby("payment_type")
        .agg(
            transactions=("order_id", "count"),
            value=("payment_value", "sum"),
            avg_installments=("payment_installments", "mean"),
        )
        .reset_index()
        .sort_values("value", ascending=False)
    )
    return payments.round(4)


def build_route_summary(mart: pd.DataFrame, item_enriched: pd.DataFrame) -> pd.DataFrame:
    item_routes = item_enriched.merge(
        mart[["order_id", "delivery_days", "late_flag", "delivered_flag", "same_state"]],
        on="order_id",
        how="left",
    )
    filtered = item_routes[item_routes["delivered_flag"] & item_routes["same_state"].notna()].copy()

    summary = (
        filtered.groupby("same_state")
        .agg(
            items=("order_item_id", "count"),
            avg_delivery_days=("delivery_days", "mean"),
            delay_rate=("late_flag", "mean"),
            avg_freight=("freight_value", "mean"),
            revenue=("price", "sum"),
        )
        .reset_index()
        .sort_values("same_state")
    )
    summary["delay_rate_pct"] = summary["delay_rate"] * 100
    return summary.round(4)


def build_category_summary(mart: pd.DataFrame, item_enriched: pd.DataFrame) -> pd.DataFrame:
    category_frame = item_enriched.merge(
        mart[["order_id", "review_score", "low_review_flag", "late_flag", "delivered_flag"]],
        on="order_id",
        how="left",
    )
    filtered = category_frame[
        category_frame["delivered_flag"]
        & category_frame["review_score"].notna()
        & category_frame["product_category_name_english"].notna()
    ].copy()

    summary = (
        filtered.groupby("product_category_name_english")
        .agg(
            orders=("order_id", "nunique"),
            items=("order_item_id", "count"),
            revenue=("price", "sum"),
            freight=("freight_value", "sum"),
            avg_review=("review_score", "mean"),
            low_review_rate=("low_review_flag", "mean"),
            delay_rate=("late_flag", "mean"),
        )
        .reset_index()
        .sort_values("revenue", ascending=False)
        .head(20)
    )
    return summary.round(4)


def build_data_quality_summary(tables: dict[str, pd.DataFrame], mart: pd.DataFrame) -> pd.DataFrame:
    summary = pd.DataFrame(
        [
            {
                "orders": len(tables["orders"]),
                "items": len(tables["items"]),
                "payments": len(tables["payments"]),
                "reviews": len(tables["reviews"]),
                "products": len(tables["products"]),
                "customers": len(tables["customers"]),
                "sellers": len(tables["sellers"]),
                "missing_review_share": mart["review_score"].isna().mean(),
                "undelivered_share": 1 - mart["delivered_flag"].mean(),
                "missing_seller_state_share": mart["seller_state"].eq("unknown").mean(),
                "missing_category_share": mart["primary_category"].eq("unknown").mean(),
            }
        ]
    )
    return summary.round(4)


def build_model_frame(mart: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    df = mart[mart["delivered_flag"] & mart["review_score"].notna()].copy()
    df["freight_ratio"] = df["freight_revenue"] / df["item_revenue"].replace(0, np.nan)
    df["same_state"] = df["same_state"].map({True: "same_state", False: "cross_state"}).fillna("unknown")
    df = df.replace([np.inf, -np.inf], np.nan)

    features = [
        "delivery_days",
        "delay_days",
        "promised_days",
        "items_count",
        "sellers_count",
        "item_revenue",
        "freight_revenue",
        "freight_ratio",
        "payment_value",
        "payment_installments",
        "customer_state",
        "seller_state",
        "same_state",
        "primary_category",
    ]
    df = df.dropna(subset=features)
    x = df[features].copy()
    y = df["low_review_flag"].astype(int)
    return x, y


def build_preprocessor() -> ColumnTransformer:
    numeric_features = [
        "delivery_days",
        "delay_days",
        "promised_days",
        "items_count",
        "sellers_count",
        "item_revenue",
        "freight_revenue",
        "freight_ratio",
        "payment_value",
        "payment_installments",
    ]
    categorical_features = [
        "customer_state",
        "seller_state",
        "same_state",
        "primary_category",
    ]
    return ColumnTransformer(
        [
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=20), categorical_features),
        ]
    )


def build_models() -> dict[str, Pipeline]:
    preprocessor = build_preprocessor()
    return {
        "Logistic Regression": Pipeline(
            [
                ("preprocess", preprocessor),
                ("model", LogisticRegression(max_iter=2000, class_weight="balanced")),
            ]
        ),
        "Random Forest": Pipeline(
            [
                ("preprocess", preprocessor),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=250,
                        min_samples_leaf=15,
                        random_state=42,
                        class_weight="balanced",
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
    }


def evaluate_models(x: pd.DataFrame, y: pd.Series) -> tuple[pd.DataFrame, dict[str, Pipeline], dict[str, np.ndarray], pd.DataFrame, pd.Series]:
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    metric_rows: list[dict[str, float | str]] = []
    fitted: dict[str, Pipeline] = {}
    probabilities: dict[str, np.ndarray] = {}

    for name, pipeline in build_models().items():
        pipeline.fit(x_train, y_train)
        predictions = pipeline.predict(x_test)
        proba = pipeline.predict_proba(x_test)[:, 1]
        metric_rows.append(
            {
                "model": name,
                "accuracy": accuracy_score(y_test, predictions),
                "roc_auc": roc_auc_score(y_test, proba),
                "precision": precision_score(y_test, predictions, zero_division=0),
                "recall": recall_score(y_test, predictions, zero_division=0),
                "f1": f1_score(y_test, predictions, zero_division=0),
            }
        )
        fitted[name] = pipeline
        probabilities[name] = proba

    metrics = pd.DataFrame(metric_rows).round(4)
    return metrics, fitted, probabilities, x_test, y_test


def build_feature_importance(
    metrics: pd.DataFrame,
    fitted_models: dict[str, Pipeline],
    x_test: pd.DataFrame,
    y_test: pd.Series,
) -> pd.DataFrame:
    best_model_name = metrics.sort_values("roc_auc", ascending=False).iloc[0]["model"]
    best_model = fitted_models[str(best_model_name)]
    importance = permutation_importance(
        best_model,
        x_test,
        y_test,
        n_repeats=8,
        random_state=42,
        n_jobs=1,
    )
    feature_importance = pd.DataFrame(
        {
            "feature": x_test.columns,
            "importance": importance.importances_mean,
        }
    ).sort_values("importance", ascending=False)
    return feature_importance.round(6)


def build_confusion_summary(
    fitted_models: dict[str, Pipeline],
    x_test: pd.DataFrame,
    y_test: pd.Series,
) -> pd.DataFrame:
    rows = []
    for name, model in fitted_models.items():
        predictions = model.predict(x_test)
        tn, fp, fn, tp = confusion_matrix(y_test, predictions).ravel()
        rows.append(
            {
                "model": name,
                "true_negative": tn,
                "false_positive": fp,
                "false_negative": fn,
                "true_positive": tp,
            }
        )
    return pd.DataFrame(rows)


def build_calibration_summary(probabilities: dict[str, np.ndarray], y_test: pd.Series) -> pd.DataFrame:
    rows = []
    for model_name, proba in probabilities.items():
        observed, predicted = calibration_curve(y_test, proba, n_bins=5, strategy="quantile")
        for index, (pred, obs) in enumerate(zip(predicted, observed), start=1):
            rows.append(
                {
                    "model": model_name,
                    "bin": index,
                    "mean_predicted_probability": pred,
                    "observed_rate": obs,
                }
            )
    return pd.DataFrame(rows).round(6)


def build_threshold_metrics(probabilities: dict[str, np.ndarray], y_test: pd.Series) -> pd.DataFrame:
    rows = []
    for model_name, proba in probabilities.items():
        for threshold in THRESHOLDS:
            predictions = (proba >= threshold).astype(int)
            rows.append(
                {
                    "model": model_name,
                    "threshold": threshold,
                    "precision": precision_score(y_test, predictions, zero_division=0),
                    "recall": recall_score(y_test, predictions, zero_division=0),
                    "f1": f1_score(y_test, predictions, zero_division=0),
                    "positive_predictions": int(predictions.sum()),
                }
            )
    return pd.DataFrame(rows).round(6)


def write_metadata(
    output_dir: Path,
    raw_dir: Path,
    sample_orders: int | None,
    x_test: pd.DataFrame,
    metrics: pd.DataFrame,
) -> None:
    metadata = {
        "raw_dir": str(raw_dir),
        "sample_orders": sample_orders,
        "test_rows": int(len(x_test)),
        "models": metrics["model"].tolist(),
        "dashboard_exports": [
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
        ],
    }
    (output_dir / "model_run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def run_pipeline(raw_dir: Path, output_dir: Path, sample_orders: int | None = None) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    tables = sample_tables(read_raw_tables(raw_dir), sample_orders)
    mart, item_enriched = build_order_mart(tables)

    executive_summary = build_executive_summary(tables, mart)
    monthly_summary = build_monthly_orders_revenue(mart)
    payment_summary = build_payment_summary(tables)
    route_summary = build_route_summary(mart, item_enriched)
    category_summary = build_category_summary(mart, item_enriched)
    data_quality_summary = build_data_quality_summary(tables, mart)

    x, y = build_model_frame(mart)
    metrics, fitted_models, probabilities, x_test, y_test = evaluate_models(x, y)
    feature_importance = build_feature_importance(metrics, fitted_models, x_test, y_test)
    confusion_summary = build_confusion_summary(fitted_models, x_test, y_test)
    calibration_summary = build_calibration_summary(probabilities, y_test)
    threshold_summary = build_threshold_metrics(probabilities, y_test)

    executive_summary.to_csv(output_dir / "executive_summary.csv", index=False)
    monthly_summary.to_csv(output_dir / "monthly_orders_revenue.csv", index=False)
    payment_summary.to_csv(output_dir / "payment_summary.csv", index=False)
    route_summary.to_csv(output_dir / "same_state_vs_cross_state_delivery.csv", index=False)
    category_summary.to_csv(output_dir / "top_categories_summary.csv", index=False)
    metrics.to_csv(output_dir / "model_metrics.csv", index=False)
    feature_importance.to_csv(output_dir / "feature_importance.csv", index=False)
    data_quality_summary.to_csv(output_dir / "data_quality_summary.csv", index=False)
    calibration_summary.to_csv(output_dir / "calibration_by_model.csv", index=False)
    confusion_summary.to_csv(output_dir / "confusion_matrix_summary.csv", index=False)
    threshold_summary.to_csv(output_dir / "threshold_metrics.csv", index=False)
    write_metadata(output_dir, raw_dir, sample_orders, x_test, metrics)
