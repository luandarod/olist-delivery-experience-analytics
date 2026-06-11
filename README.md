# Brazilian E-Commerce Delivery Experience Analytics

**E-commerce Analytics | SQL-style Modeling | Marketplace Operations | Customer Experience | Machine Learning**

This project reframes the Olist public marketplace dataset around one business idea:

**delivery experience becomes the review.**

Instead of treating customer ratings as isolated feedback, the repository connects logistics, freight, geography, seller complexity, payment behavior, and delivery promise accuracy into an order-experience analytics layer.

## Live dashboard

[Open the dashboard](https://luandarodrigues.github.io/olist-delivery-experience-analytics/)

## What this repository now shows

- Reproducible analytical exports generated from `data/raw`
- SQL-style order mart logic implemented in Python and documented in SQL
- Executive summary tables for marketplace performance
- Descriptive summaries for delay, geography, categories, and payments
- ML validation beyond a single score: confusion matrix, calibration, and threshold analysis
- Tests that verify the dashboard data contract

## Business question

How can a marketplace identify operational drivers of poor reviews and act before customer dissatisfaction becomes public?

## Dataset

The project uses the Brazilian Olist e-commerce dataset from 2016 to 2018.

Raw entities:

- Orders
- Order items
- Payments
- Reviews
- Products
- Customers
- Sellers
- Product category translation

The raw data is stored in `data/raw/` with Git LFS. The analytical outputs used by the dashboard are stored in `data/`.

## Analytical framing

The unit of analysis is the **order experience**.

The pipeline joins operational and customer-facing signals into a single mart:

- delivery lead time
- delay versus promise
- item and seller complexity
- freight and GMV
- product category
- customer and seller state
- payment behavior
- review outcome

This makes it possible to analyze logistics friction as a customer experience problem, not only a fulfillment metric.

## Current executive snapshot

| Metric | Value |
|---|---:|
| Orders | 99,441 |
| Delivered orders | 96,478 |
| Items | 112,650 |
| Unique customers | 96,096 |
| Sellers | 3,095 |
| Products | 32,951 |
| Total GMV including freight | R$ 15,843,553.24 |
| Average review score | 4.16 |
| Low review rate | 12.77% |
| Delay rate | 8.11% |
| Average delivery time | 12.56 days |

## Marketplace findings

- Cross-state delivery is slower and more expensive than same-state delivery.
- Delay is a strong dissatisfaction signal and should be monitored as a CX risk, not only an SLA failure.
- Product categories combine different levels of volume and dissatisfaction exposure.
- Reviews behave like an outcome of multiple operational layers, not only product quality.

## Modeling layer

The model predicts whether an order receives a low review score, defined as `review_score <= 2`.

| Model | Accuracy | ROC-AUC | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.763 | 0.749 | 0.290 | 0.592 | 0.389 |
| Random Forest | 0.841 | 0.770 | 0.408 | 0.540 | 0.465 |

This is intentionally positioned as an analytical prioritization layer, not a production decision engine.

## Reproducible output contract

The pipeline now regenerates the dashboard-facing artifacts directly from raw data:

- `executive_summary.csv`
- `monthly_orders_revenue.csv`
- `payment_summary.csv`
- `same_state_vs_cross_state_delivery.csv`
- `top_categories_summary.csv`
- `model_metrics.csv`
- `feature_importance.csv`
- `data_quality_summary.csv`
- `calibration_by_model.csv`
- `confusion_matrix_summary.csv`
- `threshold_metrics.csv`
- `model_run_metadata.json`

## Repository structure

```text
olist-delivery-experience-analytics/
|-- README.md
|-- pyproject.toml
|-- requirements.txt
|-- data/
|   |-- raw/
|   `-- analytical CSV exports
|-- docs/
|   `-- index.html
|-- scripts/
|   |-- _bootstrap.py
|   |-- build_olist_analytics.py
|   |-- download_olist_raw_kagglehub.py
|   `-- olist_experience_model.py
|-- sql/
|-- src/
|   `-- olist_delivery_experience_analytics/
|       |-- __init__.py
|       |-- cli.py
|       `-- pipeline.py
`-- tests/
    `-- test_pipeline_outputs.py
```

## How to run

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Build the analytical outputs

```bash
python scripts/build_olist_analytics.py
```

This reads from `data/raw/` and writes the analytical tables to `data/`.

You can also use the package entry point:

```bash
olist-analytics-pipeline data/raw data
```

For quick smoke tests:

```bash
olist-analytics-pipeline data/raw temp_outputs --sample-orders 2500
```

### 3. Run tests

```bash
pytest
```

## Why the refactor matters

Before this update, the repository had analytical assets and storytelling, but the code path was not fully aligned with the dashboard contract.

The stronger version now has:

- a single package-backed pipeline
- consistent raw-data paths
- reproducible dashboard exports
- compatibility wrappers for existing scripts
- automated validation for the exported artifacts

That moves the project closer to real analytical engineering work instead of a one-time portfolio snapshot.

## Stack

Python, Pandas, scikit-learn, pytest, SQL-style data modeling, Git LFS, HTML, CSS, JavaScript, marketplace analytics, logistics analytics, and customer experience modeling.
