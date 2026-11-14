# Demand forecasting and inventory replenishment

A small Python + SQL portfolio project exploring two questions for a distribution business:

1. Can past sales history help estimate **tomorrow's demand**?
2. How does recommended stock change as a supplier's delivery time increases?

The data are **synthetic**: 8 products and 365 daily observations per product. The purpose is to illustrate the workflow, not to claim real-world savings or performance.

## Evaluation

The data are separated chronologically. Model selection uses a 45-day validation window, followed by a final 60-day test period (Nov 2–Dec 31, 2025).

| Method | Validation MAE (units) | Test MAE (units) | Test WAPE |
| --- | ---: | ---: | ---: |
| Previous 7-day average | — | 3.015 | 32.33% |
| Random Forest | 2.502 | 2.638 | 28.29% |
| Gradient Boosting | **2.475** | **2.561** | **27.46%** |

Gradient Boosting was selected **on validation MAE** and evaluated on the holdout. On this synthetic dataset it reduced test MAE by about 15% compared with the moving-average baseline. Results on real demand data may differ.

## Project structure

- src/generate_sample_data.py — reproducible sample data generator.
- src/train_and_recommend.py — lag features, baseline, model selection, forecast, reorder calculations and plotting.
- sql/analysis.sql — SQLite queries using joins, aggregation, CTE and a window function.
- tests/test_pipeline.py — unit tests of data generation, temporal features, forecasting evaluation and inventory calculations.
- results/ — sample metrics and replenishment reports; images and additional scenarios are generated when running the script.

The script also creates **results/forecast_vs_actual.png** and **results/lead_time_sensitivity.png** on execution.

## Inventory heuristic

~~~text
safety_stock = ceil(z * std_daily_demand * sqrt(lead_time_days))
reorder_point = ceil(predicted_daily_demand * lead_time_days) + safety_stock
suggested_order = max(0, reorder_point - stock_on_hand)
~~~

The z-score represents a *nominal* service-level target, not a verified guarantee. This simple rule assumes independent daily demand and fixed lead time, and ignores on-order stock, capacity, and backorders. It is not a complete optimization algorithm.

## Run

Requires Python 3.10 or later.

~~~bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
python src/generate_sample_data.py
python src/train_and_recommend.py
python -m unittest discover -s tests -v
~~~

No external APIs or keys are required. The generator produces data/daily_sales.csv and the pipeline creates data/retail_demo.sqlite along with result CSVs and plots.

Sample SQL query once the pipeline has run:

~~~sql
SELECT product_name, stock_on_hand, reorder_point, suggested_order_qty
FROM replenishment
WHERE suggested_order_qty > 0
ORDER BY suggested_order_qty DESC;
~~~

## Limitations

- Features rely on *observed previous-day* sales and predict **one day ahead**, not recursively over 14 days.
- Promotion status is known for test dates; the next-day example assumes no planned promotion.
- Historical demand is synthetic and contains no inventory-driven stockouts.
- The inventory formula is a heuristic; stockout cost, storage cost, minimum quantities, lead-time variability and orders already in transit are not represented.
- For actual use, validate on permissioned business data, perform walk-forward evaluation, and simulate stock and service outcomes before using the policy.

**Educational prototype, not a production purchasing system.**
