"""Generate deterministic, clearly SYNTHETIC retail sales data."""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def generate(seed: int = 42):
    rng = np.random.default_rng(seed)
    products = pd.DataFrame([
        (101, 'Laptop Stand', 'Accessories', 29.0, 39, 5),
        (102, 'Wireless Mouse', 'Accessories', 22.0, 17, 7),
        (103, 'USB-C Hub', 'Electronics', 39.0, 46, 6),
        (104, 'Mechanical Keyboard', 'Electronics', 74.0, 12, 9),
        (105, 'Monitor Arm', 'Office', 89.0, 32, 10),
        (106, 'Webcam', 'Electronics', 48.0, 15, 7),
        (107, 'Notebook Pack', 'Office', 11.0, 50, 4),
        (108, 'Desk Lamp', 'Office', 45.0, 24, 8),
    ], columns=['product_id', 'product_name', 'category', 'unit_price', 'stock_on_hand', 'lead_time_days'])
    dates = pd.date_range('2025-01-01', '2025-12-31', freq='D')
    rows = []
    for i, product in products.iterrows():
        base = [7, 12, 5, 4, 3, 6, 14, 5][i]
        for day_idx, date in enumerate(dates):
            weekday = date.dayofweek
            promo = int(rng.random() < .09)
            weekend_factor = 1.2 if weekday >= 5 else 1
            seasonal = 1.0 + .20 * np.sin(2 * np.pi * day_idx / 365 + i * .4)
            trend = 1 + .0003 * day_idx
            rate = base * weekend_factor * seasonal * trend * (1.65 if promo else 1)
            quantity = int(rng.poisson(rate))
            rows.append((date.strftime('%Y-%m-%d'), int(product.product_id), promo, quantity))
    sales = pd.DataFrame(rows, columns=['date', 'product_id', 'promotion', 'quantity'])
    return products, sales


if __name__ == '__main__':
    products, sales = generate()
    (ROOT / 'data').mkdir(exist_ok=True)
    products.to_csv(ROOT / 'data' / 'products.csv', index=False)
    sales.to_csv(ROOT / 'data' / 'daily_sales.csv', index=False)
    print(f'Generated {len(sales):,} synthetic daily sales records for {len(products)} products.')
