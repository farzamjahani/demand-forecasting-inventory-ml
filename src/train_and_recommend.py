"""Compare demand models on a chronological holdout and explore inventory policy sensitivity.

The sample data is synthetic. This is a one-day-ahead forecasting exercise, not
an operational inventory planning tool.
"""
from pathlib import Path
import json
import sqlite3
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error

ROOT = Path(__file__).resolve().parents[1]
FEATURES = [
    'product_id', 'unit_price', 'weekday', 'month', 'promotion',
    'lag_1', 'lag_7', 'rolling_7', 'rolling_28'
]
Z_SCORES = {'90%': 1.282, '95%': 1.645, '99%': 2.326}


def make_features(sales: pd.DataFrame, products: pd.DataFrame) -> pd.DataFrame:
    """Lag and rolling inputs use ONLY sales observed before the target date."""
    df = sales.merge(products[['product_id', 'unit_price']], on='product_id', validate='many_to_one')
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['product_id', 'date']).reset_index(drop=True)
    by_item = df.groupby('product_id')['quantity']
    df['weekday'] = df.date.dt.dayofweek
    df['month'] = df.date.dt.month
    df['lag_1'] = by_item.shift(1)
    df['lag_7'] = by_item.shift(7)
    df['rolling_7'] = by_item.transform(lambda s: s.shift(1).rolling(7).mean())
    df['rolling_28'] = by_item.transform(lambda s: s.shift(1).rolling(28).mean())
    return df.dropna(subset=FEATURES).copy()


def wape(y_true, y_pred):
    actual = np.asarray(y_true, dtype=float)
    forecast = np.asarray(y_pred, dtype=float)
    denominator = np.abs(actual).sum()
    return float(100 * np.abs(actual - forecast).sum() / denominator) if denominator else None


def new_models():
    return {
        'Random Forest': RandomForestRegressor(
            n_estimators=120, max_depth=10, min_samples_leaf=3,
            random_state=42, n_jobs=-1
        ),
        'Gradient Boosting': HistGradientBoostingRegressor(
            max_iter=140, max_leaf_nodes=15, learning_rate=0.06,
            l2_regularization=2.0, random_state=42
        ),
    }


def next_day_features(sales: pd.DataFrame, products: pd.DataFrame) -> pd.DataFrame:
    """Assume no *planned* promotions for the next day (illustrative)."""
    history = sales.copy()
    history['date'] = pd.to_datetime(history.date)
    next_date = history.date.max() + pd.Timedelta(days=1)
    rows = []
    for _, product in products.iterrows():
        values = history.loc[history.product_id.eq(product.product_id)].sort_values('date').quantity.to_numpy()
        if len(values) < 28:
            raise ValueError('At least 28 historic daily observations per product are required')
        rows.append({
            'product_id': int(product.product_id),
            'unit_price': float(product.unit_price),
            'weekday': int(next_date.dayofweek),
            'month': int(next_date.month),
            'promotion': 0,
            'lag_1': float(values[-1]),
            'lag_7': float(values[-7]),
            'rolling_7': float(values[-7:].mean()),
            'rolling_28': float(values[-28:].mean()),
        })
    return pd.DataFrame(rows)


def reorder_point(daily_demand, daily_std, lead_time_days, service_z=1.645):
    """Constant-lead-time normal-approximation heuristic.

    Reorder point = ceil(demand * lead time + z * std * sqrt(lead time)).
    This assumes independent daily demand and does not model open purchase orders.
    """
    if daily_demand < 0 or daily_std < 0 or lead_time_days <= 0 or service_z < 0:
        raise ValueError('Expected demand/std/z must be nonnegative and lead time positive')
    safety_stock = int(np.ceil(service_z * daily_std * np.sqrt(lead_time_days)))
    point = int(np.ceil(daily_demand * lead_time_days)) + safety_stock
    return safety_stock, point


def build_replenishment(products, predictions, std_by_sku, service='95%'):
    df = products.merge(predictions[['product_id', 'predicted_daily_demand']],
                        on='product_id', validate='one_to_one')
    df = df.merge(std_by_sku.rename('daily_std'), on='product_id', validate='one_to_one')
    records = [reorder_point(
        r.predicted_daily_demand, r.daily_std, r.lead_time_days, Z_SCORES[service]
    ) for r in df.itertuples()]
    df['safety_stock'] = [v[0] for v in records]
    df['reorder_point'] = [v[1] for v in records]
    df['suggested_order_qty'] = np.maximum(0, df.reorder_point - df.stock_on_hand).astype(int)
    return df


def sensitivity_scenarios(products, predictions, std_by_sku):
    """Compare hypothetical lead times and nominal target service levels."""
    base = products.merge(predictions[['product_id', 'predicted_daily_demand']], on='product_id')
    base = base.merge(std_by_sku.rename('daily_std'), on='product_id')
    rows = []
    for sku in base.itertuples():
        for lead_time in (3, 7, 14):
            for service, z in Z_SCORES.items():
                safety, point = reorder_point(sku.predicted_daily_demand,
                                              sku.daily_std, lead_time, z)
                rows.append({
                    'product_id': sku.product_id, 'product_name': sku.product_name,
                    'lead_time_days': lead_time, 'target_service_level': service,
                    'predicted_daily_demand': round(sku.predicted_daily_demand, 2),
                    'safety_stock': safety, 'reorder_point': point,
                    'suggested_order_qty': max(0, point - int(sku.stock_on_hand)),
                })
    return pd.DataFrame(rows)


def evaluate_and_select(df):
    """Select by validation MAE, report untouched final 60-day test results."""
    last_day = df.date.max()
    test_start = last_day - pd.Timedelta(days=59)
    validation_start = test_start - pd.Timedelta(days=45)
    train = df.loc[df.date < validation_start]
    val = df.loc[(df.date >= validation_start) & (df.date < test_start)]
    test = df.loc[df.date >= test_start]
    if min(len(train), len(val), len(test)) == 0:
        raise ValueError('Not enough dates for train / validation / test split')

    validation_metrics = {}
    for name, model in new_models().items():
        model.fit(train[FEATURES], train.quantity)
        predicted = np.maximum(0, model.predict(val[FEATURES]))
        validation_metrics[name] = float(mean_absolute_error(val.quantity, predicted))
    chosen_name = min(validation_metrics, key=validation_metrics.get)

    fitted = new_models()
    fitset = pd.concat([train, val])
    scores = []
    pred_map = {'7-day rolling mean': test.rolling_7.to_numpy()}
    for name, model in fitted.items():
        model.fit(fitset[FEATURES], fitset.quantity)
        pred_map[name] = np.maximum(0, model.predict(test[FEATURES]))
    for name, p in pred_map.items():
        scores.append({
            'model': name,
            'validation_mae_units': (round(validation_metrics[name], 3)
                                     if name in validation_metrics else None),
            'test_mae_units': round(float(mean_absolute_error(test.quantity, p)), 3),
            'test_wape_percent': round(wape(test.quantity, p), 2),
        })
    summary = {
        'data': 'SYNTHETIC retail data; 8 products x 365 days',
        'forecast_task': 'next-day units; previous observed sales available for lag features',
        'promotion_assumption': 'test date promotion known, next forecast assumes no planned promotion',
        'selection': 'lowest validation MAE of two ML candidates; never select using test scores',
        'selected_model': chosen_name,
        'train_end': str((validation_start - pd.Timedelta(days=1)).date()),
        'validation_start': str(validation_start.date()),
        'validation_end': str((test_start - pd.Timedelta(days=1)).date()),
        'test_start': str(test_start.date()),
        'test_end': str(last_day.date()),
        'train_rows': int(len(train)),
        'validation_rows': int(len(val)),
        'test_rows': int(len(test)),
        'models': scores,
    }
    return summary, fitted[chosen_name], test.copy(), pred_map


def make_plots(test, pred_map, selected, sensitivity, output_dir):
    sku = test.loc[test.product_id.eq(101), ['date','quantity']].copy()
    sku['chosen'] = pred_map[selected][test.product_id.eq(101).to_numpy()]
    sku['baseline'] = pred_map['7-day rolling mean'][test.product_id.eq(101).to_numpy()]
    fig, ax = plt.subplots(figsize=(9.5, 4))
    ax.plot(sku.date, sku.quantity, label='Observed', lw=1.1)
    ax.plot(sku.date, sku.chosen, label=selected, lw=1.4)
    ax.plot(sku.date, sku.baseline, label='7-day baseline', lw=1.2)
    ax.set(xlabel='Date', ylabel='Units', title='Laptop Stand: last 60 days (synthetic data)')
    ax.legend()
    ax.grid(alpha=0.25)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(output_dir/'forecast_vs_actual.png', dpi=165)
    plt.close(fig)

    case = sensitivity[(sensitivity.product_id == 101) &
                       (sensitivity.target_service_level == '95%')].sort_values('lead_time_days')
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    ax.plot(case.lead_time_days, case.reorder_point, marker='o', label='Reorder point')
    ax.plot(case.lead_time_days, case.safety_stock, marker='s', label='Safety stock')
    ax.set(xlabel='Supplier lead time (days)', ylabel='Units',
           title='Lead-time sensitivity: Laptop Stand, 95% nominal service')
    ax.set_xticks([3, 7, 14])
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir/'lead_time_sensitivity.png', dpi=165)
    plt.close(fig)


def run():
    products = pd.read_csv(ROOT/'data/products.csv')
    sales = pd.read_csv(ROOT/'data/daily_sales.csv')
    if sales.duplicated(['date','product_id']).any():
        raise ValueError('Duplicate product-date records found')
    data = make_features(sales, products)
    summary, model, test, pred_map = evaluate_and_select(data)
    out = ROOT/'results'
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summary['models']).to_csv(out/'model_comparison.csv', index=False)
    (out/'metrics.json').write_text(json.dumps(summary, indent=2)+'\n', encoding='utf-8')

    per_sku = []
    selected = summary['selected_model']
    for _, sku in products.iterrows():
        indices = test.product_id.eq(sku.product_id).to_numpy()
        true = test.loc[indices,'quantity']
        per_sku.append({
            'product_id': int(sku.product_id), 'product_name': sku.product_name,
            'test_mae_selected': round(float(mean_absolute_error(true, pred_map[selected][indices])), 3),
            'test_mae_baseline': round(float(mean_absolute_error(true, pred_map['7-day rolling mean'][indices])), 3),
        })
    pd.DataFrame(per_sku).to_csv(out/'per_product_errors.csv', index=False)

    future = next_day_features(sales, products)
    future['predicted_daily_demand'] = np.maximum(0, model.predict(future[FEATURES]))
    recent = sales.copy()
    recent['date'] = pd.to_datetime(recent.date)
    recent = recent.loc[recent.date >= recent.date.max() - pd.Timedelta(days=27)]
    std_by_sku = recent.groupby('product_id').quantity.std().fillna(0)
    replenish = build_replenishment(products, future, std_by_sku)
    scenarios = sensitivity_scenarios(products, future, std_by_sku)
    keep = ['product_id', 'product_name', 'stock_on_hand', 'lead_time_days',
            'predicted_daily_demand', 'daily_std','safety_stock',
            'reorder_point','suggested_order_qty']
    replenish[keep].round(2).to_csv(out/'replenishment.csv', index=False)
    scenarios.to_csv(out/'lead_time_scenarios.csv', index=False)

    with sqlite3.connect(ROOT/'data/retail_demo.sqlite') as db:
        products.to_sql('products', db, if_exists='replace', index=False)
        sales.to_sql('daily_sales', db, if_exists='replace', index=False)
        replenish[keep].to_sql('replenishment', db, if_exists='replace', index=False)
        scenarios.to_sql('lead_time_scenarios', db, if_exists='replace', index=False)
        db.execute('CREATE INDEX IF NOT EXISTS idx_sales_sku_date ON daily_sales(product_id, date)')

    make_plots(test, pred_map, selected, scenarios, out)
    print(json.dumps(summary, indent=2))
    print('\nReplenishment sample:\n', replenish[['product_name','stock_on_hand','suggested_order_qty']].to_string(index=False))
    return summary


if __name__ == '__main__':
    if not (ROOT/'data/daily_sales.csv').exists():
        sys.path.insert(0, str(ROOT/'src'))
        from generate_sample_data import generate
        p, s = generate()
        (ROOT/'data').mkdir(exist_ok=True)
        p.to_csv(ROOT/'data/products.csv', index=False)
        s.to_csv(ROOT/'data/daily_sales.csv', index=False)
    run()
