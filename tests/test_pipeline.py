"""Run: python -m unittest discover -s tests -v"""
import sys
import unittest
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from generate_sample_data import generate
from train_and_recommend import (
    make_features, next_day_features, reorder_point,
    sensitivity_scenarios, wape, evaluate_and_select,
)


class TestForecasting(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.products, cls.sales = generate()

    def test_data_size_and_uniqueness(self):
        self.assertEqual(len(self.products), 8)
        self.assertEqual(len(self.sales), 2920)
        self.assertFalse(self.sales.duplicated(['date','product_id']).any())

    def test_features_use_prior_dates_only(self):
        f = make_features(self.sales, self.products)
        history = self.sales[self.sales.product_id.eq(101)].sort_values('date').quantity.to_numpy()
        first = f[f.product_id.eq(101)].iloc[0]
        self.assertEqual(first.lag_1, history[27])
        self.assertEqual(first.lag_7, history[21])
        self.assertAlmostEqual(first.rolling_7, np.mean(history[21:28]))
        self.assertAlmostEqual(first.rolling_28, np.mean(history[:28]))

    def test_next_day_features(self):
        f = next_day_features(self.sales, self.products)
        self.assertEqual(len(f), len(self.products))
        self.assertTrue((f.promotion == 0).all())

    def test_wape(self):
        self.assertAlmostEqual(wape([10, 10], [9, 11]), 10)
        self.assertIsNone(wape([0, 0], [0, 0]))

    def test_reorder_increases_with_lead_time(self):
        short = reorder_point(10, 3, 3)
        long = reorder_point(10, 3, 14)
        self.assertGreater(long[1], short[1])
        self.assertGreater(long[0], short[0])
        with self.assertRaises(ValueError):
            reorder_point(10, 2, 0)

    def test_safety_stock_increases_with_service(self):
        lo = reorder_point(10, 3, 7, 1.282)
        hi = reorder_point(10, 3, 7, 2.326)
        self.assertGreaterEqual(hi[0], lo[0])

    def test_sensitivity_scenarios(self):
        next_day = next_day_features(self.sales, self.products)
        next_day['predicted_daily_demand'] = 10.
        std = pd.Series({int(p): 3.0 for p in self.products.product_id}, name='daily_std').rename_axis('product_id')
        scenarios = sensitivity_scenarios(self.products, next_day, std)
        self.assertEqual(len(scenarios), 8*3*3)
        self.assertTrue((scenarios.suggested_order_qty >= 0).all())

    def test_split_chronological_and_metrics(self):
        prepared = make_features(self.sales, self.products)
        results, selected, test, predictions = evaluate_and_select(prepared)
        self.assertEqual(test.date.nunique(), 60)
        self.assertEqual(len(results['models']), 3)
        self.assertEqual(results['selected_model'], min(
            [x for x in results['models'] if x['validation_mae_units'] is not None],
            key=lambda x: x['validation_mae_units'])['model'])
        self.assertEqual(len(predictions[results['selected_model']]), len(test))


if __name__ == '__main__':
    unittest.main()
