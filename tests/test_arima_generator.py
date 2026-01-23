import unittest
import pandas as pd
import numpy as np
from src.models.arima.arima_forecaster import ArimaForecaster
from src.models.arima.feature_generator import ArimaFeatureGenerator

class TestArimaGenerator(unittest.TestCase):
    def setUp(self):
        # 构造一个简单的线性增长序列
        self.series = pd.Series(np.arange(10).astype(float))
        # 构造 DataFrame
        self.df = pd.DataFrame({
            'student_id': ['s1'] * 10,
            'check_date': pd.date_range('2020-01-01', periods=10, freq='Y'),
            'feat1': np.arange(10).astype(float), # 0, 1, ..., 9
            'feat2': np.arange(10, 20).astype(float) # 10, 11, ..., 19
        })

    def test_forecaster_predict_next(self):
        # 测试单步预测
        forecaster = ArimaForecaster(order=(1, 1, 0))
        forecaster.fit(self.series)
        pred = forecaster.predict_next(steps=1)
        self.assertEqual(len(pred), 1)
        # 简单线性序列，ARIMA(1,1,0) 应该能预测接近 10
        self.assertAlmostEqual(pred[0], 10.0, delta=1.0)
        
    def test_forecaster_multistep(self):
        # 测试多步预测
        forecaster = ArimaForecaster(order=(1, 1, 0))
        forecaster.fit(self.series)
        pred = forecaster.predict_next(steps=3)
        self.assertEqual(len(pred), 3)
        # 期望接近 10, 11, 12
        self.assertTrue(pred[0] < pred[1] < pred[2])

    def test_forecaster_fallback(self):
        # 测试短序列回退
        short_series = pd.Series([1.0, 2.0])
        forecaster = ArimaForecaster(fallback_strategy="linear_trend")
        forecaster.fit(short_series) # 应该触发内部不足3点无法拟合
        pred = forecaster.predict_next(steps=2, history_series=short_series)
        # 线性趋势: 1, 2 -> next: 3, 4
        self.assertEqual(len(pred), 2)
        self.assertAlmostEqual(pred[0], 3.0)
        self.assertAlmostEqual(pred[1], 4.0)
        
    def test_feature_generator(self):
        cols = ['feat1', 'feat2']
        generator = ArimaFeatureGenerator(timevarying_cols=cols)
        
        # 预测未来 2 步
        future_df = generator.predict_next(self.df, 'student_id', steps=2)
        
        self.assertEqual(len(future_df), 2)
        self.assertListEqual(list(future_df.columns), cols)
        
        # feat1: 0..9 -> next 10, 11
        self.assertAlmostEqual(future_df['feat1'].iloc[0], 10.0, delta=1.0)
        # feat2: 10..19 -> next 20, 21
        self.assertAlmostEqual(future_df['feat2'].iloc[0], 20.0, delta=1.0)

if __name__ == '__main__':
    unittest.main()
