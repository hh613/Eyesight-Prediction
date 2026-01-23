import unittest
import pandas as pd
import numpy as np
import os
import shutil
import json
from src.eval.metrics import MetricsCalculator
from src.eval.evaluator import Evaluator
from src.core.config import Config

class TestEval(unittest.TestCase):
    def setUp(self):
        self.test_dir = 'tests/temp_eval_output'
        os.makedirs(self.test_dir, exist_ok=True)
        
    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_metrics_calculator(self):
        calc = MetricsCalculator(accuracy_thresholds=[0.5])
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.1, 2.6, 3.0]) # err: 0.1, 0.6, 0.0
        
        metrics = calc.compute(y_true, y_pred)
        
        self.assertAlmostEqual(metrics['mae'], (0.1+0.6+0.0)/3)
        self.assertAlmostEqual(metrics['acc_0.50'], 2/3) # 0.1<=0.5, 0.0<=0.5

    def test_evaluator(self):
        # Mock Config
        config = Config()
        config.experiment.accuracy_thresholds = [0.5]
        
        evaluator = Evaluator(config, self.test_dir)
        
        df = pd.DataFrame({
            'person_id': ['p1', 'p1', 'p2'],
            'horizon': [1, 2, 1],
            'y_true': [1.0, 2.0, 3.0],
            'y_pred': [1.0, 2.0, 4.0], # p2 h1 err=1.0
            'start_time': ['2020', '2020', '2021'],
            'time_pred': ['2021', '2022', '2022']
        })
        
        metrics = evaluator.evaluate(df, save_results=True)
        
        # Check files
        self.assertTrue(os.path.exists(os.path.join(self.test_dir, 'metrics.json')))
        self.assertTrue(os.path.exists(os.path.join(self.test_dir, 'predictions.csv')))
        
        # Check metrics content
        # step 1: p1(err=0), p2(err=1) -> mae=0.5
        self.assertAlmostEqual(metrics['step_1']['mae'], 0.5)
        # step 2: p1(err=0) -> mae=0
        self.assertAlmostEqual(metrics['step_2']['mae'], 0.0)
        # overall: (0+0+1)/3 = 0.333
        self.assertAlmostEqual(metrics['overall']['mae'], 1/3)

if __name__ == '__main__':
    unittest.main()
