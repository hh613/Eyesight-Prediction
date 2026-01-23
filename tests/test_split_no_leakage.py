import unittest
import pandas as pd
import numpy as np
from src.utils.split import make_person_splits, check_no_leakage

class TestSplitNoLeakage(unittest.TestCase):
    def setUp(self):
        # 创建模拟数据: 100 个学生，每人多条记录
        self.n_students = 100
        pids = [f"student_{i}" for i in range(self.n_students)]
        
        data = []
        for pid in pids:
            for year in range(2020, 2024):
                data.append({'student_id': pid, 'year': year, 'value': np.random.randn()})
                
        self.df = pd.DataFrame(data)
        self.person_col = 'student_id'
        
    def test_split_ratios(self):
        """测试划分比例是否准确"""
        ratios = [0.7, 0.1, 0.2]
        splits = make_person_splits(self.df, self.person_col, ratios, seed=42)
        
        n_train = len(splits['train'])
        n_val = len(splits['val'])
        n_test = len(splits['test'])
        
        # 允许少量误差 (整数截断)
        self.assertTrue(abs(n_train - 70) <= 1)
        self.assertTrue(abs(n_val - 10) <= 1)
        self.assertTrue(abs(n_test - 20) <= 1)
        self.assertEqual(n_train + n_val + n_test, self.n_students)
        
    def test_no_leakage(self):
        """测试严格互斥性"""
        ratios = [0.7, 0.1, 0.2]
        splits = make_person_splits(self.df, self.person_col, ratios, seed=123)
        
        # 如果有泄漏，check_no_leakage 会抛出 ValueError
        try:
            check_no_leakage(splits['train'], splits['val'], splits['test'])
        except ValueError as e:
            self.fail(f"check_no_leakage failed: {e}")
            
    def test_random_seed(self):
        """测试随机种子可复现性"""
        ratios = [0.7, 0.1, 0.2]
        splits1 = make_person_splits(self.df, self.person_col, ratios, seed=999)
        splits2 = make_person_splits(self.df, self.person_col, ratios, seed=999)
        
        self.assertEqual(splits1['train'], splits2['train'])
        self.assertEqual(splits1['val'], splits2['val'])
        self.assertEqual(splits1['test'], splits2['test'])
        
        splits3 = make_person_splits(self.df, self.person_col, ratios, seed=888)
        self.assertNotEqual(splits1['train'], splits3['train'])

if __name__ == '__main__':
    unittest.main()
