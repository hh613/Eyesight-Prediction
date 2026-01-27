import numpy as np
import pandas as pd
from typing import Dict, List, Union
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

class MetricsCalculator:
    def __init__(self, accuracy_thresholds: List[float] = None):
        """
        指标计算器。
        
        参数:
            accuracy_thresholds: 准确率阈值列表 (e.g. [0.50, 0.75])
        """
        self.accuracy_thresholds = accuracy_thresholds or [0.50, 0.75]

    def compute(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        计算一组预测的指标。
        """
        # 移除 NaN
        mask = np.isfinite(y_true) & np.isfinite(y_pred)
        y_t = y_true[mask]
        y_p = y_pred[mask]
        
        if len(y_t) == 0:
            return {
                'mae': np.nan,
                'rmse': np.nan,
                'r2': np.nan,
                **{f'acc_{t:.2f}': np.nan for t in self.accuracy_thresholds}
            }
            
        mae = mean_absolute_error(y_t, y_p)
        rmse = np.sqrt(mean_squared_error(y_t, y_p))
        r2 = r2_score(y_t, y_p)
        
        metrics = {
            'mae': float(mae),
            'rmse': float(rmse),
            'r2': float(r2)
        }
        
        errors = np.abs(y_t - y_p)
        for thresh in self.accuracy_thresholds:
            acc = np.mean(errors <= thresh)
            metrics[f'acc_{thresh:.2f}'] = float(acc)
            
        return metrics

    def compute_step_wise(self, df: pd.DataFrame, 
                         horizon_col: str = 'horizon', 
                         target_col: str = 'y_true', 
                         pred_col: str = 'y_pred') -> Dict[str, Dict[str, float]]:
        """
        计算分步指标。
        
        返回:
            {
                'overall': {...},
                'step_1': {...},
                'step_2': {...},
                ...
            }
        """
        results = {}
        
        # 1. Overall
        results['overall'] = self.compute(df[target_col].values, df[pred_col].values)
        
        # 2. Step-wise
        if horizon_col in df.columns:
            steps = sorted(df[horizon_col].unique())
            for step in steps:
                step_df = df[df[horizon_col] == step]
                step_metrics = self.compute(step_df[target_col].values, step_df[pred_col].values)
                results[f'step_{step}'] = step_metrics
                
        return results
