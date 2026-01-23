import pandas as pd
import json
import os
import logging
from typing import Dict, Any, List
from src.core.config import Config
from src.eval.metrics import MetricsCalculator

logger = logging.getLogger(__name__)

class Evaluator:
    def __init__(self, config: Config, output_dir: str):
        self.config = config
        self.output_dir = output_dir
        self.calculator = MetricsCalculator(accuracy_thresholds=config.experiment.accuracy_thresholds)

    def evaluate(self, predictions_df: pd.DataFrame, save_results: bool = True) -> Dict[str, Any]:
        """
        评估预测结果，计算指标并保存文件。
        
        参数:
            predictions_df: 必须包含 [person_id, horizon, y_true, y_pred]
            save_results: 是否保存 metrics.json 和 predictions.csv
            
        返回:
            指标字典
        """
        required_cols = ['person_id', 'horizon', 'y_true', 'y_pred']
        missing = [c for c in required_cols if c not in predictions_df.columns]
        if missing:
            raise ValueError(f"预测结果缺少必要列: {missing}")
            
        logger.info(f"开始评估 {len(predictions_df)} 条预测记录...")
        
        # 计算指标
        metrics = self.calculator.compute_step_wise(
            predictions_df, 
            horizon_col='horizon', 
            target_col='y_true', 
            pred_col='y_pred'
        )
        
        if save_results:
            self._save(predictions_df, metrics)
            
        return metrics

    def _save(self, df: pd.DataFrame, metrics: Dict[str, Any]):
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 1. 保存 Metrics
        metrics_path = os.path.join(self.output_dir, 'metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=4)
        logger.info(f"指标已保存至 {metrics_path}")
        
        # 2. 保存 Predictions (Standardized Format)
        # 确保包含标准字段: person_id, start_time, horizon, true_time, y_true, y_pred, y_arima_base, residual_pred
        std_cols = [
            'person_id', 'start_time', 'horizon', 'time_pred', # time_pred 对应 true_time
            'y_true', 'y_pred', 'y_arima', 'residual_pred'
        ]
        
        # 仅保存存在的列
        cols_to_save = [c for c in std_cols if c in df.columns]
        # 如果有其他列也想保留，可以添加
        
        pred_path = os.path.join(self.output_dir, 'predictions.csv')
        df[cols_to_save].to_csv(pred_path, index=False)
        logger.info(f"预测结果已保存至 {pred_path}")
