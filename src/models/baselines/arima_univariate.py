import pandas as pd
import numpy as np
import logging
from tqdm import tqdm
from src.core.config import Config
from src.models.arima.arima_forecaster import ArimaForecaster
from src.eval.evaluator import Evaluator

logger = logging.getLogger(__name__)

class ArimaUnivariateBaseline:
    def __init__(self, config: Config):
        self.config = config
        self.forecaster = ArimaForecaster(
            order=self.config.arima.order,
            use_grid_search=self.config.arima.use_grid_search,
            fallback_strategy=self.config.arima.fallback_strategy
        )
        
    def run(self, test_df: pd.DataFrame, output_dir: str):
        """
        运行 ARIMA 单变量基线实验。
        对每个学生进行递归预测。
        """
        logger.info("开始 ARIMA 单变量基线评估...")
        
        predictions = []
        target_col = self.config.columns.target_col
        time_col = self.config.columns.time_col
        pid_col = self.config.columns.person_id_col
        horizon = self.config.experiment.forecast_horizon
        
        # 按学生分组
        grouped = test_df.groupby(pid_col)
        
        for pid, group in tqdm(grouped, desc="ARIMA Baseline"):
            # 排序
            group = group.sort_values(time_col).reset_index(drop=True)
            if len(group) < 3: # 至少需要几个点来拟合
                continue
                
            # 滚动评估 (Rolling Origin)
            # 从 t = min_len 开始预测未来
            min_history = 3
            
            for t in range(min_history, len(group) - 1):
                # 历史: 0...t
                history_df = group.iloc[:t+1]
                history_series = history_df[target_col]
                
                # 拟合
                self.forecaster.fit(history_series)
                
                # 预测未来 K 步
                # 注意：这里我们无法获得未来的真实时间戳（如果是纯单变量），
                # 除非我们假设时间是规则的或者我们有 Oracle 时间。
                # 在这里我们取真实数据的未来 K 个时间点作为 ground truth 的时间。
                
                max_step = min(horizon, len(group) - 1 - t)
                if max_step < 1:
                    continue
                    
                # 预测值
                y_pred_vals = self.forecaster.predict_next(steps=max_step, history_series=history_series)
                
                start_time = group.iloc[t][time_col]
                
                for k in range(max_step):
                    abs_idx = t + 1 + k
                    y_true = group.iloc[abs_idx][target_col]
                    y_pred = y_pred_vals[k]
                    time_pred = group.iloc[abs_idx][time_col]
                    
                    predictions.append({
                        'person_id': pid,
                        'start_time': start_time,
                        'horizon': k + 1,
                        'time_pred': time_pred,
                        'y_true': y_true,
                        'y_pred': y_pred,
                        'y_arima': y_pred, # 对于纯 ARIMA baseline，预测值即 ARIMA 值
                        'residual_pred': 0.0 # 无残差模型
                    })
                    
        # 汇总
        pred_df = pd.DataFrame(predictions)
        
        # 评估
        evaluator = Evaluator(self.config, output_dir)
        metrics = evaluator.evaluate(pred_df, save_results=True)
        
        logger.info(f"ARIMA Baseline 完成。Metrics: {metrics['overall']}")
        return metrics
