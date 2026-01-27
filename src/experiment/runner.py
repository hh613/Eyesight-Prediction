import pandas as pd
import numpy as np
import os
import json
import logging
from tqdm import tqdm
from sklearn.metrics import mean_absolute_error, mean_squared_error
from src.core.config import Config
from src.core.model import ResidualRegressor
from src.experiment.forecast import RecursiveForecaster
from src.eval.evaluator import Evaluator

logger = logging.getLogger(__name__)

class ExperimentRunner:
    def __init__(self, config: Config, data_dir: str):
        self.config = config
        self.data_dir = data_dir
        self.model = ResidualRegressor(config)

    def run(self):
        # 1. 加载数据
        logger.info("正在加载训练数据...")
        
        # 自动检测文件格式
        train_pairs_path = os.path.join(self.data_dir, 'train_pairs.csv')
        if not os.path.exists(train_pairs_path):
             train_pairs_path = os.path.join(self.data_dir, 'train_pairs.parquet')
             if not os.path.exists(train_pairs_path):
                 raise FileNotFoundError(f"在 {self.data_dir} 未找到 train_pairs.csv 或 train_pairs.parquet")
        
        # 读取 Train Pairs
        if train_pairs_path.endswith('.parquet'):
            train_pairs = pd.read_parquet(train_pairs_path)
        else:
            train_pairs = pd.read_csv(train_pairs_path)
            
        # 2. 训练模型
        logger.info(f"正在训练模型 (Target: {self.config.experiment.learning_target})...")
        
        if self.config.experiment.learning_target == "direct":
            # Direct Forecasting: 预测 y_true_next
            # 我们保留 y_arima_next 作为特征 (Stacking)
            X = train_pairs.drop(columns=['residual', 'y_true_next'])
            y = train_pairs['y_true_next']
        else:
            # Residual Learning (TEM): 预测 residual
            # 移除 y_arima_next (因为它用于计算 residual，且在推理时作为基线叠加)
            # 虽然理论上也可以加进去，但 TEM 原文中通常不加
            X = train_pairs.drop(columns=['residual', 'y_true_next', 'y_arima_next'])
            y = train_pairs['residual']
            
        self.model.fit(X, y)
        
        # 3. 在测试集上评估 (递归)
        # 测试集应该是原始序列 (Raw Test Data)
        logger.info("正在测试集上评估 (递归预测)...")
        
        test_raw_path = os.path.join(self.data_dir, 'test_raw.csv')
        if not os.path.exists(test_raw_path):
             test_raw_path = os.path.join(self.data_dir, 'test_raw.parquet')
             if not os.path.exists(test_raw_path):
                 raise FileNotFoundError(f"在 {self.data_dir} 未找到 test_raw.csv 或 test_raw.parquet")
                 
        if test_raw_path.endswith('.parquet'):
            test_raw = pd.read_parquet(test_raw_path)
        else:
            test_raw = pd.read_csv(test_raw_path)
            # 确保时间列格式
            time_col = self.config.columns.time_col
            test_raw[time_col] = pd.to_datetime(test_raw[time_col])
            
        # ... 后续评估逻辑保持不变 ...
        
        # 我们需要在测试历史记录上进行"滚动"评估。
        # 对于测试集中的每个学生:
        # 给定历史 [0..t]，预测 t+1..t+K
        # 然后移动到 [0..t+1]，预测 t+2..t+K+1 ?
        # 或者仅从某个截断点开始预测？
        # 通常采用: "Leave-K-out" 或 "Rolling origin"。
        
        # 需求: "递归预测 K 步 ... 输出未来 K 步的预测序列"
        # 我们将模拟: 对于每个学生，以 t 从 min_len 到 N-K 迭代。
        
        forecaster = RecursiveForecaster(self.config, self.model)
        
        all_preds = []
        metrics = {'mae': [], 'rmse': [], 'acc_050': [], 'acc_075': []}
        
        grouped = test_raw.groupby(self.config.columns.person_id_col)
        
        for pid, group in tqdm(grouped):
            group = group.sort_values(self.config.columns.time_col).reset_index(drop=True)
            if len(group) < 2:
                continue
                
            # 滚动评估
            # 从拥有例如 3 个点开始，预测接下来的 K 个
            min_history = 3
            horizon = self.config.experiment.forecast_horizon
            
            for t in range(min_history, len(group) - 1):
                # 可用历史: 0 ... t (包含)
                history = group.iloc[:t+1]
                
                # 地面真值 (Ground Truth) t+1 ... t+Horizon
                # 仅当我们有足够的未来数据时？
                # 或者尽可能多预测？
                # 让我们严格评估有真值的情况。
                
                max_step = min(horizon, len(group) - 1 - t)
                if max_step < 1:
                    continue
                    
                # 预测
                preds_df = forecaster.forecast(history, horizon=max_step)
                
                # 与真值比较
                for _, pred_row in preds_df.iterrows():
                    step = int(pred_row['horizon'])
                    abs_idx = t + step
                    
                    y_true = group.iloc[abs_idx][self.config.columns.target_col]
                    y_pred = pred_row['y_pred']
                    
                    err = y_pred - y_true
                    
                    all_preds.append({
                        'person_id': pid,
                        'start_time': group.iloc[t][self.config.columns.time_col],
                        'horizon': step,
                        'time_pred': group.iloc[abs_idx][self.config.columns.time_col], # 保持列名一致
                        'y_true': y_true,
                        'y_pred': y_pred,
                        'y_arima': pred_row['y_arima'],
                        'residual_pred': pred_row['residual_pred']
                    })
                    
        # 4. 统一评估
        results_df = pd.DataFrame(all_preds)
        
        # 使用统一的 Evaluator
        evaluator = Evaluator(self.config, self.config.experiment.output_dir)
        os.makedirs(self.config.experiment.output_dir, exist_ok=True)
        metrics = evaluator.evaluate(results_df, save_results=True)
            
        logger.info("实验完成。")
        logger.info(f"指标: {json.dumps(metrics['overall'], indent=2)}")
