import pandas as pd
import numpy as np
import os
from tqdm import tqdm
import logging
from typing import Dict, List
from src.core.config import Config
from src.core.arima import ArimaForecaster

logger = logging.getLogger(__name__)

class FeatureEngineer:
    def __init__(self, config: Config):
        self.config = config
        
    def compute_dynamic_features(self, history_df: pd.DataFrame) -> Dict[str, float]:
        """
        根据提供的历史记录计算动态浓缩特征。
        该函数在训练数据构建（使用真实历史）和推理（使用预测历史）期间都会使用。
        
        输入: history_df (按时间排序，直到 t)
        输出: t 时刻的特征字典
        """
        feats = {}
        target = self.config.columns.target_col
        win = self.config.features.window_size
        
        # 确保我们有足够的数据
        # 注意: history_df 包含当前点 t 作为最后一行
        series = history_df[target]
        
        # 1. 滚动均值 (Rolling Mean)
        feats[f'mean_{win}'] = series.iloc[-win:].mean() if len(series) >= 1 else np.nan
        
        # 2. 滚动方差 (Rolling Variance)
        feats[f'var_{win}'] = series.iloc[-win:].var() if len(series) >= 2 else 0.0
        
        # 3. 变化率 (Rate, 年化变化率)
        # 需要时间列 delta
        if len(history_df) >= 2:
            y_curr = series.iloc[-1]
            y_prev = series.iloc[-2]
            
            t_curr = history_df[self.config.columns.time_col].iloc[-1]
            t_prev = history_df[self.config.columns.time_col].iloc[-2]
            
            # 计算 delta_t (年)。假设是 datetime 对象。
            delta_days = (t_curr - t_prev).days
            if delta_days > 0:
                delta_year = delta_days / 365.25
                rate = (y_curr - y_prev) / delta_year
                feats['rate_last'] = rate
                
                # --- 迁移: 月度进展率 ---
                feats['monthly_rate'] = rate / 12.0
            else:
                feats['rate_last'] = 0.0 # 在有效数据中不应发生
                feats['monthly_rate'] = 0.0
        else:
            feats['rate_last'] = 0.0 # 回退
            feats['monthly_rate'] = 0.0
            
        # 4. 加权移动平均 (WMA)
        # 需要历史月度速率
        # 注意: 这里的 history_df 是截至 t 的。
        # WMA 是对过去速率的加权。
        # 我们需要计算过去 3 次的 monthly_rate
        # t0: (y_t - y_{t-1})
        # t1: (y_{t-1} - y_{t-2})
        # t2: (y_{t-2} - y_{t-3})
        
        # 为了简单和性能，我们在这里动态计算最近几次的速率
        # 这比维护一个完整的速率列要好，因为它适应递归预测中的动态变化
        
        rates = []
        for i in range(3): # 需要最近3个速率
            idx_curr = -1 - i
            idx_prev = -2 - i
            
            if len(history_df) >= abs(idx_prev):
                y_c = series.iloc[idx_curr]
                y_p = series.iloc[idx_prev]
                t_c = history_df[self.config.columns.time_col].iloc[idx_curr]
                t_p = history_df[self.config.columns.time_col].iloc[idx_prev]
                
                d_days = (t_c - t_p).days
                if d_days > 0:
                    r = ((y_c - y_p) / (d_days / 365.25)) / 12.0
                    rates.append(r)
                else:
                    rates.append(0.0)
            else:
                rates.append(0.0) # 历史不足补0
                
        # 权重: 0.5, 0.3, 0.2 (t0, t1, t2)
        w1, w2, w3 = 0.5, 0.3, 0.2
        feats['rate_wma'] = w1 * rates[0] + w2 * rates[1] + w3 * rates[2]
            
        return feats

class DataBuilder:
    def __init__(self, config: Config):
        self.config = config
        self.arima = ArimaForecaster(config)
        self.fe = FeatureEngineer(config)

    def build_and_save(self, input_path: str, output_path: str):
        logger.info("正在加载原始数据...")
        df = pd.read_csv(input_path) # 假设输入是 CSV
        
        # 确保 datetime 格式
        df[self.config.columns.time_col] = pd.to_datetime(df[self.config.columns.time_col])
        
        # 排序
        df = df.sort_values(by=[self.config.columns.person_id_col, self.config.columns.time_col])
        
        pairs = []
        
        grouped = df.groupby(self.config.columns.person_id_col)
        
        logger.info("正在生成转换对...")
        for pid, group in tqdm(grouped):
            if len(group) < 2:
                continue
                
            # 遍历时间点 t -> t+1
            # 我们至少需要历史数据来拟合 ARIMA。
            # 策略: 从索引 1 (第 2 条记录) 开始作为目标 t+1
            # 历史记录为索引 0..t
            
            # 尽管 ARIMA 通常需要 >2 个点，
            # 我们仍然生成对，ARIMA 会在需要时回退。
            
            # 重置索引以策万全
            group = group.reset_index(drop=True)
            
            for i in range(len(group) - 1):
                # t 是索引 i, t+1 是索引 i+1
                t_idx = i
                next_idx = i + 1
                
                # 1. 用于 ARIMA 拟合的历史数据 (直到 t)
                # 为了防止泄漏，我们使用 [0 ... t] 的数据来预测 t+1
                history_df = group.iloc[:t_idx+1]
                
                # 2. t+1 时刻的 ARIMA 预测 (基于 t 时刻的历史)
                y_arima = self.arima.predict_next(history_df, self.config.columns.target_col, steps=1)[0]
                
                # 3. 构造 X_t 的特征
                # 静态特征 (取自当前行 t)
                row_t = group.iloc[t_idx]
                row_next = group.iloc[next_idx]
                
                # 基础特征
                sample = {
                    'person_id': pid,
                    't_idx': t_idx, # 逻辑索引
                    'time_t': row_t[self.config.columns.time_col],
                    'time_next': row_next[self.config.columns.time_col],
                    'y_true_next': row_next[self.config.columns.target_col],
                    'y_arima_next': y_arima,
                    'residual': row_next[self.config.columns.target_col] - y_arima
                }
                
                # 添加静态列
                for col in self.config.columns.static_cols:
                    if col in row_t:
                        sample[col] = row_t[col]
                
                # 添加随时间变化列 (在 t 时刻的观测值)
                for col in self.config.columns.timevarying_cols:
                    if col in row_t:
                        sample[col] = row_t[col]
                        
                # 添加动态浓缩特征 (基于直到 t 的历史计算)
                dynamic_feats = self.fe.compute_dynamic_features(history_df)
                sample.update(dynamic_feats)
                
                # 添加 delta_t
                delta_days = (sample['time_next'] - sample['time_t']).days
                sample['delta_t'] = delta_days / 365.25
                
                pairs.append(sample)
                
        # 创建 DataFrame
        pairs_df = pd.DataFrame(pairs)
        
        # 保存
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        pairs_df.to_parquet(output_path) if output_path.endswith('.parquet') else pairs_df.to_csv(output_path, index=False)
        logger.info(f"已保存 {len(pairs_df)} 个转换对到 {output_path}")

class DataSplitter:
    def __init__(self, config: Config):
        self.config = config

    def split_and_save(self, input_path: str):
        # 加载完整数据集
        if input_path.endswith('.parquet'):
            df = pd.read_parquet(input_path)
        else:
            df = pd.read_csv(input_path)
            
        # 按 Person ID 划分以避免泄漏
        pids = df['person_id'].unique()
        np.random.seed(self.config.experiment.random_seed)
        np.random.shuffle(pids)
        
        n_total = len(pids)
        n_test = int(n_total * self.config.experiment.test_size)
        n_val = int(n_total * self.config.experiment.val_size)
        n_train = n_total - n_test - n_val
        
        train_pids = set(pids[:n_train])
        val_pids = set(pids[n_train:n_train+n_val])
        test_pids = set(pids[n_train+n_val:])
        
        # 过滤
        train_df = df[df['person_id'].isin(train_pids)]
        val_df = df[df['person_id'].isin(val_pids)]
        test_df = df[df['person_id'].isin(test_pids)]
        
        # 保存
        base_dir = os.path.dirname(input_path)
        
        # 根据输入文件扩展名决定输出格式
        ext = '.parquet' if input_path.endswith('.parquet') else '.csv'
        
        if ext == '.parquet':
            train_df.to_parquet(os.path.join(base_dir, 'train.parquet'))
            val_df.to_parquet(os.path.join(base_dir, 'val.parquet'))
            test_df.to_parquet(os.path.join(base_dir, 'test.parquet'))
        else:
            train_df.to_csv(os.path.join(base_dir, 'train.csv'), index=False)
            val_df.to_csv(os.path.join(base_dir, 'val.csv'), index=False)
            test_df.to_csv(os.path.join(base_dir, 'test.csv'), index=False)
        
        logger.info(f"划分完成。Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
