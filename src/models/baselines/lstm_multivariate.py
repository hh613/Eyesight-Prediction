import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import logging
import os
from tqdm import tqdm
from src.core.config import Config
from src.eval.evaluator import Evaluator
from src.models.arima.feature_generator import ArimaFeatureGenerator

logger = logging.getLogger(__name__)

class LSTMMultivariateModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(LSTMMultivariateModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1) # 输出仍然是单变量 SE

    def forward(self, x):
        # x: (batch, seq_len, input_size)
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

class WindowDataset(Dataset):
    def __init__(self, sequences, targets):
        self.sequences = torch.FloatTensor(sequences)
        self.targets = torch.FloatTensor(targets)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx]

class LSTMMultivariateBaseline:
    def __init__(self, config: Config):
        self.config = config
        
        # 确定特征
        # 必须使用 timevarying_cols 中可获得的特征
        # 简单起见，我们使用配置中的 timevarying_cols
        # 注意：不包括 target_col 本身，target_col 是 y，也会作为输入的一部分 (Auto-regressive)
        # 所以 input features = [target_col] + timevarying_cols (excl target)
        
        self.target_col = self.config.columns.target_col
        # 过滤掉 target_col 以免重复，虽然通常 timevarying_cols 不包含 target
        self.feature_cols = [c for c in self.config.columns.timevarying_cols if c != self.target_col]
        self.input_size = 1 + len(self.feature_cols) # y + features
        
        self.hidden_size = config.lstm.hidden_size
        self.num_layers = config.lstm.num_layers
        self.epochs = config.lstm.epochs
        self.lr = config.lstm.learning_rate
        self.batch_size = config.lstm.batch_size
        self.window_size = config.lstm.window_size
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"LSTM Multivariate 使用设备: {self.device}, 输入维度: {self.input_size}")
        
        self.model = LSTMMultivariateModel(self.input_size, self.hidden_size, self.num_layers).to(self.device)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        
        # 初始化 ARIMA 特征生成器 (用于递归预测时生成未来 V)
        self.arima_gen = ArimaFeatureGenerator(timevarying_cols=self.feature_cols)

    def _create_sequences(self, df_group):
        """
        构造多变量序列。
        df_group: 包含 target 和 features 的 DataFrame (已按时间排序)
        """
        # 提取数据矩阵 (N, input_size)
        # 列顺序: [target, feat1, feat2, ...]
        cols = [self.target_col] + self.feature_cols
        data = df_group[cols].values
        
        sequences = []
        targets = []
        
        if len(data) <= self.window_size:
            return [], []
            
        for i in range(len(data) - self.window_size):
            # Input: [t-w, ..., t-1] (长度 w)
            # 这里的定义略有不同：通常是用 [t-w+1 ... t] 预测 t+1
            # 让我们保持一致：输入长度 window_size，预测下一个点
            
            seq = data[i:i+self.window_size] # (window_size, input_size)
            label = data[i+self.window_size, 0] # target is column 0
            
            sequences.append(seq)
            targets.append(label)
            
        return np.array(sequences), np.array(targets)

    def fit(self, train_raw_path: str):
        logger.info("正在准备 LSTM 多变量训练数据...")
        if train_raw_path.endswith('.parquet'):
            df = pd.read_parquet(train_raw_path)
        else:
            df = pd.read_csv(train_raw_path)
            
        # 简单的缺失值填充 (均值) - 仅针对训练
        # 实际应使用更复杂的插值
        for col in self.feature_cols:
            if df[col].isnull().any():
                df[col] = df[col].fillna(df[col].mean())
        
        pid_col = self.config.columns.person_id_col
        time_col = self.config.columns.time_col
        
        all_sequences = []
        all_targets = []
        
        grouped = df.groupby(pid_col)
        for pid, group in tqdm(grouped, desc="Building LSTM Data"):
            group = group.sort_values(time_col)
            seqs, tgts = self._create_sequences(group)
            if len(seqs) > 0:
                all_sequences.append(seqs)
                all_targets.append(tgts)
                
        if not all_sequences:
            logger.warning("没有足够数据进行训练")
            return

        X = np.concatenate(all_sequences, axis=0)
        y = np.concatenate(all_targets, axis=0)
        
        # X shape: (N, window_size, input_size)
        y = y.reshape(-1, 1)
        
        dataset = WindowDataset(X, y)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        logger.info(f"开始训练 (Epochs={self.epochs})...")
        self.model.train()
        for epoch in range(self.epochs):
            total_loss = 0
            for seqs, labels in dataloader:
                seqs, labels = seqs.to(self.device), labels.to(self.device)
                
                self.optimizer.zero_grad()
                outputs = self.model(seqs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
            
            if (epoch + 1) % 5 == 0:
                logger.info(f"Epoch [{epoch+1}/{self.epochs}], Loss: {total_loss/len(dataloader):.4f}")

    def predict_next(self, current_seq):
        """
        current_seq: (window_size, input_size) numpy array
        """
        seq_tensor = torch.FloatTensor(current_seq).unsqueeze(0).to(self.device)
        self.model.eval()
        with torch.no_grad():
            pred = self.model(seq_tensor)
        return pred.item()

    def run(self, test_df: pd.DataFrame, output_dir: str):
        logger.info("开始 LSTM 多变量基线评估...")
        
        predictions = []
        pid_col = self.config.columns.person_id_col
        time_col = self.config.columns.time_col
        horizon = self.config.experiment.forecast_horizon
        
        # 填充测试集缺失值 (避免报错)
        for col in self.feature_cols:
            if test_df[col].isnull().any():
                 test_df[col] = test_df[col].fillna(test_df[col].mean())

        grouped = test_df.groupby(pid_col)
        
        for pid, group in tqdm(grouped, desc="LSTM Multi Eval"):
            group = group.sort_values(time_col).reset_index(drop=True)
            if len(group) <= self.window_size:
                continue
                
            # 使用 ARIMA Generator 预测未来的特征 V
            # 我们需要整个历史来拟合 ARIMA (Rolling)
            # 但为了效率，我们可以在每个 t 时刻，只取截至 t 的历史传给 generator
            
            for t in range(self.window_size, len(group) - 1):
                # 截至时刻 t 的历史 DataFrame
                history_df = group.iloc[:t+1]
                
                # 1. 生成未来 K 步的特征 V (horizon)
                # 这会返回一个 DataFrame，包含预测的 features
                max_step = min(horizon, len(group) - 1 - t)
                if max_step < 1:
                    continue

                future_features_df = self.arima_gen.predict_next(history_df, pid_col, steps=max_step)
                
                # 2. 递归预测 y
                # 初始输入序列: 最后的 window_size 个点
                # Shape: (window_size, input_size)
                # Cols: [target, feat1, feat2, ...]
                cols = [self.target_col] + self.feature_cols
                current_seq = group.iloc[t-self.window_size+1 : t+1][cols].values
                
                preds = []
                
                for k in range(max_step):
                    # 预测 y_{t+k+1}
                    pred_y = self.predict_next(current_seq)
                    preds.append(pred_y)
                    
                    # 准备下一步的输入向量 (y_pred, V_arima)
                    # V_arima 来自 future_features_df 的第 k 行
                    next_feats = future_features_df.iloc[k][self.feature_cols].values
                    next_input_vector = np.concatenate(([pred_y], next_feats))
                    
                    # 更新滑窗: 移除最早的，追加新的
                    current_seq = np.vstack([current_seq[1:], next_input_vector])
                
                # 记录结果
                start_time = group.iloc[t][time_col]
                for k in range(max_step):
                    abs_idx = t + 1 + k
                    y_true = group.iloc[abs_idx][self.target_col]
                    y_pred = preds[k]
                    time_pred = group.iloc[abs_idx][time_col]
                    
                    predictions.append({
                        'person_id': pid,
                        'start_time': start_time,
                        'horizon': k + 1,
                        'time_pred': time_pred,
                        'y_true': y_true,
                        'y_pred': y_pred,
                        'y_arima': 0.0,
                        'residual_pred': 0.0,
                        'future_V_source': 'ARIMA-feature-generator' # 标记来源
                    })
                    
        pred_df = pd.DataFrame(predictions)
        evaluator = Evaluator(self.config, output_dir)
        os.makedirs(output_dir, exist_ok=True)
        metrics = evaluator.evaluate(pred_df, save_results=True)
        
        logger.info(f"LSTM Multivariate 完成。Metrics: {metrics['overall']}")
        return metrics
