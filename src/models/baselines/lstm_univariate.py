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

logger = logging.getLogger(__name__)

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # x: (batch, seq_len, input_size)
        out, _ = self.lstm(x)
        # 取最后一个时间步的输出
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

class LSTMUnivariateBaseline:
    def __init__(self, config: Config):
        self.config = config
        # 从配置读取超参
        self.hidden_size = config.lstm.hidden_size
        self.num_layers = config.lstm.num_layers
        self.epochs = config.lstm.epochs
        self.lr = config.lstm.learning_rate
        self.batch_size = config.lstm.batch_size
        self.window_size = config.lstm.window_size
        
        # GPU 检测
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"LSTM Baseline 使用设备: {self.device}")
        
        self.model = LSTMModel(input_size=1, hidden_size=self.hidden_size, num_layers=self.num_layers).to(self.device)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

    def _create_sequences(self, data, window_size):
        sequences = []
        targets = []
        if len(data) <= window_size:
            return [], []
        
        for i in range(len(data) - window_size):
            seq = data[i:i+window_size]
            label = data[i+window_size]
            sequences.append(seq)
            targets.append(label)
        return np.array(sequences), np.array(targets)

    def fit(self, train_raw_path: str):
        """
        训练 LSTM 模型。
        输入: 原始训练数据路径 (csv/parquet)，包含多人的时间序列。
        """
        logger.info("正在准备 LSTM 训练数据...")
        if train_raw_path.endswith('.parquet'):
            df = pd.read_parquet(train_raw_path)
        else:
            df = pd.read_csv(train_raw_path)

        target_col = self.config.columns.target_col
        pid_col = self.config.columns.person_id_col
        time_col = self.config.columns.time_col
        
        all_sequences = []
        all_targets = []
        
        # 按人分组构造滑窗样本
        grouped = df.groupby(pid_col)
        for pid, group in tqdm(grouped, desc="Building LSTM Data"):
            group = group.sort_values(time_col)
            series = group[target_col].values
            seqs, tgts = self._create_sequences(series, self.window_size)
            if len(seqs) > 0:
                all_sequences.append(seqs)
                all_targets.append(tgts)
                
        if not all_sequences:
            logger.warning("训练数据不足以构建滑窗样本！")
            return

        X = np.concatenate(all_sequences, axis=0) # (N, window_size)
        y = np.concatenate(all_targets, axis=0)   # (N,)
        
        # Reshape X for LSTM: (N, seq_len, input_size=1)
        X = X.reshape(-1, self.window_size, 1)
        y = y.reshape(-1, 1)
        
        dataset = WindowDataset(X, y)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        logger.info(f"开始训练 LSTM (Epochs={self.epochs})...")
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

    def predict_next(self, history_series):
        """
        单步预测。
        history_series: list or np.array, 长度至少为 window_size
        """
        if len(history_series) < self.window_size:
            return history_series[-1] # Fallback
            
        seq = np.array(history_series[-self.window_size:]).reshape(1, self.window_size, 1)
        seq_tensor = torch.FloatTensor(seq).to(self.device)
        
        self.model.eval()
        with torch.no_grad():
            pred = self.model(seq_tensor)
        return pred.item()

    def run(self, test_df: pd.DataFrame, output_dir: str):
        """
        在测试集上评估。
        """
        logger.info("开始 LSTM 单变量基线评估...")
        
        predictions = []
        target_col = self.config.columns.target_col
        time_col = self.config.columns.time_col
        pid_col = self.config.columns.person_id_col
        horizon = self.config.experiment.forecast_horizon
        
        grouped = test_df.groupby(pid_col)
        
        for pid, group in tqdm(grouped, desc="LSTM Evaluation"):
            group = group.sort_values(time_col).reset_index(drop=True)
            if len(group) <= self.window_size:
                continue
                
            # Rolling Origin
            for t in range(self.window_size, len(group) - 1):
                # 历史观测
                history = group.iloc[:t+1][target_col].tolist()
                
                max_step = min(horizon, len(group) - 1 - t)
                if max_step < 1:
                    continue
                
                # 递归预测 K 步
                current_hist = history[:] # copy
                preds = []
                
                for k in range(max_step):
                    pred_val = self.predict_next(current_hist)
                    preds.append(pred_val)
                    current_hist.append(pred_val) # 自回归更新
                    
                start_time = group.iloc[t][time_col]
                
                for k in range(max_step):
                    abs_idx = t + 1 + k
                    y_true = group.iloc[abs_idx][target_col]
                    y_pred = preds[k]
                    time_pred = group.iloc[abs_idx][time_col]
                    
                    predictions.append({
                        'person_id': pid,
                        'start_time': start_time,
                        'horizon': k + 1,
                        'time_pred': time_pred,
                        'y_true': y_true,
                        'y_pred': y_pred,
                        'y_arima': 0.0, # LSTM 无 arima 基线
                        'residual_pred': 0.0
                    })

        pred_df = pd.DataFrame(predictions)
        
        evaluator = Evaluator(self.config, output_dir)
        os.makedirs(output_dir, exist_ok=True)
        metrics = evaluator.evaluate(pred_df, save_results=True)
        
        logger.info(f"LSTM Baseline 完成。Metrics: {metrics['overall']}")
        return metrics
