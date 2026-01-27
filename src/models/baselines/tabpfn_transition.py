import pandas as pd
import numpy as np
import logging
import os
from tqdm import tqdm
from src.core.config import Config
from src.eval.evaluator import Evaluator
from src.models.arima.feature_generator import ArimaFeatureGenerator

logger = logging.getLogger(__name__)

class TabularPredictor:
    """
    TabPFN 的封装类。如果环境未安装 tabpfn，则回退到 sklearn 模型。
    """
    def __init__(self, config: Config, device='cpu'):
        self.config = config
        self.device = device
        self.model = self._init_model()
        
    def _init_model(self):
        try:
            from tabpfn import TabPFNClassifier, TabPFNRegressor
            # TabPFN 主要是 Classifier，Regressor 支持可能有限或处于实验阶段
            # 官方 TabPFN 0.1.x 主要针对分类。
            # 如果是回归任务，通常需要 hack 或等待新版。
            # 这里为了演示，我们假设有 TabPFNRegressor 或者使用回退。
            # 实际上 TabPFN 官方库目前主要是 Classifier。
            # 对于回归，我们先使用回退策略 (如 RandomForest/Ridge) 除非确认安装了支持回归的扩展。
            
            # 检查是否真的安装了 tabpfn
            import tabpfn
            logger.info("TabPFN 库已检测到。尝试初始化 TabPFNRegressor...")
            return TabPFNRegressor(device=self.device, N_ensemble_configurations=self.config.tabpfn.N_ensemble_configurations)
            
        except ImportError:
            logger.warning("未安装 TabPFN，回退到 RandomForestRegressor")
            from sklearn.ensemble import RandomForestRegressor
            return RandomForestRegressor(n_estimators=100, n_jobs=-1)
        except Exception as e:
            logger.warning(f"TabPFN 初始化失败 ({e})，回退到 RandomForestRegressor")
            from sklearn.ensemble import RandomForestRegressor
            return RandomForestRegressor(n_estimators=100, n_jobs=-1)

    def fit(self, X, y):
        # TabPFN 不需要显式 fit (它是 In-Context Learning)，但 sklearn 接口需要。
        # 如果是 TabPFN 官方接口，fit 只是存储数据。
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

class TabPFNTransitionBaseline:
    def __init__(self, config: Config):
        self.config = config
        
        # 确定特征
        # 这里的输入 X_t 包含:
        # 1. 静态特征 S
        # 2. 时变特征 V_t
        # 3. 历史特征 H_t (动态计算)
        # 我们使用与 TEM 相同的数据构建逻辑，即 DataBuilder 生成的 feature cols
        # 但是在递归预测时，我们需要动态更新 H_t
        
        # 为了简化，我们假设模型输入的特征列是固定的
        # target_col 是 y_{t+1}
        # "TabPFN表格（多步时同一 ARIMA-feature-generator + 用模型自身 \hat y 更新历史）"
        # 直接预测 target_col。
        
        self.target_col = self.config.columns.target_col
        self.timevarying_cols = [c for c in self.config.columns.timevarying_cols if c != self.target_col]
        
        # 静态特征
        self.static_cols = self.config.columns.static_cols # 假设，应从 config 读取
        
        # 初始化模型
        self.predictor = TabularPredictor(config, device=config.tabpfn.device)
        
        # ARIMA 生成器
        self.arima_gen = ArimaFeatureGenerator(timevarying_cols=self.timevarying_cols)

    def fit(self, train_pairs_path: str):
        """
        训练 TabPFN (或回退模型)。
        输入: 转换对数据 (X_t, y_{t+1})。
        """
        logger.info("正在准备 TabPFN 训练数据...")
        if train_pairs_path.endswith('.parquet'):
            df = pd.read_parquet(train_pairs_path)
        else:
            df = pd.read_csv(train_pairs_path)
            
        # 排除非特征列
        meta_cols = ['person_id', 't_idx', 'time_t', 'time_next', 'y_true_next', 'y_arima_next', 'residual']
        # 注意: train_pairs 是为残差模型构建的，target 是 residual。
        # 如果我们要预测真实值 y_{t+1}，我们需要 y_true_next 作为 label。
        # 这里的 train_pairs 包含 y_true_next。
        
        self.feature_cols = [c for c in df.columns if c not in meta_cols]
        
        X = df[self.feature_cols]
        # 处理 categorical
        X = pd.get_dummies(X, dummy_na=True) # TabPFN 需要数值输入
        self.final_feature_cols = X.columns.tolist()
        
        # Target: 直接预测 y_true_next
        y = df['y_true_next']
        
        logger.info(f"开始训练 TabPFN Baseline (Features: {len(self.final_feature_cols)})...")
        self.predictor.fit(X, y)

    def run(self, test_raw_path: str, output_dir: str):
        """
        递归评估。
        """
        logger.info("开始 TabPFN Baseline 评估...")
        if test_raw_path.endswith('.parquet'):
            test_df = pd.read_parquet(test_raw_path)
        else:
            test_df = pd.read_csv(test_raw_path)
            time_col = self.config.columns.time_col
            test_df[time_col] = pd.to_datetime(test_df[time_col])
            
        pid_col = self.config.columns.person_id_col
        time_col = self.config.columns.time_col
        horizon = self.config.experiment.forecast_horizon
        
        grouped = test_df.groupby(pid_col)
        predictions = []
        
        from src.core.data import FeatureEngineer
        self.fe = FeatureEngineer(self.config)
        
        for pid, group in tqdm(grouped, desc="TabPFN Eval"):
            group = group.sort_values(time_col).reset_index(drop=True)
            if len(group) < 3:
                continue
                
            # Rolling Origin
            for t in range(3, len(group) - 1):
                # 历史直到 t
                history_df = group.iloc[:t+1].copy()
                
                # 递归预测最大步数
                max_step = min(horizon, len(group) - 1 - t)
                if max_step < 1:
                    continue
                    
                # 初始化 current_history 为真实历史
                current_history = history_df.copy()
                
                preds = []
                
                for k in range(max_step):
                    # --- 1. 准备输入特征 X_{T+k} ---
                    # 取 current_history 的最后一行作为基础
                    input_row = current_history.iloc[-1]
                    
                    feat_dict = {}
                    
                    # 静态特征
                    for c in self.config.columns.static_cols:
                        if c in input_row: feat_dict[c] = input_row[c]
                        
                    # 随时间变化列 (Current V)
                    for c in self.config.columns.timevarying_cols:
                        if c in input_row: feat_dict[c] = input_row[c]
                    
                    # 外推特征 (Next V)
                    # 需要基于 current_history 预测 T+k+1 的特征值
                    for c in self.config.columns.timevarying_cols:
                        if c != self.config.columns.target_col:
                            try:
                                val_arima = self.arima_gen.predict_next(current_history, c, steps=1)[0]
                                feat_dict[f"{c}_arima_next"] = val_arima
                            except:
                                feat_dict[f"{c}_arima_next"] = 0.0

                    # 动态特征 (Rolling, Rate etc.)
                    dynamic_feats = self.fe.compute_dynamic_features(current_history)
                    feat_dict.update(dynamic_feats)
                    
                    # Delta t (简化: 假设每年一次或基于平均间隔)
                    # 也可以尝试用 ARIMA 预测 delta_t，但这里简化为 1.0 (如果单位是年) 或 0.5
                    # 或者取历史平均 delta_t
                    if 'delta_t' in self.config.columns.timevarying_cols:
                        # 如果 delta_t 是特征之一，它已经被处理了
                        pass 
                    else:
                        # 如果需要额外计算 delta_t
                        feat_dict['delta_t'] = 0.5 # Default placeholder
                    
                    # --- 2. 构造模型输入向量 ---
                    # 需要转换为 DataFrame 并对齐列
                    input_df = pd.DataFrame([feat_dict])
                    input_df = pd.get_dummies(input_df) # One-hot
                    
                    # 对齐列 (补全缺失列，忽略多余列)
                    input_vector = pd.DataFrame(0, index=[0], columns=self.final_feature_cols)
                    for c in input_df.columns:
                        if c in self.final_feature_cols:
                            input_vector[c] = input_df[c]
                            
                    # --- 3. 预测目标 y_{T+k+1} ---
                    # TabPFN 直接预测 target
                    pred_y = self.predictor.predict(input_vector)[0]
                    preds.append(pred_y)
                    
                    # --- 4. 更新历史 (为下一步 k+1 做准备) ---
                    new_row = input_row.copy()
                    
                    # 填入预测的目标值
                    new_row[self.target_col] = pred_y
                    
                    # 填入预测的特征值 (Next V 变为 Current V)
                    # 我们刚才计算了 feat_dict[f"{c}_arima_next"]，这就是下一步的特征值
                    for c in self.config.columns.timevarying_cols:
                        if c != self.config.columns.target_col:
                            key = f"{c}_arima_next"
                            if key in feat_dict:
                                new_row[c] = feat_dict[key]
                                
                    # 更新时间 (简单递增，防止索引重复或乱序)
                    # 真实应用中应预测 delta_t
                    last_time = input_row[time_col]
                    new_time = last_time + pd.Timedelta(days=180) # 假设半年
                    new_row[time_col] = new_time
                    
                    # 追加
                    current_history = pd.concat([current_history, pd.DataFrame([new_row])], ignore_index=True)

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
                        'y_arima': 0.0, # TabPFN 不使用 ARIMA 基线叠加
                        'residual_pred': 0.0,
                        'future_V_source': 'ARIMA-feature-generator'
                    })

        pred_df = pd.DataFrame(predictions)
        evaluator = Evaluator(self.config, output_dir)
        os.makedirs(output_dir, exist_ok=True)
        metrics = evaluator.evaluate(pred_df, save_results=True)
        
        logger.info(f"TabPFN Baseline 完成。Metrics: {metrics['overall']}")
        return metrics
