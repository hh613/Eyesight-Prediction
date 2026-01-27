import pandas as pd
import numpy as np
import logging
import os
from typing import List, Dict
from src.core.config import Config
from src.core.arima import ArimaForecaster
from src.core.model import ResidualRegressor
from src.core.data import FeatureEngineer

logger = logging.getLogger(__name__)

class RecursiveForecaster:
    def __init__(self, config: Config, residual_model: ResidualRegressor):
        self.config = config
        self.residual_model = residual_model
        self.arima = ArimaForecaster(config)
        self.fe = FeatureEngineer(config)

    def forecast(self, person_history_df: pd.DataFrame, horizon: int) -> pd.DataFrame:
        """
        对单个人执行递归预测。
        
        参数:
            person_history_df: 截至时间 T 的真实历史。
            horizon: 预测步数。
            
        返回:
            包含预测结果的 DataFrame。
        """
        # 我们需要模拟未来的步骤。
        # 当前状态: history_df 包含 [0 ... T]
        
        # 我们将把预测追加到本地历史缓冲区以生成下一个特征
        current_history = person_history_df.copy()
        current_history = current_history.sort_values(self.config.columns.time_col).reset_index(drop=True)
        
        predictions = []
        
        pid = current_history[self.config.columns.person_id_col].iloc[0]
        
        # 最后已知的真实时间
        last_real_time = current_history[self.config.columns.time_col].iloc[-1]
        
        for step in range(1, horizon + 1):
            # 1. ARIMA 预测目标基线 (y_arima)
            # 使用所有可用历史 (真实 + 预测)
            y_arima = self.arima.predict_next(current_history, self.config.columns.target_col, steps=1)[0]
            
            # 2. ARIMA 预测随时间变化的特征
            # 我们需要这些来构造残差模型的特征向量 X_{T+step-1}
            # 注意: X_{T+step-1} 预测 y_{T+step}
            
            # 为了构建 X_{T+step-1}，我们需要 T+step-1 的状态。
            # 在 step=1 时，我们在 T。我们有 T 的真实数据。
            # 在 step=2 时，我们在 T+1。我们有 step 1 预测的 T+1 数据。
            
            # 识别代表模型输入"当前状态"的行
            input_row_idx = len(current_history) - 1
            input_row = current_history.iloc[input_row_idx].copy()
            
            # 基于 current_history 构造动态特征
            dynamic_feats = self.fe.compute_dynamic_features(current_history)
            
            # 准备特征向量
            # 我们需要组装一个与训练列匹配的单行 DataFrame
            feat_dict = {}
            
            # 静态列
            for col in self.config.columns.static_cols:
                feat_dict[col] = input_row[col]
                
            # 随时间变化列 (已在 input_row 中，无论是真实的还是预测的)
            for col in self.config.columns.timevarying_cols:
                feat_dict[col] = input_row[col]
                
                # 同步计算 ARIMA 外推特征 (用于推理时的特征一致性)
                # 基于 current_history 预测下一时刻 (step+1) 的特征? 
                # 不，这里的逻辑是: 我们正在预测 step (即 T+step)。
                # ML 模型需要 X_{T+step-1} 来预测 Residual_{T+step}。
                # X_{T+step-1} 包含了 T+step-1 时刻的状态。
                # 在 data.build 中，我们为 X_t 添加了 {col}_arima_next，这是基于 0...t 预测的 t+1 的特征。
                # 所以在这里，我们需要基于 current_history (它已经包含到了 T+step-1) 预测 T+step 的特征。
                
                if col != self.config.columns.target_col:
                    try:
                        # 注意: predict_feature_next 返回的是下一步的预测
                        val_arima = self.arima.predict_feature_next(current_history, col, steps=1)[0]
                        feat_dict[f"{col}_arima_next"] = val_arima
                    except:
                        feat_dict[f"{col}_arima_next"] = 0.0

            # 动态特征
            feat_dict.update(dynamic_feats)
            
            # Delta T
            # 问题: 我们不知道 T+step 的确切日期。
            # 我们需要估计它。假设 1 年间隔？或使用平均 delta？
            # 或者使用 ARIMA 预测 'time'？时间通常是规则的或输入的。
            # 这里假设未来步骤间隔约为 1 年 (365 天)。
            estimated_delta_year = 1.0 
            feat_dict['delta_t'] = estimated_delta_year
            
            # 创建 X DataFrame
            X_input = pd.DataFrame([feat_dict])
            
            # 3. 预测 (根据配置选择 Residual 或 Direct)
            if self.config.experiment.learning_target == "direct":
                # Direct Forecasting
                # 如果训练时使用了 y_arima_next 作为特征，这里需要确保 X_input 中包含它
                # 我们在上面 feat_dict 中并没有显式添加 'y_arima_next'
                # 让我们加上它 (target 的 ARIMA 预测)
                feat_dict['y_arima_next'] = y_arima
                
                # 更新 X_input
                X_input = pd.DataFrame([feat_dict])
                
                # 模型直接输出 y_pred
                direct_pred = self.residual_model.predict(X_input)[0]
                y_final = direct_pred
                residual_pred = 0.0 # 这种模式下没有残差概念，或者说 residual = y_final - y_arima (为了兼容记录)
            else:
                # Residual Forecasting (Default)
                # 模型输出 residual
                residual_pred = self.residual_model.predict(X_input)[0]
                y_final = y_arima + residual_pred
            
            # 5. 准备下一步 (T+step) 的状态
            # 我们需要向 current_history 追加一行，代表 T+step 的预测状态
            # 这一行将用作 step+1 的输入
            
            next_time = last_real_time + pd.Timedelta(days=365 * step) # 近似时间
            
            new_row = {
                self.config.columns.person_id_col: pid,
                self.config.columns.time_col: next_time,
                self.config.columns.target_col: y_final, # 预测目标
            }
            
            # 静态列继承
            for col in self.config.columns.static_cols:
                new_row[col] = input_row[col]
                
            # 随时间变化列: 使用 ARIMA 预测它们
            for col in self.config.columns.timevarying_cols:
                if col == self.config.columns.target_col:
                    continue # 已经做过
                
                # 预测特征值
                # 注意: 我们需要基于 current_history 预测 T+step 的特征值。
                # 实际上 predict_feature_next(history, steps=1) 预测的是 history 之后的一步。
                # current_history 此时包含 [0...T+step-1] (因为我们在循环开始时追加了 T+step-1 的预测)
                # 所以调用 predict_feature_next 会返回 T+step 的预测值。
                
                feat_val = self.arima.predict_feature_next(current_history, col, steps=1)[0]
                new_row[col] = feat_val
                
            # 追加到历史
            current_history = pd.concat([current_history, pd.DataFrame([new_row])], ignore_index=True)
            
            # 记录预测
            predictions.append({
                'person_id': pid,
                'horizon': step,
                'time_pred': next_time,
                'y_arima': y_arima,
                'residual_pred': residual_pred,
                'y_pred': y_final,
                'delta_t': estimated_delta_year
            })
            
        return pd.DataFrame(predictions)
