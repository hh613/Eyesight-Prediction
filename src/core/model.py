import pandas as pd
import numpy as np
import logging
from typing import Dict, Any
from src.core.config import Config

# 尝试导入模型，如果未安装则处理
try:
    import xgboost as xgb
except ImportError:
    xgb = None
    
try:
    import lightgbm as lgb
except ImportError:
    lgb = None
    
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

logger = logging.getLogger(__name__)

class ResidualRegressor:
    def __init__(self, config: Config):
        self.config = config
        self.model = self._init_model()
        self.feature_cols = None # 将在 fit 期间设置

    def _init_model(self):
        m_type = self.config.model.model_type
        params = self.config.model.model_params
        
        # 复制参数，避免修改原始配置
        model_params = params.copy()
        
        # 如果参数中包含 'learning_rate' 且模型不是 boosting，则移除
        # RandomForest 不支持 learning_rate
        if 'learning_rate' in model_params and m_type == 'random_forest':
            del model_params['learning_rate']
        
        if m_type == "xgboost":
            if xgb is None:
                logger.warning("未安装 XGBoost，回退到 Random Forest")
                # 回退时也要清理不兼容参数
                if 'learning_rate' in model_params:
                    del model_params['learning_rate']
                return RandomForestRegressor(**model_params)
            
            # XGBoost 需要 enable_categorical=True 来支持 category 类型
            # 同时也建议设置 tree_method='hist' 或 'gpu_hist' (如果可用)
            if 'enable_categorical' not in model_params:
                model_params['enable_categorical'] = True
                
            # 尝试启用 GPU 加速
            # 这里简单硬编码，实际可根据 torch.cuda.is_available() 动态判断
            try:
                import torch
                if torch.cuda.is_available():
                    model_params['device'] = 'cuda'
                    # model_params['tree_method'] = 'hist' # XGBoost 2.0+ 自动选择，显式设置device即可
            except ImportError:
                pass
                
            return xgb.XGBRegressor(**model_params)
            
        elif m_type == "lightgbm":
            if lgb is None:
                logger.warning("未安装 LightGBM，回退到 Random Forest")
                # 回退时也要清理不兼容参数
                if 'learning_rate' in model_params:
                    del model_params['learning_rate']
                return RandomForestRegressor(**model_params)
            return lgb.LGBMRegressor(**model_params)
            
        elif m_type == "random_forest":
            return RandomForestRegressor(**model_params)
            
        else:
            raise ValueError(f"未知的模型类型: {m_type}")

    def fit(self, X_df: pd.DataFrame, y: pd.Series):
        """
        训练残差模型。
        X_df: 特征 DataFrame
        y: 残差目标
        """
        # 排除可能混入的元数据列
        meta_cols = ['person_id', 't_idx', 'time_t', 'time_next', 'y_true_next', 'y_arima_next', 'residual']
        self.feature_cols = [c for c in X_df.columns if c not in meta_cols]
        
        X = X_df[self.feature_cols].copy()
        
        # 确保所有 object 类型的列转换为 category (针对 XGBoost/LightGBM)
        for col in X.columns:
            if X[col].dtype == 'object':
                X[col] = X[col].astype('category')
                
        logger.info(f"正在使用特征训练模型: {self.feature_cols}")
        
        self.model.fit(X, y)

    def predict(self, X_df: pd.DataFrame) -> np.ndarray:
        """
        预测残差。
        """
        if self.feature_cols is None:
            raise ValueError("模型尚未训练。")
            
        # 确保列匹配
        X = X_df[self.feature_cols].copy()
        
        # 同样需要转换类型
        for col in X.columns:
            if X[col].dtype == 'object':
                X[col] = X[col].astype('category')
                
        return self.model.predict(X)
