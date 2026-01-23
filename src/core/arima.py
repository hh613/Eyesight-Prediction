import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import logging
import warnings
from typing import Dict, Optional, Tuple

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

class ArimaForecaster:
    def __init__(self, config):
        self.config = config.arima
        self.models = {} # 如果需要，可按 student_id 缓存模型或参数
        # 注意: 为 6000 名学生存储完整的 ARIMA 结果对象可能非常耗费内存。
        # 我们可能只存储预测所需的最后一段历史数据。
        # 在此实现中，为了节省内存，我们在预测时即时拟合，
        # 或者我们可以实现轻量级的状态存储。
    
    def fit_predict_single(self, series: pd.Series, steps: int = 1) -> np.ndarray:
        """
        在单个序列上拟合 ARIMA 并预测后续步骤。
        包含失败回退处理。
        """
        # 数据验证
        clean_series = series.dropna()
        if len(clean_series) < 3:
            return self._fallback_forecast(clean_series, steps)

        try:
            # TODO: 如果 config.use_grid_search 为 True，则实现网格搜索
            # 目前使用固定阶数 (order)
            model = ARIMA(clean_series, order=self.config.order)
            model_fit = model.fit()
            forecast = model_fit.forecast(steps=steps)
            return forecast.values
        except Exception as e:
            # logger.warning(f"ARIMA 拟合失败: {e}。使用回退策略。")
            return self._fallback_forecast(clean_series, steps)

    def _fallback_forecast(self, series: pd.Series, steps: int) -> np.ndarray:
        """
        回退策略: 'linear_trend' (线性趋势) 或 'last_value' (最后值)
        """
        if len(series) == 0:
            return np.array([np.nan] * steps)
            
        last_val = series.iloc[-1]
        
        if self.config.fallback_strategy == "last_value" or len(series) < 2:
            return np.array([last_val] * steps)
            
        if self.config.fallback_strategy == "linear_trend":
            # 基于最后两点的简单斜率
            slope = series.iloc[-1] - series.iloc[-2]
            # 是否限制斜率以避免爆炸？目前保持原始线性。
            forecast = [last_val + slope * (i + 1) for i in range(steps)]
            return np.array(forecast)
            
        return np.array([last_val] * steps)

    def predict_next(self, history_df: pd.DataFrame, target_col: str, steps: int = 1) -> np.ndarray:
        """
        基于历史数据预测目标列的未来值。
        假设 history_df 包含单个人的数据，并已按时间排序。
        """
        return self.fit_predict_single(history_df[target_col], steps)

    def predict_feature_next(self, history_df: pd.DataFrame, feature_col: str, steps: int = 1) -> np.ndarray:
        """
        预测随时间变化的特征的未来值。
        """
        return self.fit_predict_single(history_df[feature_col], steps)
