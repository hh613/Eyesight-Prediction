import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import logging
import warnings
from typing import Optional, Tuple, List, Dict, Any
from itertools import product

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

class ArimaForecaster:
    def __init__(self, 
                 order: Tuple[int, int, int] = (1, 1, 0), 
                 use_grid_search: bool = False,
                 grid_search_params: Optional[Dict[str, List[int]]] = None,
                 fallback_strategy: str = "linear_trend"):
        """
        ARIMA 预测器，支持单序列拟合与预测。
        
        参数:
            order: 默认 (p, d, q)
            use_grid_search: 是否启用网格搜索选择最佳 (p, d, q)
            grid_search_params: 网格搜索范围，例如 {'p': [0,1,2], 'd': [0,1], 'q': [0,1]}
            fallback_strategy: 'linear_trend' (线性趋势) 或 'last_value' (最后值)
        """
        self.default_order = order
        self.use_grid_search = use_grid_search
        self.grid_search_params = grid_search_params or {'p': [0, 1, 2], 'd': [0, 1], 'q': [0, 1]}
        self.fallback_strategy = fallback_strategy
        self.fallback_count = 0
        self.best_order = None # 缓存最佳参数
        self.model_fit = None  # 缓存拟合好的模型对象

    def fit(self, series: pd.Series) -> 'ArimaForecaster':
        """
        拟合模型。
        """
        clean_series = series.dropna()
        if len(clean_series) < 3:
            # 数据点太少，无法拟合，标记模型未拟合
            self.model_fit = None
            return self

        try:
            if self.use_grid_search:
                self._fit_grid_search(clean_series)
            else:
                self._fit_fixed(clean_series, self.default_order)
        except Exception as e:
            # logger.debug(f"ARIMA fit failed: {e}")
            self.model_fit = None
            
        return self

    def predict_next(self, steps: int = 1, history_series: Optional[pd.Series] = None) -> np.ndarray:
        """
        预测未来步骤。
        如果 history_series 提供，则用于 fallback 计算（当模型未拟合时）。
        """
        if self.model_fit is not None:
            try:
                # 使用拟合好的模型预测
                forecast = self.model_fit.forecast(steps=steps)
                return forecast.values
            except Exception as e:
                # 预测出错，转入 fallback
                pass
        
        # Fallback
        self.fallback_count += 1
        target_series = history_series if history_series is not None else pd.Series([])
        target_series = target_series.dropna()
        return self._fallback_forecast(target_series, steps)

    def _fit_fixed(self, series: pd.Series, order: Tuple[int, int, int]):
        model = ARIMA(series, order=order)
        self.model_fit = model.fit()
        self.best_order = order

    def _fit_grid_search(self, series: pd.Series):
        best_aic = float('inf')
        best_order = self.default_order
        best_model = None
        
        ps = self.grid_search_params.get('p', [0, 1])
        ds = self.grid_search_params.get('d', [0, 1])
        qs = self.grid_search_params.get('q', [0, 1])
        
        for p, d, q in product(ps, ds, qs):
            try:
                order = (p, d, q)
                model = ARIMA(series, order=order)
                res = model.fit()
                if res.aic < best_aic:
                    best_aic = res.aic
                    best_order = order
                    best_model = res
            except:
                continue
                
        if best_model is not None:
            self.model_fit = best_model
            self.best_order = best_order
        else:
            # Grid search 全部失败，尝试默认
            self._fit_fixed(series, self.default_order)

    def _fallback_forecast(self, series: pd.Series, steps: int) -> np.ndarray:
        """
        回退策略实现。
        """
        if len(series) == 0:
            return np.array([np.nan] * steps)
            
        last_val = series.iloc[-1]
        
        if self.fallback_strategy == "last_value" or len(series) < 2:
            return np.array([last_val] * steps)
            
        if self.fallback_strategy == "linear_trend":
            # 简单线性趋势：取最后两点
            # 也可以改为取整体趋势 np.polyfit(..., 1)
            slope = series.iloc[-1] - series.iloc[-2]
            # 限制斜率过大? 暂时保持简单
            forecast = [last_val + slope * (i + 1) for i in range(steps)]
            return np.array(forecast)
            
        return np.array([last_val] * steps)
