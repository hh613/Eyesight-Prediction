import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
from src.models.arima.arima_forecaster import ArimaForecaster

class ArimaFeatureGenerator:
    def __init__(self, 
                 timevarying_cols: List[str], 
                 forecaster_params: Dict = None):
        """
        ARIMA 特征生成器，用于多步递归预测中生成未来的协变量。
        
        参数:
            timevarying_cols: 需要预测的特征列名列表
            forecaster_params: 传递给 ArimaForecaster 的参数字典
        """
        self.timevarying_cols = timevarying_cols
        self.forecaster_params = forecaster_params or {}
        # 缓存: {(person_id, feature_col): ArimaForecaster}
        self.model_cache: Dict[Tuple[str, str], ArimaForecaster] = {}

    def predict_next(self, 
                     history_df: pd.DataFrame, 
                     person_id_col: str, 
                     steps: int = 1) -> pd.DataFrame:
        """
        预测指定个体的未来特征值。
        
        参数:
            history_df: 单个个体的历史数据 DataFrame
            person_id_col: 个体 ID 列名
            steps: 预测步数
            
        返回:
            DataFrame, 包含 timevarying_cols 的未来预测值，索引为 0..steps-1
        """
        if len(history_df) == 0:
            return pd.DataFrame(columns=self.timevarying_cols, index=range(steps))
            
        pid = history_df[person_id_col].iloc[0]
        future_feats = {}
        
        for col in self.timevarying_cols:
            if col not in history_df.columns:
                continue
                
            cache_key = (str(pid), col)
            series = history_df[col]
            
            # 检查缓存
            if cache_key in self.model_cache:
                forecaster = self.model_cache[cache_key]
                # 理想情况下，如果是严格的滚动预测，每一步有了新真实值都应 refit。
                # 但如果在递归生成未来多步（K步）时，我们在 t 时刻 fit 一次，
                # 然后生成 t+1, t+2... t+K。
                # 这个函数通常是在 t 时刻调用一次，生成 K 步。
                
                # 如果 history_df 变长了（有了新的真实观测），我们应该 refit。
                # 简单起见，这里假设如果缓存存在，我们就复用模型参数（best order），
                # 但 ARIMA 需要在新的 history 上 re-fit/update 状态。
                # statsmodels 的 fit 比较重。
                
                # 策略: 始终 refit，但复用 grid search 找到的最佳 order (如果启用了 grid search)。
                # 或者：如果这是一个完全全新的预测请求（不同的 t），我们需要 refit。
                
                # 为了简化且符合"缓存模型或参数"的要求：
                # 我们重新创建一个 Forecaster，但如果之前 grid search 过，可以复用参数。
                # 这里我们直接缓存 Forecaster 对象，并在需要时调用 fit。
                pass
            else:
                forecaster = ArimaForecaster(**self.forecaster_params)
                self.model_cache[cache_key] = forecaster
            
            # 拟合 (Statsmodels ARIMA fit is fast enough for single series if order is fixed)
            # 如果启用了 Grid Search，且已经找到 best_order，我们可以锁定它以加速后续 refit?
            # ArimaForecaster 内部逻辑：如果 use_grid_search=True，每次 fit 都会搜。
            # 改进: 可以在 ArimaForecaster 里增加 logic，如果已有 best_order，下次 fit 可选跳过 search。
            # 目前 ArimaForecaster 每次 fit 都会重跑。
            
            forecaster.fit(series)
            pred = forecaster.predict_next(steps, history_series=series)
            future_feats[col] = pred
            
        return pd.DataFrame(future_feats)
    
    def clear_cache(self):
        self.model_cache = {}
