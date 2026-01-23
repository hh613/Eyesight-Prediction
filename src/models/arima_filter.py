import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import warnings
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

# 忽略 ARIMA 拟合时的警告 (如收敛警告)
warnings.filterwarnings("ignore")

class ARIMATrendFilter:
    def __init__(self, order=(1, 1, 0), forecast_steps=1):
        """
        初始化 ARIMA 滤波器。
        
        Args:
            order (tuple): ARIMA 参数 (p, d, q)。默认 (1, 1, 0) 强调自回归趋势。
                           d=1 表示一阶差分，适用于非平稳的近视进展过程。
                           MA(q) 成分用于锚定基线斜率。
            forecast_steps (int): 向前预测步数。
        """
        self.order = order
        self.forecast_steps = forecast_steps

    def fit_predict_single(self, series, delta_t=None):
        """
        对单个序列进行拟合和预测。
        
        Args:
            series (array-like): 历史 SE 值序列。
            delta_t (float, optional): 距离下一个时间点的间隔。
                                       注意: 标准 ARIMA 假设等间距。对于非等间距，
                                       我们通常假设 series 是按时间排序的观测值，
                                       预测的是 "Next Step"。
                                       如果要处理 delta_t，可以在预测后进行线性缩放，
                                       或者将 ARIMA 建模在速率空间。
                                       
                                       这里采用简化假设: ARIMA 捕捉的是序列的内在惯性趋势。
                                       预测值 Y_{t+1} 代表了 "下一个自然观测点" 的趋势。
                                       
        Returns:
            float: 下一步的预测趋势值 (Trend Anchor)。
        """
        # 数据点太少无法拟合 ARIMA
        if len(series) < 3:
            # 回退策略: 线性外推 或 均值外推
            if len(series) >= 2:
                # 简单线性外推: Last + (Last - Prev)
                slope = series[-1] - series[-2]
                return series[-1] + slope
            elif len(series) == 1:
                # 只有 1 个点，假设按 -0.5D/年 进展 (需外部 delta_t)
                # 如果没有 delta_t，只能返回原值
                return series[-1] - 0.25 # 假设半年的量? 这是一个非常粗糙的 fallback
            else:
                return np.nan

        try:
            # 拟合 ARIMA
            # 注意: 如果序列是常数，ARIMA 可能会报错或警告，需处理
            if np.std(series) < 1e-6:
                return series[-1]

            model = ARIMA(series, order=self.order)
            model_fit = model.fit()
            
            # 预测
            forecast = model_fit.forecast(steps=self.forecast_steps)
            return forecast[0]
            
        except Exception as e:
            # 拟合失败回退策略: 线性外推
            slope = series[-1] - series[-2]
            return series[-1] + slope

    def batch_process(self, df, id_col='student_id', val_col='SE_right', time_col='check_date'):
        """
        批量处理 DataFrame 中的所有学生。
        为每个时间点生成 'SE_trend_arima'。
        
        注意: 这是一个 "Rolling Forecast" (滚动预测) 问题。
        对于 t 时刻，我们只能利用 t 之前 (包含 t-1, t-2...) 的数据来构建模型，
        然后预测 t 时刻的趋势值，作为 Baseline。
        
        Args:
            df (pd.DataFrame): 包含所有数据的 DataFrame。
            
        Returns:
            pd.Series: 与 df 索引对齐的预测趋势值序列。
        """
        # 确保按 ID 和 时间 排序
        df = df.sort_values(by=[id_col, time_col])
        
        # 结果容器
        predictions = pd.Series(index=df.index, dtype=float)
        predictions[:] = np.nan
        
        # 按学生分组
        grouped = df.groupby(id_col)
        
        # 这是一个计算密集型任务，对于大数据集可能很慢。
        # 针对每个学生的每个时间点 t (t >= 2)，取 history [0...t-1] 训练并预测 t。
        
        # 优化: 我们可以只针对每个学生计算一次吗？
        # 不，用户需求是 "Y_trend, t+dt"，即每个点的趋势值。
        # 这意味着我们需要对每个记录行，根据其之前的历史进行预测。
        
        print(f"正在进行 ARIMA 滚动滤波 (总学生数: {len(grouped)})... 这可能需要一些时间。")
        
        # 为了加速，我们可以使用简单的循环，因为每个分组通常很小 (<10 条记录)
        # 甚至可以不做复杂的 ARIMA，只对长度 > 3 的序列做。
        
        results = []
        
        for student_id, group in grouped:
            indices = group.index.tolist()
            values = group[val_col].tolist()
            
            # 对组内每个点进行滚动预测
            # 第 0 个点: 无历史，无法预测 (NaN)
            # 第 1 个点: 只有 0，无法 ARIMA (Fallback)
            # 第 2 个点: 有 0, 1 (Fallback Linear)
            # 第 3 个点: 有 0, 1, 2 (ARIMA start)
            
            for i in range(len(indices)):
                current_idx = indices[i]
                
                # 历史数据: 0 到 i-1
                history = values[:i]
                
                if len(history) < 1:
                    continue # 保持 NaN
                    
                # 预测
                pred = self.fit_predict_single(history)
                predictions[current_idx] = pred
                
        return predictions

