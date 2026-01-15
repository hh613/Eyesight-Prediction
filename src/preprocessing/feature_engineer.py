import pandas as pd
import numpy as np
from src.preprocessing import mapper
# 尝试导入 arima filter，如果尚未实现则忽略
try:
    from src.models.arima_filter import ARIMATrendFilter
    HAS_ARIMA = True
except ImportError:
    HAS_ARIMA = False

class FeatureEngineer:
    def __init__(self):
        pass

    def transform(self, df):
        """
        特征工程主流程
        """
        print("开始特征工程...")
        
        # 1. 字段重命名与标准化
        # 映射: 身份证号 -> student_id, 检查时间 -> check_date
        # 注意: cleaner.py 中已经生成了 age
        rename_map = {
            '身份证号': 'student_id',
            '检查时间': 'check_date',
            '性别': 'gender_raw',
            '筛查区': 'district',
            '矫正方式': 'correction_method',
            '裸眼右': 'VA_unaided_right',
            '戴镜右': 'VA_corrected_right',
            '裸眼左': 'VA_unaided_left',
            '戴镜左': 'VA_corrected_left',
            '球镜右': 'sph_right',
            '柱镜右': 'cyl_right',
            '等效球镜右': 'SE_right',
            '球镜左': 'sph_left',
            '柱镜左': 'cyl_left',
            '等效球镜左': 'SE_left'
        }
        
        # 仅重命名存在的列
        df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
        
        # 如果 cleaner 已经计算了 age，保留。如果没有，重新计算 (这里假设 cleaner 已计算)
        
        # 2. 核心屈光指标
        # 确保列名为 C_cyl_right (用户要求)
        if 'cyl_right' in df.columns:
            df['C_cyl_right'] = df['cyl_right']
        if 'cyl_left' in df.columns:
            df['C_cyl_left'] = df['cyl_left']
            
        # 统一 VA 列名 (如果用户需要 VA_unaided 无左右之分? 通常取右眼或双眼。
        # 用户输入: "VA_unaided / VA_corrected". 暂且保留右眼作为主眼，或者保留两者)
        # 习惯上眼视光研究常取右眼 (OD) 分析，除非另有说明。这里保留右眼作为主指标。
        if 'VA_unaided_right' in df.columns:
            df['VA_unaided'] = df['VA_unaided_right']
        if 'VA_corrected_right' in df.columns:
            df['VA_corrected'] = df['VA_corrected_right']

        # 3. 静态与干预特征
        if 'gender_raw' in df.columns:
            df['gender'] = df['gender_raw'].apply(mapper.map_gender)
            
        if 'district' in df.columns:
            df['school_type'] = df['district'].apply(mapper.map_school_type)
            
        # has_glasses
        # 逻辑: 戴镜视力不为空 OR 矫正方式为框架眼镜
        df['has_glasses'] = 0
        mask_glasses = (df['correction_method'].astype(str).str.contains('框架', na=False)) | \
                       (df['VA_corrected'].notna())
        df.loc[mask_glasses, 'has_glasses'] = 1
        
        # correct_level
        # 依赖于 has_glasses 和 视力数据
        df['correct_level'] = df.apply(lambda row: mapper.calculate_correct_level(
            row.get('correction_method'), 
            row.get('VA_corrected'), 
            row.get('VA_unaided')
        ), axis=1)

        # 4. 时间序列特征 (纵向建模基础)
        # 需要按 student_id 排序
        if 'student_id' in df.columns and 'check_date' in df.columns:
            df = df.sort_values(by=['student_id', 'check_date'])
            
            # 计算 delta_t (距离上一次检查的时间间隔，单位：年)
            # 使用 shift
            df['prev_date'] = df.groupby('student_id')['check_date'].shift(1)
            df['delta_t'] = (df['check_date'] - df['prev_date']).dt.days / 365.25
            
            # 第一均为 NaN，delta_t 此时可以设为 0 或者保持 NaN (马尔可夫通常需要间隔)
            # 用户描述: "当前检查距离上一次检查的时间间隔"。第一没有上一次，故为 NaN 或 0。
            # 通常处理: 第一点无法作为转移的目标(无来源)，或者作为序列起点。
            
            # SE_rate_hist: (SE_t - SE_{t-1}) / delta_t
            if 'SE_right' in df.columns:
                df['prev_SE'] = df.groupby('student_id')['SE_right'].shift(1)
                df['SE_change'] = df['SE_right'] - df['prev_SE']
                
                # 避免除以0 (虽然预处理保证了间隔 > 5个月)
                df['SE_rate_hist'] = df['SE_change'] / df['delta_t']
                
        # 5. ARIMA 相关特征 (占位或简单实现)
        # SE_trend_arima: 历史预测趋势
        # SE_residual: 真实 - 趋势
        # 由于 ARIMA 比较耗时且通常是在训练阶段作为滤波器，这里我们做一个简单的线性趋势作为 "Baseline Trend"
        # 或者如果 arima_filter 模块可用，调用它。
        # 鉴于性能，这里暂时填充 NaN，或者使用简单的线性外推 (Linear Extrapolation)
        # 简单的线性外推: SE_trend = prev_SE + prev_rate * delta_t
        if 'prev_SE' in df.columns and 'SE_rate_hist' in df.columns:
            # 使用上一步的速率预测当前? 不，SE_rate_hist 是利用当前计算出来的。
            # 真正的预测应该用 (t-2) 到 (t-1) 的速率来预测 t。
            
            df['prev_SE_rate'] = df.groupby('student_id')['SE_rate_hist'].shift(1)
            
            # 如果有前一步的速率，则 SE_trend = prev_SE + prev_SE_rate * delta_t
            # 如果没有(即序列第二点)，可以用一个全局平均速率或假设为0
            
            # 简单实现: 
            # 序列点 1: 无历史，无趋势
            # 序列点 2: 有 delta_t, 但无 "prev_rate"。只能用当前斜率? 不，那是作弊。
            # 这是一个 "Filtering" 过程。
            # 这里先创建一个简单的 Baseline: 假设近视进展率为 -0.5 D/年 (常见均值)
            mean_progression = -0.5
            
            # 逻辑: SE_trend_arima = prev_SE + (mean_progression * delta_t) 
            # 这只是一个 Placeholder，实际应该调用 ARIMA 模型
            df['SE_trend_arima'] = df['prev_SE'] + (mean_progression * df['delta_t'])
            
            # 修正: 如果有 prev_SE_rate (即至少是第三次检查)，可以用个体的历史速率
            mask_has_hist = df['prev_SE_rate'].notna()
            df.loc[mask_has_hist, 'SE_trend_arima'] = df.loc[mask_has_hist, 'prev_SE'] + \
                (df.loc[mask_has_hist, 'prev_SE_rate'] * df.loc[mask_has_hist, 'delta_t'])
            
            # 计算 Residual
            df['SE_residual'] = df['SE_right'] - df['SE_trend_arima']

        # 清理临时列，并只保留用户指定的字段
        # 目标字段列表
        final_columns = [
            'student_id', 'check_date', 'age', 'delta_t',
            'SE_right', 'SE_left', 'C_cyl_right', 'C_cyl_left', 'VA_unaided', 'VA_corrected',
            'SE_trend_arima', 'SE_residual', 'SE_rate_hist',
            'gender', 'school_type', 'has_glasses', 'correct_level'
        ]
        
        # 仅保留存在的列
        existing_final_cols = [c for c in final_columns if c in df.columns]
        df = df[existing_final_cols]

        print("特征工程完成。")
        return df
