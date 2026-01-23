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
        特征工程主流程 (简化版 - 方案A)
        只负责基础清洗、重命名、静态映射和基础时间计算 (delta_t)。
        复杂的动态特征 (ARIMA, Rolling Stats, WMA) 移交给 src/core 在实验阶段动态计算。
        """
        print("开始特征工程 (基础)...")
        
        # 1. 字段重命名与标准化
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
    
        # 2. 核心屈光指标
        if 'cyl_right' in df.columns:
            df['C_cyl_right'] = df['cyl_right']
            
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
        df['has_glasses'] = 0
        mask_glasses = (df['correction_method'].astype(str).str.contains('框架', na=False)) | \
                       (df['VA_corrected'].notna())
        df.loc[mask_glasses, 'has_glasses'] = 1
        
        # correct_level
        df['correct_level'] = df.apply(lambda row: mapper.calculate_correct_level(
            row.get('correction_method'), 
            row.get('VA_corrected'), 
            row.get('VA_unaided')
        ), axis=1)

        # 4. 基础时间特征
        if 'student_id' in df.columns and 'check_date' in df.columns:
            df = df.sort_values(by=['student_id', 'check_date'])
            
            # 计算 delta_t (距离上一次检查的时间间隔，单位：年)
            df['prev_date'] = df.groupby('student_id')['check_date'].shift(1)
            df['delta_t'] = (df['check_date'] - df['prev_date']).dt.days / 365.25
            
        # 定义保留列
        final_columns = [
            'student_id', 'check_date', 'age', 'delta_t',
            'SE_right', 'C_cyl_right', 'VA_unaided', 'VA_corrected',
            'gender', 'school_type', 'has_glasses', 'correct_level'
        ]
        
        # 仅保留存在的列
        existing_final_cols = [c for c in final_columns if c in df.columns]
        df = df[existing_final_cols]

        print("特征工程 (基础) 完成。")
        return df
