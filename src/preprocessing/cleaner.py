import pandas as pd
import numpy as np
from src.preprocessing import mapper

class DataCleaner:
    def __init__(self, config=None):
        self.config = config or {}
        #设置默认阈值
        self.missing_threshold = 0.8
        self.min_interval_months = 5
        self.valid_corrections = ['不戴镜', '框架眼镜']
        
        # 异常值检测范围
        self.ranges = {
            'sph': (-20, 10),
            'cyl': (-10, 10),
            'se': (-20, 10) # 远视会被单独过滤，这里保留作为一般性的合理性检查
        }

    def load_data(self, filepath):
        print(f"正在从 {filepath} 加载数据...")
        return pd.read_excel(filepath)

    def preprocess(self, df):
        """
        主预处理流程
        """
        original_count = len(df)
        print(f"原始记录数: {original_count}")

        # 1. 根据计划名称过滤 (仅保留筛查计划)
        if '计划名称' in df.columns:
            df = df[df['计划名称'].astype(str).str.contains('筛查', na=False)]
            print(f"过滤非筛查计划后: {len(df)}")
        
        # 2. 根据矫正方式过滤
        if '矫正方式' in df.columns:
            df = df[df['矫正方式'].isin(self.valid_corrections)]
            print(f"过滤矫正方式后: {len(df)}")

        # 3. 处理缺失值 (列)
        # 删除缺失率 > 80% 的列
        missing_ratios = df.isnull().mean()
        cols_to_keep = missing_ratios[missing_ratios <= self.missing_threshold].index
        df = df[cols_to_keep]
        print(f"保留列数 (缺失率 < {self.missing_threshold*100}%): {len(cols_to_keep)}/{len(missing_ratios)}")
        
        # 4. 计算/重新计算 SE 并移除异常值
        # 确保光学相关列为数值类型
        optical_cols = ['球镜右', '柱镜右', '等效球镜右', '球镜左', '柱镜左', '等效球镜左']
        for col in optical_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # 如果存在球镜和柱镜，重新计算 SE 以填充空缺
        if '球镜右' in df.columns and '柱镜右' in df.columns:
            df['calculated_se_right'] = mapper.calculate_se(df['球镜右'], df['柱镜右'])
            df['等效球镜右'] = df['等效球镜右'].fillna(df['calculated_se_right'])
        
        if '球镜左' in df.columns and '柱镜左' in df.columns:
            df['calculated_se_left'] = mapper.calculate_se(df['球镜左'], df['柱镜左'])
            df['等效球镜左'] = df['等效球镜左'].fillna(df['calculated_se_left'])

        # 移除异常测量值
        df = self._remove_outliers(df)
        print(f"移除异常值后: {len(df)}")

        # 5. 移除远视数据 (SE > 0)
        # 假设我们关注近视预测，过滤掉右眼 SE > 0 的数据
        if '等效球镜右' in df.columns:
            df = df[df['等效球镜右'] <= 0]
            print(f"移除远视数据后 (SE_R > 0): {len(df)}")

        # 6. 计算年龄
        if '身份证号' in df.columns and '检查时间' in df.columns:
            df['birth_date'] = df['身份证号'].apply(mapper.get_birth_date_from_id)
            df['检查时间'] = pd.to_datetime(df['检查时间'], errors='coerce')
            
            # 移除日期无效的行
            df = df.dropna(subset=['birth_date', '检查时间'])
            
            df['age'] = df.apply(lambda row: mapper.calculate_age(row['birth_date'], row['检查时间']), axis=1)
            print(f"计算年龄后 (有效日期): {len(df)}")
        
        # 7. 时间间隔过滤 (相邻记录间隔 > 5个月)
        df = self._filter_intervals(df)
        print(f"时间间隔过滤后: {len(df)}")

        return df

    def _remove_outliers(self, df):
        # 基于范围过滤
        if '球镜右' in df.columns:
            df = df[(df['球镜右'] >= self.ranges['sph'][0]) & (df['球镜右'] <= self.ranges['sph'][1])]
        if '柱镜右' in df.columns:
            df = df[(df['柱镜右'] >= self.ranges['cyl'][0]) & (df['柱镜右'] <= self.ranges['cyl'][1])]
        if '等效球镜右' in df.columns:
            df = df[(df['等效球镜右'] >= self.ranges['se'][0]) & (df['等效球镜右'] <= self.ranges['se'][1])]
        return df

    def _filter_intervals(self, df):
        """
        仅保留相邻记录间隔 > 5个月的数据。
        策略: 对每个学生按日期排序。保留第一条。
        遍历后续记录，如果 (当前时间 - 上次保留时间) > 5个月，则保留。
        """
        if '学生标识' not in df.columns or '检查时间' not in df.columns:
            return df
            
        df = df.sort_values(by=['学生标识', '检查时间'])
        
        kept_indices = []
        
        # 按学生分组
        grouped = df.groupby('学生标识')
        
        for student_id, group in grouped:
            if len(group) < 1:
                continue
                
            group_indices = group.index.tolist()
            dates = group['检查时间'].tolist()
            
            # 总是保留第一条记录
            last_kept_date = dates[0]
            kept_indices.append(group_indices[0])
            
            for i in range(1, len(dates)):
                current_date = dates[i]
                # 计算月份差
                diff_days = (current_date - last_kept_date).days
                diff_months = diff_days / 30.44
                
                if diff_months > self.min_interval_months:
                    kept_indices.append(group_indices[i])
                    last_kept_date = current_date
        
        return df.loc[kept_indices]
