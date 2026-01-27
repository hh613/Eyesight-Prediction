import pandas as pd
import numpy as np
import os

def calculate_stats():
    # 数据文件路径，基于 preprocessing 处理完的 featured_data.csv
    file_path = 'data/processed/featured_data.csv'
    
    if not os.path.exists(file_path):
        print(f"错误: 找不到文件 {file_path}")
        return

    # 加载数据
    df = pd.read_csv(file_path)
    
    # 确保日期列是 datetime 格式
    df['check_date'] = pd.to_datetime(df['check_date'])

    # 1. 获取基线数据 (Baseline Data)
    df_sorted = df.sort_values(['student_id', 'check_date'])
    baseline_df = df_sorted.groupby('student_id').first()
    
    total_participants = len(baseline_df)

    # --- A. 年龄分段统计 (Age Grouping) ---
    # 定义年龄段
    bins = [0, 6, 11.99, 15.99, 18.99, 100]
    labels = ['<6', '6-11', '12-15', '16-18', '>18']
    baseline_df['age_group'] = pd.cut(baseline_df['age'], bins=bins, labels=labels, right=True)
    
    age_group_counts = baseline_df['age_group'].value_counts().reindex(labels)
    age_group_pct = baseline_df['age_group'].value_counts(normalize=True).reindex(labels) * 100

    # --- B. 性别分布 (Gender) ---
    gender_counts = baseline_df['gender'].value_counts()
    gender_pct = baseline_df['gender'].value_counts(normalize=True) * 100
    
    # --- C. 基线等效球镜度数 (Baseline SE) ---
    mean_se = baseline_df['SE_right'].mean()
    std_se = baseline_df['SE_right'].std()

    # --- D. 城市/郊区分布 (Urban/Rural) ---
    school_counts = baseline_df['school_type'].value_counts()
    school_pct = baseline_df['school_type'].value_counts(normalize=True) * 100

    # --- E. 矫正方式分布 (Correction) ---
    glasses_counts = baseline_df['has_glasses'].value_counts()
    glasses_pct = baseline_df['has_glasses'].value_counts(normalize=True) * 100

    # --- F. 其他追踪统计 ---
    duration_stats = df.groupby('student_id')['check_date'].agg(['min', 'max'])
    duration_years = (duration_stats['max'] - duration_stats['min']).dt.days / 365.25
    mean_duration = duration_years.mean()

    exam_counts = df.groupby('student_id').size()
    mean_exams = exam_counts.mean()

    df_sorted['prev_date'] = df_sorted.groupby('student_id')['check_date'].shift(1)
    df_sorted['interval_days'] = (df_sorted['check_date'] - df_sorted['prev_date']).dt.days
    intervals_months = df_sorted['interval_days'].dropna() / 30.44
    min_interval = intervals_months.min()
    max_interval = intervals_months.max()

    # 打印统计结果
    print("="*50)
    print("           人群队列详细统计概览 (Detailed Cohort Stats)")
    print("="*50)
    print(f"总参与人数 (Total Participants): {total_participants}")
    print("-" * 40)
    
    print(f"1. 年龄分段 (Age Groups):")
    for label in labels:
        count = age_group_counts[label]
        pct = age_group_pct[label]
        if not np.isnan(count):
            print(f"   {label} 岁: {int(count)} 人 ({pct:.1f}%)")
    
    print(f"\n2. 性别分布 (Gender):")
    for val, count in gender_counts.items():
        label = "男(假设)" if val == 1 else "女(假设)"
        print(f"   数值 {val} ({label}): {count} 人 ({gender_pct[val]:.1f}%)")
        
    print(f"\n3. 基线等效球镜 (Baseline SE_right):")
    print(f"   平均度数: {mean_se:.2f} ± {std_se:.2f} D")

    print(f"\n4. 城市/郊区比率 (Urban/Rural):")
    for val, count in school_counts.items():
        print(f"   {val}: {count} 人 ({school_pct[val]:.1f}%)")

    print(f"\n5. 矫正方式 (Correction - Has Glasses):")
    for val, count in glasses_counts.items():
        label = "配镜" if val == 1 else "未配镜"
        print(f"   数值 {val} ({label}): {count} 人 ({glasses_pct[val]:.1f}%)")

    print("-" * 40)
    print(f"追踪统计 (Tracking Stats):")
    print(f"   平均追踪时长: {mean_duration:.2f} 年")
    print(f"   平均检查次数: {mean_exams:.2f} 次")
    print(f"   检查间隔范围: {min_interval:.2f} 到 {max_interval:.2f} 个月")
    print("="*50)

if __name__ == "__main__":
    calculate_stats()
