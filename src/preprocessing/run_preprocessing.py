import os
import sys
import pandas as pd

# 将项目根目录添加到 python path，以便可以导入 src 下的模块
# 假设当前脚本位于 src/preprocessing/
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../../'))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.preprocessing.cleaner import DataCleaner
from src.preprocessing.feature_engineer import FeatureEngineer

def main():
    # 配置路径 (使用绝对路径或相对于项目根目录的路径)
    raw_data_path = os.path.join(project_root, 'data/raw/多次筛查结果v1.xlsx')
    processed_data_dir = os.path.join(project_root, 'data/processed')
    processed_file_name = 'cleaned_data.csv'
    final_feature_file = 'featured_data.csv'
    
    # 确保输出目录存在
    os.makedirs(processed_data_dir, exist_ok=True)
    
    # 初始化
    cleaner = DataCleaner()
    engineer = FeatureEngineer()
    
    # 检查原始数据是否存在
    if not os.path.exists(raw_data_path):
        print(f"错误: 未在 {raw_data_path} 找到原始数据文件")
        return
    
    # 运行处理流程
    try:
        # 1. 清洗
        df = cleaner.load_data(raw_data_path)
        cleaned_df = cleaner.preprocess(df)
        
        # 保存清洗中间结果 (可选)
        clean_output_path = os.path.join(processed_data_dir, processed_file_name)
        cleaned_df.to_csv(clean_output_path, index=False)
        print(f"基础清洗完成。数据已保存至 {clean_output_path}")
        
        # 2. 特征工程
        featured_df = engineer.transform(cleaned_df)
        
        # 保存最终特征数据
        feature_output_path = os.path.join(processed_data_dir, final_feature_file)
        featured_df.to_csv(feature_output_path, index=False)
        print(f"特征工程完成。数据已保存至 {feature_output_path}")
        
        # 展示样本
        print("\n最终数据样本:")
        # 展示关键列
        key_cols = ['student_id', 'check_date', 'age', 'delta_t', 'SE_right', 'school_type', 'has_glasses', 'correct_level']
        existing_cols = [c for c in key_cols if c in featured_df.columns]
        print(featured_df[existing_cols].head(10))
        print(f"\n数据形状: {featured_df.shape}")
        
    except Exception as e:
        print(f"预处理过程中发生错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
