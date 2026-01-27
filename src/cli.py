import sys
import argparse
import logging
import os
import pandas as pd
from src.core.config import Config
# 其他模块将在实现后导入

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="眼科视力预测实验 CLI")
    subparsers = parser.add_subparsers(dest="command", help="要运行的命令")

    # 命令: data.build
    build_parser = subparsers.add_parser("data.build", help="构建用于训练的转换对数据")
    build_parser.add_argument("--config", type=str, required=True, help="config.yaml 的路径")
    build_parser.add_argument("--split_dir", type=str, required=False, help="包含 split ID 文件的目录 (默认: config.data_dir/splits)")
    build_parser.add_argument("--raw_input", type=str, required=False, help="原始清洗后的数据路径 (默认: config.data_dir/featured_data.csv)")
    build_parser.add_argument("--output_dir", type=str, required=False, help="输出训练对的目录 (默认: config.data_dir)")

    # 命令: data.split
    split_parser = subparsers.add_parser("data.split", help="将原始数据划分为 train/val/test ID")
    split_parser.add_argument("--config", type=str, required=True, help="config.yaml 的路径")
    split_parser.add_argument("--input", type=str, required=False, help="原始清洗后的数据路径 (默认: config.data_dir/featured_data.csv)")
    split_parser.add_argument("--output_dir", type=str, required=False, help="输出 split ID 和数据子集的目录 (默认: config.data_dir)")
    
    # 命令: exp.run
    run_parser = subparsers.add_parser("exp.run", help="运行完整实验 (训练 -> 评估)")
    run_parser.add_argument("--config", type=str, required=True, help="config.yaml 的路径")
    run_parser.add_argument("--data_dir", type=str, required=False, help="包含划分后数据的目录 (默认: config.data_dir)")
    run_parser.add_argument("--model", type=str, required=False, default=None, choices=["tem", "arima", "lstm", "lstm_multi", "tabpfn"], help="选择模型: 'tem', 'arima', 'lstm', 'lstm_multi', 'tabpfn'")
    run_parser.add_argument("--output_dir", type=str, required=False, help="覆盖默认输出目录")
    run_parser.add_argument("--tag", type=str, required=False, help="实验标签，将附加到输出目录名")

    args = parser.parse_args()

    if args.command == "data.build":
        run_data_build(args)
    elif args.command == "data.split":
        run_data_split(args)
    elif args.command == "exp.run":
        run_experiment(args)
    else:
        parser.print_help()

def run_data_split(args):
    from src.utils.split import make_person_splits, check_no_leakage, save_splits
    
    config = Config.load(args.config)
    
    # Resolve paths
    input_path = args.input if args.input else os.path.join(config.experiment.data_dir, 'featured_data.csv')
    output_dir = args.output_dir if args.output_dir else config.experiment.data_dir
    
    logger.info(f"正在划分数据: {input_path}")
    
    # Load raw data
    if input_path.endswith('.parquet'):
        df = pd.read_parquet(input_path)
    else:
        df = pd.read_csv(input_path)
        
    pid_col = config.columns.person_id_col
    ratios = [1.0 - config.experiment.test_size - config.experiment.val_size, 
              config.experiment.val_size, 
              config.experiment.test_size]
              
    # 1. 生成 ID 列表
    splits = make_person_splits(df, pid_col, ratios, seed=config.experiment.random_seed)
    
    # 2. 检查防泄漏
    check_no_leakage(splits['train'], splits['val'], splits['test'])
    
    # 3. 保存 ID 列表
    split_dir = os.path.join(output_dir, 'splits')
    save_splits(splits, split_dir)
    
    # 4. 保存原始数据子集 (Raw Subsets)
    # 这些子集是 Raw Cleaned Data 的切片，用于后续的 Build (Train) 或 Baseline (Test)
    
    for split_name, ids in splits.items():
        subset_df = df[df[pid_col].isin(ids)]
        
        # 统一保存为 CSV (或根据配置，这里暂时用 CSV)
        out_name = f"{split_name}_raw.csv"
        out_path = os.path.join(output_dir, out_name)
        
        subset_df.to_csv(out_path, index=False)
        logger.info(f"保存 {split_name} 原始子集 ({len(subset_df)} 行) 到 {out_path}")

def run_data_build(args):
    from src.core.data import DataBuilder
    config = Config.load(args.config)
    
    # Resolve paths
    split_dir = args.split_dir if args.split_dir else os.path.join(config.experiment.data_dir, 'splits')
    raw_input = args.raw_input if args.raw_input else os.path.join(config.experiment.data_dir, 'featured_data.csv')
    output_dir = args.output_dir if args.output_dir else config.experiment.data_dir
    
    # 只需要为 Train 集构建转换对用于 TEM 训练
    # Val/Test 集通常在评估时动态处理，或者如果需要 Val Loss，也可以构建 Val Pairs
    
    logger.info(f"正在构建训练对，基于 Split: {split_dir}")
    
    # 加载 Train ID
    train_id_path = os.path.join(split_dir, 'train_ids.txt')
    if not os.path.exists(train_id_path):
        raise FileNotFoundError(f"找不到 {train_id_path}")
        
    with open(train_id_path, 'r') as f:
        train_ids = [line.strip() for line in f if line.strip()]
        
    # 加载原始数据并过滤
    if raw_input.endswith('.parquet'):
        df = pd.read_parquet(raw_input)
    else:
        df = pd.read_csv(raw_input)
        
    # 过滤 Train 数据
    train_df = df[df[config.columns.person_id_col].astype(str).isin(train_ids)]
    
    # 构建转换对
    builder = DataBuilder(config)
    
    # 保存 Train Pairs
    out_path = os.path.join(output_dir, 'train_pairs.csv')
    # 临时保存 train_raw 以供 builder 使用 (builder 接口目前接受路径)
    # 为了避免写临时文件，我们可以重构 builder.build_and_save 接受 DataFrame
    # 但为了最小修改，我们先保存一个临时文件
    
    temp_train_path = os.path.join(output_dir, 'temp_train_raw.csv')
    train_df.to_csv(temp_train_path, index=False)
    
    logger.info("正在生成 Train 转换对...")
    builder.build_and_save(temp_train_path, out_path)
    
    # 清理
    if os.path.exists(temp_train_path):
        os.remove(temp_train_path)
        
    # 也可以选择构建 Val Pairs
    # ... (可选)


def run_experiment(args):
    config = Config.load(args.config)
    
    # Resolve paths
    data_dir = args.data_dir if args.data_dir else config.experiment.data_dir
    model_name = args.model if args.model else config.experiment.default_model
    
    
    config.experiment.output_dir = os.path.join(config.experiment.output_dir, args.tag) if args.tag else config.experiment.output_dir
    
    logger.info(f"正在运行实验，配置: {args.config}, 模型: {model_name}, 输出目录: {config.experiment.output_dir}")
    
    if model_name == "tem":
        from src.experiment.runner import ExperimentRunner
        runner = ExperimentRunner(config, data_dir)
        runner.run()
    elif model_name == "arima":
        from src.models.baselines.arima_univariate import ArimaUnivariateBaseline
        
        # 加载测试数据 (Raw Test Data)
        test_path = os.path.join(data_dir, 'test_raw.csv')
        if not os.path.exists(test_path):
            test_path = os.path.join(data_dir, 'test_raw.parquet')
            if not os.path.exists(test_path):
                raise FileNotFoundError(f"未找到测试数据 test_raw.csv 或 test_raw.parquet 在 {data_dir}")
                
        if test_path.endswith('.parquet'):
            test_df = pd.read_parquet(test_path)
        else:
            test_df = pd.read_csv(test_path)
            test_df[config.columns.time_col] = pd.to_datetime(test_df[config.columns.time_col])
            
        # 设置输出目录
        out_dir = os.path.join(config.experiment.output_dir, 'baseline_arima')
        
        baseline = ArimaUnivariateBaseline(config)
        baseline.run(test_df, out_dir)
        
    elif model_name == "lstm":
        from src.models.baselines.lstm_univariate import LSTMUnivariateBaseline
        
        # LSTM 需要 Train Raw 进行训练，以及 Test Raw 进行评估
        train_path = os.path.join(data_dir, 'train_raw.csv')
        if not os.path.exists(train_path):
             train_path = os.path.join(data_dir, 'train_raw.parquet')
             if not os.path.exists(train_path):
                 raise FileNotFoundError(f"未找到训练数据 train_raw.csv/parquet 在 {data_dir}")

        test_path = os.path.join(data_dir, 'test_raw.csv')
        if not os.path.exists(test_path):
            test_path = os.path.join(data_dir, 'test_raw.parquet')
            if not os.path.exists(test_path):
                raise FileNotFoundError(f"未找到测试数据 test_raw.csv/parquet 在 {data_dir}")
                
        # 读取数据
        if test_path.endswith('.parquet'):
            test_df = pd.read_parquet(test_path)
        else:
            test_df = pd.read_csv(test_path)
            test_df[config.columns.time_col] = pd.to_datetime(test_df[config.columns.time_col])
            
        out_dir = os.path.join(config.experiment.output_dir, 'baseline_lstm')
        
        lstm_baseline = LSTMUnivariateBaseline(config)
        # 1. 训练
        lstm_baseline.fit(train_path)
         # 2. 评估
        lstm_baseline.run(test_df, out_dir)
         
    elif model_name == "lstm_multi":
        from src.models.baselines.lstm_multivariate import LSTMMultivariateBaseline
        
        # 路径逻辑同 LSTM
        train_path = os.path.join(data_dir, 'train_raw.csv')
        if not os.path.exists(train_path):
             train_path = os.path.join(data_dir, 'train_raw.parquet')
             if not os.path.exists(train_path):
                 raise FileNotFoundError(f"未找到训练数据 train_raw.csv/parquet")
                 
        test_path = os.path.join(data_dir, 'test_raw.csv')
        if not os.path.exists(test_path):
            test_path = os.path.join(data_dir, 'test_raw.parquet')
            if not os.path.exists(test_path):
                raise FileNotFoundError(f"未找到测试数据 test_raw.csv/parquet")
                
        if test_path.endswith('.parquet'):
            test_df = pd.read_parquet(test_path)
        else:
            test_df = pd.read_csv(test_path)
            test_df[config.columns.time_col] = pd.to_datetime(test_df[config.columns.time_col])
            
        out_dir = os.path.join(config.experiment.output_dir, 'baseline_lstm_multi')
        
        baseline = LSTMMultivariateBaseline(config)
        baseline.fit(train_path)
        baseline.run(test_df, out_dir)
         
    elif model_name == "tabpfn":
        from src.models.baselines.tabpfn_transition import TabPFNTransitionBaseline
        
        # 1. 训练需要转换对 (train_pairs.csv)
        train_pairs_path = os.path.join(data_dir, 'train_pairs.csv')
        if not os.path.exists(train_pairs_path):
             train_pairs_path = os.path.join(data_dir, 'train_pairs.parquet')
             if not os.path.exists(train_pairs_path):
                 raise FileNotFoundError(f"未找到训练转换对 train_pairs.csv/parquet")
                 
        # 2. 评估需要原始测试序列 (test_raw.csv)
        test_raw_path = os.path.join(data_dir, 'test_raw.csv')
        if not os.path.exists(test_raw_path):
            test_raw_path = os.path.join(data_dir, 'test_raw.parquet')
            if not os.path.exists(test_raw_path):
                raise FileNotFoundError(f"未找到测试数据 test_raw.csv/parquet")
                
        out_dir = os.path.join(config.experiment.output_dir, 'baseline_tabpfn')
        
        baseline = TabPFNTransitionBaseline(config)
        baseline.fit(train_pairs_path)
        baseline.run(test_raw_path, out_dir)
        
    else:
        raise ValueError(f"未知模型: {model_name}")

if __name__ == "__main__":
    main()
