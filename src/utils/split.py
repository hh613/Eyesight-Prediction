import pandas as pd
import numpy as np
import os
import logging
from typing import List, Tuple, Dict

logger = logging.getLogger(__name__)

def make_person_splits(df: pd.DataFrame, person_id_col: str, ratios: List[float], seed: int) -> Dict[str, List[str]]:
    """
    按 person_id 将数据集划分为 train/val/test。
    
    参数:
        df: 包含 person_id_col 的 DataFrame
        person_id_col: 学生 ID 列名
        ratios: [train_ratio, val_ratio, test_ratio]，和应为 1.0 (例如 [0.7, 0.1, 0.2])
        seed: 随机种子
        
    返回:
        字典 {'train': [ids...], 'val': [ids...], 'test': [ids...]}
    """
    # 验证比例
    if not np.isclose(sum(ratios), 1.0):
        raise ValueError(f"划分比例之和必须为 1.0，当前为: {sum(ratios)}")
        
    pids = df[person_id_col].unique()
    n_total = len(pids)
    
    np.random.seed(seed)
    np.random.shuffle(pids)
    
    n_test = int(n_total * ratios[2])
    n_val = int(n_total * ratios[1])
    n_train = n_total - n_test - n_val
    
    train_ids = pids[:n_train].tolist()
    val_ids = pids[n_train:n_train+n_val].tolist()
    test_ids = pids[n_train+n_val:].tolist()
    
    logger.info(f"划分完成: Total={n_total}, Train={len(train_ids)}, Val={len(val_ids)}, Test={len(test_ids)}")
    
    return {
        'train': train_ids,
        'val': val_ids,
        'test': test_ids
    }

def check_no_leakage(train_ids: List[str], val_ids: List[str], test_ids: List[str]):
    """
    检查数据泄漏：保证 train/val/test 的 ID 集合互斥。
    """
    s_train = set(train_ids)
    s_val = set(val_ids)
    s_test = set(test_ids)
    
    leak_tv = s_train.intersection(s_val)
    leak_tt = s_train.intersection(s_test)
    leak_vt = s_val.intersection(s_test)
    
    if leak_tv or leak_tt or leak_vt:
        msg = "检测到数据泄漏！\n"
        if leak_tv: msg += f"Train-Val 重叠: {len(leak_tv)} 个\n"
        if leak_tt: msg += f"Train-Test 重叠: {len(leak_tt)} 个\n"
        if leak_vt: msg += f"Val-Test 重叠: {len(leak_vt)} 个\n"
        raise ValueError(msg)
        
    logger.info("防泄漏检查通过：所有集合互斥。")

def save_splits(split_dict: Dict[str, List[str]], output_dir: str):
    """
    保存 ID 列表到文本文件。
    """
    os.makedirs(output_dir, exist_ok=True)
    
    for split_name, ids in split_dict.items():
        file_path = os.path.join(output_dir, f"{split_name}_ids.txt")
        with open(file_path, 'w') as f:
            for pid in ids:
                f.write(f"{str(pid)}\n")
        logger.info(f"已保存 {split_name} ID 列表到 {file_path}")
