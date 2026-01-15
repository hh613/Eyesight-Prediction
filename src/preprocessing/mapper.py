import pandas as pd
import numpy as np
from datetime import datetime

def calculate_se(sphere, cylinder):
    """
    计算等效球镜 (SE)。
    公式: SE = 球镜 (Sphere) + 0.5 * 柱镜 (Cylinder)
    """
    return sphere + 0.5 * cylinder

def get_birth_date_from_id(id_no):
    """
    从身份证号（18位）解析出生日期。
    """
    if pd.isna(id_no) or len(str(id_no)) != 18:
        return None
    
    try:
        id_str = str(id_no)
        year = int(id_str[6:10])
        month = int(id_str[10:12])
        day = int(id_str[12:14])
        return datetime(year, month, day)
    except:
        return None

def calculate_age(birth_date, check_date):
    """
    计算年龄（以年为单位）。
    """
    if pd.isna(birth_date) or pd.isna(check_date):
        return np.nan
    
    # 确保 check_date 是 datetime 类型
    if not isinstance(check_date, datetime):
        try:
            check_date = pd.to_datetime(check_date)
        except:
            return np.nan
            
    # 计算年龄
    age_days = (check_date - birth_date).days
    return age_days / 365.25

def map_grade_to_age(grade):
    """
    将年级字符串映射为近似年龄（如果需要）。
    """
    # 简单的映射示例，可以根据需要扩展
    grade_map = {
        '一年级': 7, '二年级': 8, '三年级': 9, '四年级': 10, '五年级': 11, '六年级': 12,
        '初一': 13, '初二': 14, '初三': 15,
        '高一': 16, '高二': 17, '高三': 18
    }
    for k, v in grade_map.items():
        if k in str(grade):
            return v
    return np.nan

def map_school_type(district_name):
    """
    映射学校类型 (城镇/农村) 基于筛查区。
    城镇: 河西区、和平区、河东区、河北区、红桥区、南开区、滨海新区
    农村: 其他
    """
    if pd.isna(district_name):
        return None
        
    urban_districts = ['河西区', '和平区', '河东区', '河北区', '红桥区', '南开区', '滨海新区']
    
    # 模糊匹配
    for dist in urban_districts:
        if dist in str(district_name):
            return 'urban'
    
    return 'rural'

def map_gender(gender_val):
    """
    映射性别为 0/1。
    通常: 男=1, 女=0 (或根据项目具体定义，这里假设 男=1, 女=0)
    如果原始数据是 '男'/'女'
    """
    if pd.isna(gender_val):
        return None
        
    s = str(gender_val).strip()
    if s == '男' or s == '1':
        return 1
    elif s == '女' or s == '0':
        return 0
    return None

def calculate_correct_level(correction_method, va_corrected, va_unaided):
    """
    计算矫正程度 (correct_level)。
    (1) 近视且矫正完全: 戴镜视力 >= 5.0
    (2) 近视且矫正不完全: 戴镜视力 < 5.0 (且戴镜)
    (3) 近视但未进行矫正: 未戴镜
    
    注意: 这里假设输入数据已经是近视人群 (SE <= 0)。
    """
    # 判断是否戴镜
    is_corrected = False
    if pd.notna(correction_method) and ('框架' in str(correction_method) or '隐形' in str(correction_method) or '塑形' in str(correction_method)):
        is_corrected = True
    elif pd.notna(va_corrected): 
        # 如果有戴镜视力数据，通常意味着戴镜，但需要小心空值或0值
        try:
            if float(va_corrected) > 0:
                is_corrected = True
        except:
            pass

    if not is_corrected:
        return 3 # 未矫正
    
    # 已戴镜，判断效果
    try:
        if pd.isna(va_corrected):
            # 戴镜但无数据，暂归为不完全? 或者未知。这里归为不完全(保守)
            return 2 
        
        val = float(va_corrected)
        if val >= 5.0:
            return 1 # 矫正完全
        else:
            return 2 # 矫正不完全
    except:
        return 2
