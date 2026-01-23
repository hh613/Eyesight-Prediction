import yaml
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any

@dataclass
class ColumnConfig:
    person_id_col: str = "student_id"
    time_col: str = "check_date"  # 应可被解析为 datetime
    target_col: str = "SE_right"
    static_cols: List[str] = field(default_factory=lambda: ["gender", "school_type", "has_glasses", "age","C_cyl_right"])
    timevarying_cols: List[str] = field(default_factory=lambda: ["SE_right", "VA_unaided", "correct_level"])
    # 这些是纯衍生列，不应直接进行 ARIMA 预测（或在内部处理）
    # 但在递归预测中，我们将 timevarying_cols 视为需要 ARIMA 预测的列。

@dataclass
class FeatureConfig:
    window_size: int = 3
    use_delta_t: bool = True
    
@dataclass
class ArimaConfig:
    order: tuple = (1, 1, 0)
    use_grid_search: bool = False
    grid_search_params: Optional[Dict[str, List[int]]] = None
    fallback_strategy: str = "linear_trend" # 'linear_trend' (线性趋势), 'last_value' (最后值)

@dataclass
class ModelConfig:
    model_type: str = "xgboost" # 'xgboost', 'lightgbm', 'random_forest'
    model_params: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ExperimentConfig:
    test_size: float = 0.2
    val_size: float = 0.1
    random_seed: int = 42
    forecast_horizon: int = 3
    accuracy_thresholds: List[float] = field(default_factory=lambda: [0.50, 0.75])
    output_dir: str = "experiments/output"

@dataclass
class Config:
    columns: ColumnConfig = field(default_factory=ColumnConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    arima: ArimaConfig = field(default_factory=ArimaConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)

    @classmethod
    def load(cls, path: str) -> 'Config':
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        
        # 递归实例化 dataclass 的辅助方法
        # 这是一个简化的加载器。生产环境建议使用 dacite 或 pydantic。
        cols = ColumnConfig(**data.get('columns', {}))
        feats = FeatureConfig(**data.get('features', {}))
        arima = ArimaConfig(**data.get('arima', {}))
        model = ModelConfig(**data.get('model', {}))
        exp = ExperimentConfig(**data.get('experiment', {}))
        
        return cls(columns=cols, features=feats, arima=arima, model=model, experiment=exp)
