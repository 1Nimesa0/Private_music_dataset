import yaml
from pathlib import Path
from dataclasses import dataclass

@dataclass
class QualityThresholds:
    max_silence_ratio: float
    max_clipping_ratio: float
    min_rms_dbfs: float
    min_duration_tolerance_sec: float

@dataclass
class EnvConstraints:
    min_environment_types: int
    min_non_clean_ratio: float

class Config:
    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path, 'r', encoding='utf-8') as f:
            self._cfg = yaml.safe_load(f)
            
        self.quality = QualityThresholds(**self._cfg['quality_thresholds'])
        self.env_constraints = EnvConstraints(**self._cfg['environment_constraints'])
        
        # Initialize paths
        for path_name, path_val in self._cfg['paths'].items():
            Path(path_val).mkdir(parents=True, exist_ok=True)
            
    def get_path(self, path_name: str) -> Path:
        return Path(self._cfg['paths'].get(path_name, "./"))

# Usage: config = Config()