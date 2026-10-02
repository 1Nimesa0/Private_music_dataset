import yaml
from pathlib import Path
from dataclasses import dataclass
import math

class ConfigError(Exception):
    pass

@dataclass(frozen=True)
class Config:
    dataset: dict
    splits: dict
    quality_thresholds: dict
    segmentation: dict
    environment_constraints: dict
    genre_mapping: dict
    licensing: dict
    gtzan_overlap: dict
    download: dict
    augmentation: dict
    paths: dict
    base_dir: Path
    min_artists_per_genre: int

    @classmethod
    def load(cls, config_path: str | Path):
        path = Path(config_path).resolve()
        if not path.exists():
            raise ConfigError(f"Không tìm thấy file config tại {path}")
            
        with open(path, 'r', encoding='utf-8') as f:
            raw_cfg = yaml.safe_load(f)

        # Validate Invariants (I-5, P1-1)
        splits = raw_cfg.get('splits', {})
        total_splits = sum([v for k, v in splits.items() if isinstance(v, (int, float))])
        if not math.isclose(total_splits, 1.0):
            raise ConfigError(f"Tổng tỉ lệ các split phải bằng 1.0, hiện tại là {total_splits}")

        dataset = raw_cfg.get('dataset', {})
        if dataset.get('channels') != 1:
            raise ConfigError("Hệ thống chỉ hỗ trợ audio mono (channels: 1).")
            
        qt = raw_cfg.get('quality_thresholds', {})
        if qt.get('silence_frame_dbfs', 0) >= 0 or qt.get('min_rms_dbfs', 0) >= 0:
            raise ConfigError("Các ngưỡng dBFS phải là giá trị âm.")
            
        if len(set(dataset.get('genres', []))) != len(dataset.get('genres', [])):
            raise ConfigError("Danh sách genres bị trùng lặp.")
            
        target = dataset.get('target_songs_per_genre', 15)
        max_artist = dataset.get('max_songs_per_artist', 2)
        min_artists = math.ceil(target / max_artist)

        paths = {k: str(path.parent / v) for k, v in raw_cfg.get('paths', {}).items()}

        return cls(
            dataset=dataset,
            splits=splits,
            quality_thresholds=qt,
            segmentation=raw_cfg.get('segmentation', {}),
            environment_constraints=raw_cfg.get('environment_constraints', {}),
            genre_mapping=raw_cfg.get('genre_mapping', {}),
            licensing=raw_cfg.get('licensing', {}),
            gtzan_overlap=raw_cfg.get('gtzan_overlap', {}),
            download=raw_cfg.get('download', {}),
            augmentation=raw_cfg.get('augmentation', {}),
            paths=paths,
            base_dir=path.parent,
            min_artists_per_genre=min_artists
        )

    def ensure_dirs(self):
        for key, p in self.paths.items():
            Path(p).mkdir(parents=True, exist_ok=True)