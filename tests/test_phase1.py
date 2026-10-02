import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from src.config import Config, ConfigError
from src.metadata import normalize_text, make_artist_id, env_origin, validate_schema
from src.genre_mapping import GenreMapper
from src.split_dataset import split_songs, SplitError, check_leakage, LeakageError
from src.segmentation import check_segment_quality, extract_segments, SegmentationError
from src.local_source import load_local_metadata

# Mock cấu hình hợp lệ dựa trên Phụ lục A với Wrapper tương thích ngược
@pytest.fixture
def base_config():
    from src.config import Config
    from pathlib import Path
    
    real_config = Config(
        dataset={
            'genres': ['blues', 'pop'], 
            'target_songs_per_genre': 15, 
            'segments_per_song': 2,
            'max_songs_per_artist': 2, 
            'sample_rate': 22050, 
            'channels': 1, 
            'segment_duration_sec': 10.0,
            'random_seed': 42
        },
        splits={
            'strategy': 'artist_disjoint', 
            'adaptation': 0.6,
            'validation': 0.1, 
            'test': 0.2, 
            'demo': 0.1
        },
        quality_thresholds={
            'max_silence_ratio': 0.3, 
            'max_clipping_ratio': 0.01, 
            'min_rms_dbfs': -45,
            'min_duration_tolerance_sec': 0.5, 
            'silence_frame_dbfs': -50.0
        },
        segmentation={
            'edge_margin_ratio': 0.1, 
            'offset_candidates': [0.5, 0.25, 0.75, 0.0, 1.0]
        },
        environment_constraints={
            'min_environment_types': 3, 
            'min_non_clean_ratio': 0.5, 
            'min_test_real_samples': 50
        },
        genre_mapping={'min_genre_confidence': 1.0},
        licensing={'allowed_cc': ['by', 'by-sa'], 'allow_nd': False},
        gtzan_overlap={
            'ber_threshold': 0.35, 
            'metadata_match_threshold': 92, 
            'metadata_review_threshold': 75,
            'min_reference_files': 1000
        },
        download={
            'oversample_candidates': 50, 
            'max_retries': 5,
            'backoff_base_sec': 1.0, 
            'min_interval_sec': 0.5, 
            'timeout_sec': 30
        },
        augmentation={'max_variants_per_segment': 1, 'effects': {}},
        paths={'output': str(Path('out'))},
        base_dir=Path('.'),
        min_artists_per_genre=8
    )

    class ConfigWrapper:
        """Wrapper tự động ánh xạ các thuộc tính phẳng (legacy) vào nested dict (mới)."""
        def __init__(self, cfg):
            self._cfg = cfg
            
        def __getattr__(self, name):
            # Duyệt qua các dictionary con để tìm thuộc tính nếu mã nguồn cũ yêu cầu flat access
            groups = ['dataset', 'splits', 'quality_thresholds', 'segmentation', 
                      'environment_constraints', 'genre_mapping', 'licensing', 
                      'gtzan_overlap', 'download', 'augmentation', 'paths']
            for group_name in groups:
                group = getattr(self._cfg, group_name, {})
                if isinstance(group, dict) and name in group:
                    return group[name]
            return getattr(self._cfg, name)

    return ConfigWrapper(real_config)

def test_metadata_normalization():
    assert normalize_text("Beyoncé feat. X") == "beyonce x"
    assert make_artist_id("Artist") == make_artist_id("artist")
    with pytest.raises(ValueError):
         make_artist_id("?!")

def test_genre_mapping():
    mapper = GenreMapper(1.0)
    res = mapper.evaluate(["Hip-Hop"])
    assert "hiphop" in res.matched_genres
    res2 = mapper.evaluate(["rock", "pop"])
    assert res2.confidence == 0.5 and not res2.accepted

def test_split_leakage(base_config):
    groups = [f"artist_{i}" for i in range(20)]
    ratios = {'adaptation': 0.6, 'validation': 0.1, 'test': 0.2, 'demo': 0.1}
    splits = split_songs(groups, ratios, 42)
    assert len(splits) == 20
    
    # Kiểm tra LeakageError
    df = pd.DataFrame([
        {'song_id': 's1', 'artist_id': 'a1', 'split': 'test', 'parent_segment_id': '', 'segment_id': 'seg1'},
        {'song_id': 's2', 'artist_id': 'a1', 'split': 'adaptation', 'parent_segment_id': '', 'segment_id': 'seg2'}
    ])
    with pytest.raises(LeakageError):
        check_leakage(df, 'artist_disjoint')

def test_segmentation_logic(base_config):
    sr = 22050
    audio_short = np.random.randn(10 * sr)
    with pytest.raises(SegmentationError):
        extract_segments(audio_short, sr, base_config)

    audio_ok = np.random.randn(30 * sr) * 0.1 # RMS an toàn
    segs = extract_segments(audio_ok, sr, base_config)
    assert len(segs) == 2
    assert segs[0][2] - segs[0][1] == int(10.0 * sr)

def test_local_source_quarantine(base_config, tmp_path):
    csv = tmp_path / "meta.csv"
    csv.write_text("file_name,artist,usage_rights_confirmed_by_user,usage_rights_note,genre\n1.wav,A,False,,pop")
    res = load_local_metadata(csv, tmp_path, base_config)
    assert len(res.quarantined) == 1
    assert "Rights not confirmed" in res.quarantined.iloc[0]['reason']