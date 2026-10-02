import pytest
import os
import json
from pathlib import Path
from unittest.mock import patch, MagicMock
from src.gtzan_overlap import GTZANOverlapDetector
from src.config import Config

@pytest.fixture
def mock_config(tmp_path):
    paths = {
        'gtzan_reference': tmp_path / 'genres_original',
        'interim': tmp_path / 'interim'
    }
    paths['gtzan_reference'].mkdir(parents=True)
    paths['interim'].mkdir(parents=True)
    
    # Tạo đệ quy 10 file giả lập kèm metadata hệ thống để kiểm tra Cache
    for i in range(10):
        f = paths['gtzan_reference'] / f"blues/fake_{i}.wav"
        f.parent.mkdir(exist_ok=True, parents=True)
        f.write_text("dummy content")
    
    real_config = Config(
        dataset={
            'genres': [], 
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
            'offset_candidates': []
        },
        environment_constraints={
            'min_environment_types': 3, 
            'min_non_clean_ratio': 0.5, 
            'min_test_real_samples': 50
        },
        genre_mapping={'min_genre_confidence': 1.0},
        licensing={'allowed_cc': [], 'allow_nd': False},
        gtzan_overlap={
            'ber_threshold': 0.35, 
            'metadata_match_threshold': 92, 
            'metadata_review_threshold': 75,
            'min_reference_files': 5
        },
        download={
            'oversample_candidates': 50, 
            'max_retries': 5,
            'backoff_base_sec': 1.0, 
            'min_interval_sec': 0.5, 
            'timeout_sec': 30
        },
        augmentation={'max_variants_per_segment': 1, 'effects': {}},
        paths={k: str(v) for k, v in paths.items()}, 
        base_dir=tmp_path,
        min_artists_per_genre=8
    )

    class ConfigWrapper:
        """Wrapper tự động ánh xạ các thuộc tính phẳng, đồng thời ép kiểu Path cho đường dẫn"""
        def __init__(self, cfg):
            self._cfg = cfg
            
        @property
        def paths(self):
            # Tự động cast tất cả các value trong dictionary paths thành pathlib.Path
            return {k: Path(v) for k, v in self._cfg.paths.items()}

        def __getattr__(self, name):
            groups = ['dataset', 'splits', 'quality_thresholds', 'segmentation', 
                      'environment_constraints', 'genre_mapping', 'licensing', 
                      'gtzan_overlap', 'download', 'augmentation', 'paths']
            for group_name in groups:
                group = getattr(self._cfg, group_name, {})
                if isinstance(group, dict) and name in group:
                    val = group[name]
                    # Nếu thuộc tính thuộc group paths, trả về đối tượng Path
                    if group_name == 'paths':
                        return Path(val)
                    return val
            return getattr(self._cfg, name)

    return ConfigWrapper(real_config)

def test_ber_calculation(mock_config):
    detector = GTZANOverlapDetector(mock_config)
    
    # 1. Cùng tín hiệu => BER ≈ 0
    fp1 = [12345, 67890]
    assert detector.calculate_ber(fp1, fp1) == 0.0
    
    # 2. Tín hiệu độc lập hoàn toàn (nghịch đảo bit) => BER = 1.0
    fp_a = [int('10101010' * 4, 2)]
    fp_b = [int('01010101' * 4, 2)]
    assert detector.calculate_ber(fp_a, fp_b) == 1.0

@patch('src.gtzan_overlap.subprocess.run')
def test_cache_usage(mock_run, mock_config, tmp_path):
    """Bản vá lỗi: Cache chỉ được xác nhận khi file vật lý trùng khớp size/mtime"""
    
    # Lần 1: Cho phép mọi lệnh (cả -version và -raw) chạy bình thường
    def mock_subprocess_run_1(cmd, *args, **kwargs):
        if '-raw' in cmd:
            return MagicMock(returncode=0, stdout=json.dumps({"fingerprint": [1, 2, 3]}))
        return MagicMock(returncode=0, stdout="fpcalc version 1.5.0")
        
    mock_run.side_effect = mock_subprocess_run_1
    
    detector = GTZANOverlapDetector(mock_config)
    
    # Chạy lần 1: Sẽ gọi fpcalc 10 lần và ghi Cache
    detector.load_references()
    assert len(detector.reference_fps) == 10
    
    cache_file = mock_config.paths['interim'] / 'gtzan_fingerprints_cache.json'
    assert cache_file.exists()
    
    # Lần 2: Ném lỗi NẾU gọi -raw (bắt buộc phải lấy từ cache), nhưng -version thì vẫn cho phép
    def mock_subprocess_run_2(cmd, *args, **kwargs):
        if '-raw' in cmd:
            raise Exception("Không được gọi lại fpcalc (không tối ưu)!")
        return MagicMock(returncode=0, stdout="fpcalc version 1.5.0")
        
    mock_run.side_effect = mock_subprocess_run_2
    
    detector2 = GTZANOverlapDetector(mock_config)
    detector2.load_references() # Lấy từ cache
    
    assert len(detector2.reference_fps) == 10

@patch('src.gtzan_overlap.GTZANOverlapDetector._get_raw_fingerprint')
def test_decision_matrix(mock_get_fp, mock_config, tmp_path):
    detector = GTZANOverlapDetector(mock_config)
    detector.has_fpcalc = True
    
    detector.is_reference_complete = True
    
    # Sử dụng chuỗi số nguyên lớn phủ kín 32 bit để tránh bị trùng khớp (leading zeros)
    ref_fp = [int('10101010'*4, 2)] * 10
    match_fp = [int('10101010'*4, 2)] * 10
    diff_fp = [int('01010101'*4, 2)] * 10
    
    detector.reference_fps = {"fake_ref.wav": {"fp": ref_fp}}
    detector.metadata_db = ["beatles let it be"]
    
    cand_audio = tmp_path / "cand.wav"
    cand_audio.write_text("d")
    
    # Ô 1: Fingerprint khớp -> true
    mock_get_fp.return_value = match_fp
    res = detector.check_overlap(str(cand_audio), "unknown", "song")
    assert res == ("true", "fingerprint", False)
    
    # Ô 2: Metadata khớp (Fingerprint khác biệt) -> true
    mock_get_fp.return_value = diff_fp
    res = detector.check_overlap(str(cand_audio), "Beatles", "Let It Be")
    assert res == ("true", "metadata", False)
    
    # Ô 3: Không khớp, tham chiếu đầy đủ -> false
    res = detector.check_overlap(str(cand_audio), "Queen", "Bohemian Rhapsody")
    assert res == ("false", "fingerprint", False)
    
    # Ô 4: Không khớp, tham chiếu thiếu (không đầy đủ) -> unknown + review_required
    detector.is_reference_complete = False
    res = detector.check_overlap(str(cand_audio), "Queen", "Bohemian Rhapsody")
    assert res == ("unknown", "none", True)

def test_metadata_no_token_set_ratio(mock_config):
    detector = GTZANOverlapDetector(mock_config)
    detector.metadata_db = ["elvis presley love me tender"]
    detector.is_reference_complete = False
    
    # "Love" vs "Love Me Tender": fuzz.ratio cho điểm thấp (~35), token_set_ratio sẽ cho 100
    res = detector.check_overlap("cand.wav", "elvis presley", "love")
    
    # Vì điểm thấp, nó rơi về nhánh không khớp (ở reference thiếu => unknown)
    assert res == ("unknown", "none", True)