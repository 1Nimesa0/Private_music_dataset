import os
import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch
from src.environment_augmentation import apply_simulated_noise, augment_dataset

# Kế thừa ConfigWrapper từ Phase 2 để tránh lỗi lồng dictionary
from tests.test_phase2 import mock_config 

def test_noise_colors(mock_config):
    silent_audio = np.zeros(44100)
    rng = np.random.default_rng(42)
    params = {'snr_db': [5, 20], 'color': ['white', 'pink', 'brown']}
    
    # Audio im lặng phải trả về im lặng (B-16)
    out_audio = apply_simulated_noise(silent_audio, 22050, rng, params)
    assert np.all(out_audio == 0)
    
def test_augmentation_quotas_and_splits(mock_config, tmp_path):
    df = pd.DataFrame([
        {'song_id': 's1', 'segment_id': 'seg1', 'split': 'test', 'parent_segment_id': None, 'file_path': 'dummy.wav'},
        {'song_id': 's2', 'segment_id': 'seg2', 'split': 'adaptation', 'parent_segment_id': None, 'file_path': 'dummy.wav'}
    ])
    
    with patch('src.environment_augmentation.load_audio_mono') as mock_load:
        mock_load.return_value = (np.random.randn(22050), 22050)
        with patch('src.environment_augmentation.sf.write'):
            new_df = augment_dataset(df, mock_config, str(tmp_path))
            
    # Test phải có đúng 1 test_shifted
    test_shifted = new_df[(new_df['split'] == 'test') & (~new_df['parent_segment_id'].isna())]
    assert len(test_shifted) == 1
    
    # B-14: Biến thể phải kế thừa đúng split
    for _, row in new_df[~new_df['parent_segment_id'].isna()].iterrows():
        parent_split = new_df[new_df['segment_id'] == row['parent_segment_id']].iloc[0]['split']
        assert row['split'] == parent_split

@patch('sys.argv', ['main.py', '--stage', 'all'])
def test_main_exit_code():
    from main import main
    with patch('main.validate_dataset') as mock_val:
        mock_val.return_value.status = 'DATASET NOT READY'
        mock_val.return_value.reasons = ['Thiếu test_real']
        with pytest.raises(SystemExit) as exc:
            main()
        # Không crash khi dataset NOT READY, trả về code 0
        assert exc.value.code == 0