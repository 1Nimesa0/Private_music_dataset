import os
import json
import hashlib
import numpy as np
import scipy.signal
import soundfile as sf
import pandas as pd
from typing import List, Tuple
from src.preprocess_audio import load_audio_mono
from src.segmentation import check_segment_quality, SegmentationError
from src.metadata import make_segment_id

class AugmentationError(Exception):
    pass

def _get_rng(seed: int, segment_id: str) -> np.random.Generator:
    """I-6: Sinh RNG ổn định từ seed và id."""
    hash_val = int(hashlib.sha256(f"{seed}_{segment_id}".encode()).hexdigest(), 16)
    return np.random.default_rng(hash_val % (2**32))

def apply_simulated_reverb(audio: np.ndarray, sr: int, rng: np.random.Generator, params: dict) -> np.ndarray:
    rt60 = rng.uniform(params['rt60_sec'][0], params['rt60_sec'][1])
    wet_mix = rng.uniform(params['wet_mix'][0], params['wet_mix'][1])
    
    t = np.arange(0, int(sr * rt60)) / sr
    impulse_response = np.exp(-t * (6.91 / rt60)) * rng.standard_normal(len(t))
    
    wet = scipy.signal.fftconvolve(audio, impulse_response, mode='full')[:len(audio)]
    max_wet = np.max(np.abs(wet))
    if max_wet > 0:
        wet = wet / max_wet
    else:
        return audio
        
    return (1 - wet_mix) * audio + wet_mix * wet

def _generate_noise(length: int, color: str, rng: np.random.Generator) -> np.ndarray:
    white = rng.standard_normal(length)
    if color == 'white':
        return white
    elif color == 'brown':
        return np.cumsum(white)
    elif color == 'pink':
        # Pink noise approximation
        b = [0.049922035, -0.095993537, 0.050612699, -0.004408786]
        a = [1, -2.494956002, 2.017265875, -0.522189400]
        return scipy.signal.lfilter(b, a, white)
    return white

def apply_simulated_noise(audio: np.ndarray, sr: int, rng: np.random.Generator, params: dict) -> np.ndarray:
    snr_db = rng.uniform(params['snr_db'][0], params['snr_db'][1])
    color = rng.choice(params['color'])
    
    signal_power = np.mean(audio**2)
    if signal_power == 0:
        return audio
        
    noise = _generate_noise(len(audio), color, rng)
    noise_power = np.mean(noise**2)
    
    if noise_power > 0:
        target_noise_power = signal_power / (10**(snr_db / 10))
        noise = noise * np.sqrt(target_noise_power / noise_power)
        
    return audio + noise

def apply_simulated_bandlimited_mic(audio: np.ndarray, sr: int, rng: np.random.Generator, params: dict) -> np.ndarray:
    hp = rng.uniform(params['highpass_hz'][0], params['highpass_hz'][1])
    lp = rng.uniform(params['lowpass_hz'][0], params['lowpass_hz'][1])
    b, a = scipy.signal.butter(4, [hp, lp], btype='bandpass', fs=sr)
    return scipy.signal.lfilter(b, a, audio)

def augment_dataset(df: pd.DataFrame, cfg, out_dir: str) -> pd.DataFrame:
    os.makedirs(out_dir, exist_ok=True)
    aug_cfg = cfg.augmentation
    
    clean_mask = df['parent_segment_id'].isna() | (df['parent_segment_id'] == "")
    df_clean = df[clean_mask]
    
    # Tính toán hạn ngạch D-4
    df_adv = df_clean[df_clean['split'].isin(['adaptation', 'validation', 'demo'])]
    total_clean_adv = len(df_adv)
    min_ratio = cfg.environment_constraints['min_non_clean_ratio']
    
    # N / (total_clean_adv + N) >= min_ratio => N >= (min_ratio * total_clean_adv) / (1 - min_ratio)
    required_variants = 0
    if min_ratio < 1.0:
        required_variants = int(np.ceil((min_ratio * total_clean_adv) / (1 - min_ratio)))
    else:
        required_variants = total_clean_adv * aug_cfg['max_variants_per_segment']
        
    # Tính toán số lượng hiệu ứng để cân bằng
    effects_pool = [
        ('simulated_reverb', apply_simulated_reverb, aug_cfg['effects']['simulated_reverb']),
        ('simulated_noise', apply_simulated_noise, aug_cfg['effects']['simulated_noise']),
        ('simulated_bandlimited_mic', apply_simulated_bandlimited_mic, aug_cfg['effects']['simulated_bandlimited_mic'])
    ]
    
    new_rows = []
    variants_created = 0
    
    for idx, row in df_clean.iterrows():
        rng = _get_rng(cfg.dataset['random_seed'], row['segment_id'])
        
        # Test split: đúng 1 biến thể (test_shifted)
        if row['split'] == 'test':
            effects_to_apply = [rng.choice(effects_pool)]
        else:
            if variants_created < required_variants:
                effects_to_apply = [effects_pool[variants_created % len(effects_pool)]]
                variants_created += 1
            else:
                continue
                
        audio, sr = load_audio_mono(row['file_path'], cfg.dataset['sample_rate'])
        if np.max(np.abs(audio)) == 0:
            continue
            
        for effect_name, effect_func, params in effects_to_apply:
            try:
                aug_audio = effect_func(audio, sr, rng, params)
                
                if np.max(np.abs(aug_audio)) > 1.0:
                    aug_audio = aug_audio / np.max(np.abs(aug_audio))
                
                # Check QC lại sau augment
                check_segment_quality(aug_audio, sr, cfg)
                
                new_seg_id = make_segment_id(row['song_id'], int(row['start_time']), f"aug_{effect_name}")
                out_path = os.path.join(out_dir, f"{new_seg_id}.wav")
                sf.write(out_path, aug_audio, sr, subtype='PCM_16')
                
                new_row = row.copy()
                new_row['segment_id'] = new_seg_id
                new_row['parent_segment_id'] = row['segment_id']
                new_row['environment_type'] = effect_name
                new_row['augmentation_params'] = json.dumps({'effect': effect_name, 'seed': cfg.dataset['random_seed']})
                new_row['file_path'] = out_path
                new_rows.append(new_row)
            except SegmentationError:
                pass # Lọc bỏ nếu augment làm rớt QC
                
    if variants_created < required_variants and len(df_adv) > 0:
        print(f"Warning: Không đạt đủ {required_variants} variants yêu cầu do QC fail.")
        
    return pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)