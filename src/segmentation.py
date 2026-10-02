import numpy as np
from typing import List, Tuple

class SegmentationError(Exception):
    pass

def frame_rms_dbfs(audio: np.ndarray, frame_length: int = 2048, hop_length: int = 512) -> np.ndarray:
    if len(audio) == 0: return np.array([])
    pad_len = frame_length - (len(audio) % hop_length)
    if pad_len > 0: audio = np.pad(audio, (0, pad_len))
    num_frames = 1 + (len(audio) - frame_length) // hop_length
    rms_vals = []
    for i in range(num_frames):
        frame = audio[i * hop_length : i * hop_length + frame_length]
        rms = np.sqrt(np.mean(frame**2))
        dbfs = 20 * np.log10(rms) if rms > 0 else -100.0
        rms_vals.append(dbfs)
    return np.array(rms_vals)

def check_segment_quality(audio: np.ndarray, sr: int, duration_sec: float, cfg) -> Tuple[bool, str]:
    expected_samples = int(duration_sec * sr)
    tolerance_samples = int(cfg.min_duration_tolerance_sec * sr)
    if abs(len(audio) - expected_samples) > tolerance_samples:
        return False, "Length outside tolerance."
        
    if np.any(np.isnan(audio)) or np.any(np.isinf(audio)):
        return False, "Contains NaN or Inf."
        
    if np.any(np.abs(audio) >= 0.99):
        clipping_ratio = np.sum(np.abs(audio) >= 0.99) / len(audio)
        if clipping_ratio > cfg.max_clipping_ratio:
            return False, f"Clipping ratio {clipping_ratio:.3f} > {cfg.max_clipping_ratio}."
            
    rms_db = frame_rms_dbfs(audio)
    if len(rms_db) > 0 and np.mean(rms_db) < cfg.min_rms_dbfs:
         return False, f"Mean RMS {np.mean(rms_db):.1f} < {cfg.min_rms_dbfs}."
         
    silence_ratio = np.sum(rms_db < cfg.silence_frame_dbfs) / len(rms_db) if len(rms_db) > 0 else 1.0
    if silence_ratio > cfg.max_silence_ratio:
         return False, f"Silence ratio {silence_ratio:.3f} > {cfg.max_silence_ratio}."
         
    return True, "OK"

def plan_slots(total_samples: int, sr: int, cfg) -> List[Tuple[int, int]]:
    margin = int(total_samples * cfg.edge_margin_ratio)
    usable_start, usable_end = margin, total_samples - margin
    usable_samples = usable_end - usable_start
    segment_samples = int(cfg.segment_duration_sec * sr)
    
    if usable_samples < cfg.segments_per_song * segment_samples:
        usable_start, usable_end = 0, total_samples
        usable_samples = total_samples
        
    if usable_samples < cfg.segments_per_song * segment_samples:
        raise SegmentationError(f"Audio too short for {cfg.segments_per_song} segments.")
        
    slot_size = usable_samples // cfg.segments_per_song
    slots = []
    for i in range(cfg.segments_per_song):
        start = usable_start + i * slot_size
        end = start + slot_size
        slots.append((start, end))
    return slots

def extract_segments(audio: np.ndarray, sr: int, cfg) -> List[Tuple[np.ndarray, int, int]]:
    slots = plan_slots(len(audio), sr, cfg)
    segment_samples = int(cfg.segment_duration_sec * sr)
    segments = []
    for slot_start, slot_end in slots:
        slot_dur = slot_end - slot_start
        found = False
        for offset_ratio in cfg.offset_candidates:
            start = slot_start + int((slot_dur - segment_samples) * offset_ratio)
            end = start + segment_samples
            if end > len(audio): continue
            
            seg_audio = audio[start:end]
            ok, _ = check_segment_quality(seg_audio, sr, cfg.segment_duration_sec, cfg)
            if ok:
                segments.append((seg_audio, start, end))
                found = True
                break
        if not found:
            raise SegmentationError("No valid segment found in slot.")
    return segments