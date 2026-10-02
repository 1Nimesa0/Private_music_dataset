from pathlib import Path
from typing import List, Dict, Any, Callable
import soundfile as sf
import numpy as np

def load_audio_mono(path: str | Path, sr: int) -> np.ndarray:
    import librosa
    audio, _ = librosa.load(path, sr=sr, mono=True)
    return audio

def write_wav(path: str | Path, audio: np.ndarray, sr: int):
    sf.write(path, audio, sr, subtype='PCM_16')

def process_song(input_path: str | Path, out_dir: str | Path, song_id: str, cfg, loader: Callable = load_audio_mono, writer: Callable = write_wav) -> List[Dict[str, Any]]:
    from .segmentation import extract_segments, SegmentationError
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        audio = loader(input_path, cfg.sample_rate)
    except Exception as e:
        raise Exception(f"Failed to load audio: {e}")
        
    peak = np.max(np.abs(audio))
    if peak > 1.0:
        audio = audio / peak
        
    segments_meta = []
    segs = extract_segments(audio, cfg.sample_rate, cfg)
    
    for idx, (seg_audio, start_idx, end_idx) in enumerate(segs):
        from .metadata import make_segment_id
        seg_id = make_segment_id(song_id, idx, "clean")
        out_file = out_dir / f"{seg_id}.wav"
        writer(out_file, seg_audio, cfg.sample_rate)
        
        segments_meta.append({
            "segment_id": seg_id,
            "start_time": start_idx / cfg.sample_rate,
            "end_time": end_idx / cfg.sample_rate,
            "file_path": str(out_file.relative_to(cfg.paths['output'].parent) if 'output' in cfg.paths else out_file),
            "environment_type": "clean",
            "parent_segment_id": "",
            "augmentation_params": ""
        })
    return segments_meta