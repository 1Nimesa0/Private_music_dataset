import pandas as pd
from pathlib import Path
from dataclasses import dataclass
import shutil

@dataclass
class LocalLoadResult:
    accepted: pd.DataFrame
    quarantined: pd.DataFrame

def load_local_metadata(csv_path: str | Path, audio_dir: str | Path, cfg) -> LocalLoadResult:
    from .metadata import normalize_text
    audio_dir = Path(audio_dir).resolve()
    
    df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
    
    accepted_rows = []
    quarantined_rows = []
    
    for _, row in df.iterrows():
        reasons = []
        is_valid = True
        
        rights = str(row.get('usage_rights_confirmed_by_user', '')).lower()
        if rights not in ['true', '1', 'yes', 'y']:
            is_valid = False
            reasons.append("Rights not confirmed")
            
        if not str(row.get('usage_rights_note', '')).strip():
            is_valid = False
            reasons.append("Missing rights note")
            
        if not str(row.get('artist', '')).strip():
            is_valid = False
            reasons.append("Empty artist")
            
        file_path = audio_dir / row.get('file_name', '')
        try:
            if not file_path.resolve().is_relative_to(audio_dir):
                is_valid = False
                reasons.append("Path traversal attempt")
        except AttributeError:
            if not str(file_path.resolve()).startswith(str(audio_dir)):
                is_valid = False
                reasons.append("Path traversal attempt")
                
        if not file_path.exists():
            is_valid = False
            reasons.append("File not found")
            
        genre = normalize_text(row.get('genre', ''))
        if genre not in cfg.genres:
            is_valid = False
            reasons.append(f"Invalid genre: {genre}")
            
        env_type = row.get('environment_type', 'clean')
        if not (env_type == 'clean' or env_type.startswith('simulated_') or env_type.startswith('real_')):
            is_valid = False
            reasons.append("Invalid environment_type")
            
        row_dict = row.to_dict()
        if is_valid:
            row_dict['source'] = 'local'
            accepted_rows.append(row_dict)
        else:
            row_dict['reason'] = '; '.join(reasons)
            quarantined_rows.append(row_dict)
            
    return LocalLoadResult(pd.DataFrame(accepted_rows), pd.DataFrame(quarantined_rows))

def quarantine_files(quarantined: pd.DataFrame, audio_dir: str | Path, quarantine_dir: str | Path):
    audio_dir = Path(audio_dir)
    quarantine_dir = Path(quarantine_dir)
    quarantine_dir.mkdir(parents=True, exist_ok=True)
    
    for _, row in quarantined.iterrows():
        src = audio_dir / row.get('file_name', '')
        if src.exists():
            dest = quarantine_dir / src.name
            if not dest.exists():
                shutil.move(str(src), str(dest))