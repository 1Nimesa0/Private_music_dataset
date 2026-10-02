import unicodedata
import re
import hashlib
import pandas as pd
from typing import List, Dict, Any

def normalize_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = unicodedata.normalize('NFKD', text).encode('ASCII', 'ignore').decode('utf-8')
    text = text.lower()
    text = re.sub(r'\b(feat\.?|ft\.?)\b', '', text)
    text = text.replace('&', 'and')
    text = re.sub(r'[^\w\s]', '', text)
    return ' '.join(text.split())

def make_artist_id(artist_name: str) -> str:
    norm = normalize_text(artist_name)
    if not norm:
        raise ValueError("Artist name cannot be empty after normalization.")
    sha1 = hashlib.sha1(norm.encode('utf-8')).hexdigest()[:10]
    return f"artist_{sha1}"

def make_song_id(artist_id: str, title: str) -> str:
    norm = normalize_text(title)
    sha1 = hashlib.sha1(norm.encode('utf-8')).hexdigest()[:10]
    return f"song_{artist_id}_{sha1}"

def make_segment_id(song_id: str, segment_idx: int, variant: str = "clean") -> str:
    if variant == "clean":
        return f"{song_id}_seg{segment_idx}"
    return f"{song_id}_seg{segment_idx}_{variant}"

def env_origin(env_type: str) -> str:
    if env_type == "clean": return "clean"
    if env_type.startswith("simulated_"): return "simulated"
    if env_type.startswith("real_"): return "real"
    return "unknown"

def add_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if 'environment_type' in df.columns:
        df['env_origin'] = df['environment_type'].apply(env_origin)
    if 'split' in df.columns and 'environment_type' in df.columns:
        def get_eval_set(row):
            if row['split'] == 'test':
                if row['environment_type'] == 'clean': return 'test_clean'
                if str(row['environment_type']).startswith('simulated_'): return 'test_shifted'
                if str(row['environment_type']).startswith('real_'): return 'test_real'
            return row['split']
        df['eval_set'] = df.apply(get_eval_set, axis=1)
    return df

def validate_schema(df: pd.DataFrame) -> List[str]:
    issues = []
    required_cols = [
        'segment_id', 'song_id', 'artist_id', 'genre', 'source', 'source_id',
        'source_url', 'license', 'artist', 'title', 'album', 'split',
        'environment_type', 'augmentation_params', 'parent_segment_id',
        'start_time', 'end_time', 'file_path', 'gtzan_overlap', 
        'gtzan_overlap_basis', 'review_required', 'usage_rights_confirmed_by_user',
        'usage_rights_note'
    ]
    for col in required_cols:
        if col not in df.columns:
            issues.append(f"Missing column: {col}")
    
    if 'segment_id' in df.columns and df['segment_id'].duplicated().any():
        issues.append("Duplicate segment_id found.")
        
    if 'gtzan_overlap' in df.columns:
        invalid_overlap = df[~df['gtzan_overlap'].isin(['true', 'false', 'unknown'])]
        if not invalid_overlap.empty:
            issues.append("Invalid gtzan_overlap values.")
            
        unreviewed = df[(df['gtzan_overlap'] == 'unknown') & (df['review_required'] != True) & (df['review_required'] != 'True')]
        if not unreviewed.empty:
            issues.append("Found unknown gtzan_overlap without review_required=True.")
            
    if 'environment_type' in df.columns:
        invalid_env = df[~df['environment_type'].apply(lambda x: x == 'clean' or str(x).startswith('simulated_') or str(x).startswith('real_'))]
        if not invalid_env.empty:
            issues.append("Invalid environment_type format.")
            
    return issues