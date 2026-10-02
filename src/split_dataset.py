import hashlib
import pandas as pd
from typing import Dict, List
import itertools

class SplitError(Exception):
    pass

class LeakageError(Exception):
    pass

def split_songs(groups: List[str], ratios: Dict[str, float], seed: int) -> Dict[str, str]:
    groups = sorted(list(set(groups)))
    if len(groups) < len(ratios):
        raise SplitError(f"Not enough groups ({len(groups)}) for splits ({len(ratios)})")
    
    group_hashes = []
    for g in groups:
        h = int(hashlib.sha256(f"{seed}_{g}".encode()).hexdigest(), 16)
        group_hashes.append((h, g))
    group_hashes.sort()
    
    total = len(groups)
    allocations = {}
    current_idx = 0
    for split_name, ratio in ratios.items():
        count = int(round(ratio * total))
        if count == 0 and ratio > 0: count = 1
        allocations[split_name] = count
        
    diff = total - sum(allocations.values())
    if diff != 0:
        allocations[list(ratios.keys())[0]] += diff
        
    result = {}
    for split_name, count in allocations.items():
        if count == 0:
            raise SplitError(f"Split {split_name} is empty.")
        for _ in range(count):
            if current_idx < len(group_hashes):
                result[group_hashes[current_idx][1]] = split_name
                current_idx += 1
    return result

def check_leakage(df: pd.DataFrame, strategy: str):
    splits = df['split'].unique()
    for s1, s2 in itertools.combinations(splits, 2):
        df1 = df[df['split'] == s1]
        df2 = df[df['split'] == s2]
        
        shared_songs = set(df1['song_id']).intersection(set(df2['song_id']))
        if shared_songs:
            raise LeakageError(f"Song leakage between {s1} and {s2}: {shared_songs}")
            
        if strategy == 'artist_disjoint':
            shared_artists = set(df1['artist_id']).intersection(set(df2['artist_id']))
            if shared_artists:
                raise LeakageError(f"Artist leakage between {s1} and {s2}: {shared_artists}")
                
    parent_splits = df[df['parent_segment_id'].isna() | (df['parent_segment_id'] == '')][['segment_id', 'split']].set_index('segment_id').to_dict()['split']
    for idx, row in df.iterrows():
        parent = row['parent_segment_id']
        if pd.notna(parent) and parent != '' and parent in parent_splits:
            if row['split'] != parent_splits[parent]:
                raise LeakageError(f"Variant {row['segment_id']} split {row['split']} does not match parent split {parent_splits[parent]}")

def split_dataset(df: pd.DataFrame, cfg) -> pd.DataFrame:
    df = df.copy()
    ratios = {
        'adaptation': cfg.split_adaptation,
        'validation': cfg.split_validation,
        'test': cfg.split_test,
        'demo': cfg.split_demo
    }
    
    clean_df = df[df['environment_type'] == 'clean']
    
    split_assignments = {}
    for genre in cfg.genres:
        genre_df = clean_df[clean_df['genre'] == genre]
        if genre_df.empty: continue
        
        if cfg.split_strategy == 'artist_disjoint':
            groups = genre_df['artist_id'].tolist()
        else:
            groups = genre_df['song_id'].tolist()
            
        try:
            genre_splits = split_songs(groups, ratios, cfg.random_seed)
        except SplitError as e:
            raise SplitError(f"Error in genre {genre}: {e}")
            
        for g, s in genre_splits.items():
            split_assignments[g] = s
            
    def assign_split(row):
        group_val = row['artist_id'] if cfg.split_strategy == 'artist_disjoint' else row['song_id']
        return split_assignments.get(group_val, 'unknown')
        
    df['split'] = df.apply(assign_split, axis=1)
    
    # Inherit from parent
    parent_map = df[df['parent_segment_id'].isna() | (df['parent_segment_id'] == '')][['segment_id', 'split']].set_index('segment_id').to_dict()['split']
    df.loc[df['parent_segment_id'].notna() & (df['parent_segment_id'] != ''), 'split'] = df['parent_segment_id'].map(parent_map)
    
    check_leakage(df, cfg.split_strategy)
    return df