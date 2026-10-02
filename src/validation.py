import pandas as pd
from dataclasses import dataclass
from typing import List, Dict

@dataclass
class ValidationResult:
    status: str
    reasons: List[str]
    stats: Dict

def validate_dataset(df: pd.DataFrame, cfg) -> ValidationResult:
    from .metadata import validate_schema, add_derived_columns
    from .split_dataset import check_leakage, LeakageError
    
    reasons = []
    df = add_derived_columns(df)
    schema_issues = validate_schema(df)
    if schema_issues: reasons.extend(schema_issues)
    
    clean_original = df[(df['environment_type'] == 'clean') & (df['parent_segment_id'] == '')]
    
    # 1. Genres limit
    incomplete_genres = []
    for g in cfg.genres:
        count = clean_original[clean_original['genre'] == g]['song_id'].nunique()
        if count < cfg.target_songs_per_genre:
            incomplete_genres.append(f"{g} ({count}/{cfg.target_songs_per_genre})")
    if incomplete_genres:
        reasons.append(f"Incomplete genres: {', '.join(incomplete_genres)}")
        
    # 2. Segments per song
    song_counts = clean_original.groupby('song_id').size()
    bad_songs = song_counts[song_counts != cfg.segments_per_song]
    if not bad_songs.empty:
        reasons.append(f"Songs not having {cfg.segments_per_song} segments: {len(bad_songs)}")
        
    # 3. Songs per artist
    artist_counts = clean_original.groupby('artist_id')['song_id'].nunique()
    bad_artists = artist_counts[artist_counts > cfg.max_songs_per_artist]
    if not bad_artists.empty:
        reasons.append(f"Artists exceeding max songs: {len(bad_artists)}")
        
    # 4. Leakage
    try:
        check_leakage(df, cfg.split_strategy)
    except LeakageError as e:
        reasons.append(str(e))
        
    # 5. Environment diversity
    env_types = df['environment_type'].nunique()
    if env_types < cfg.min_environment_types:
        reasons.append(f"environment diversity insufficient — thí nghiệm domain adaptation sẽ không có ý nghĩa. ({env_types} < {cfg.min_environment_types})")
        
    # 6. Non-clean ratio
    non_test = df[df['split'].isin(['adaptation', 'validation', 'demo'])]
    if not non_test.empty:
        non_clean = non_test[non_test['env_origin'] != 'clean']
        ratio = len(non_clean) / len(non_test)
        if ratio < cfg.min_non_clean_ratio:
            reasons.append(f"Non-clean ratio {ratio:.2f} < {cfg.min_non_clean_ratio}")
            
    # 7. Test constraints
    test_clean = df[df['eval_set'] == 'test_clean']
    for seg_id in test_clean['segment_id']:
        shifted = df[(df['parent_segment_id'] == seg_id) & (df['eval_set'] == 'test_shifted')]
        if len(shifted) != 1:
            reasons.append(f"Test segment {seg_id} does not have exactly 1 test_shifted variant.")
            
    test_real = df[df['eval_set'] == 'test_real']
    if len(test_real) < cfg.min_test_real_samples:
        reasons.append(f"test_real samples {len(test_real)} < {cfg.min_test_real_samples}")
        
    # 8. Overlap
    if 'gtzan_overlap' in df.columns:
        if (df['gtzan_overlap'] == 'true').any():
            reasons.append("Contains gtzan_overlap == true.")
        if ((df['gtzan_overlap'] == 'unknown') & (df['review_required'] == True)).any():
            reasons.append("Contains unresolved review_required samples.")
            
    status = "READY" if not reasons else "DATASET NOT READY"
    return ValidationResult(status, reasons, {})