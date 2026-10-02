import os
import shutil
import pandas as pd
from typing import Any

def export_test_playback(df: pd.DataFrame, cfg: Any, export_dir: str):
    os.makedirs(export_dir, exist_ok=True)
    test_clean = df[(df['split'] == 'test') & (df['parent_segment_id'].isna() | (df['parent_segment_id'] == ""))]
    
    template_rows = []
    for _, row in test_clean.iterrows():
        dest_path = os.path.join(export_dir, f"{row['segment_id']}.wav")
        shutil.copy2(row['file_path'], dest_path)
        
        template_rows.append({
            'file_name': f"{row['segment_id']}.wav",
            'parent_segment_id': row['segment_id'],
            'genre': row['genre'],
            'artist': row['artist'],
            'title': row['title'],
            'environment_type': 'real_room',
            'recording_device': 'iphone_mic',
            'playback_device': 'laptop_speaker',
            'usage_rights_confirmed_by_user': 'True',
            'usage_rights_note': 'Self-recorded test playback'
        })
        
    pd.DataFrame(template_rows).to_csv(os.path.join(export_dir, "local_metadata_template.csv"), index=False)