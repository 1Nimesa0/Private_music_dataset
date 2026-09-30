import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

def split_disjoint_dataset(metadata_df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Chia dataset đảm bảo KHÔNG có Data Leakage.
    Tất cả các segments của cùng 1 song_id (hoặc artist_id) 
    phải nằm trọn trong 1 split duy nhất.
    """
    strategy = config['splits']['strategy']
    group_col = 'artist_id' if strategy == 'artist_disjoint' else 'song_id'
    
    # Tính tỷ lệ thực tế
    val_test_demo_ratio = config['splits']['validation'] + config['splits']['test'] + config['splits']['demo']
    
    # Tách Adaptation (Train) và phần còn lại
    gss_1 = GroupShuffleSplit(n_splits=1, test_size=val_test_demo_ratio, random_state=config['dataset']['random_seed'])
    train_idx, temp_idx = next(gss_1.split(metadata_df, groups=metadata_df[group_col]))
    
    train_df = metadata_df.iloc[train_idx].copy()
    train_df['split'] = 'adaptation'
    temp_df = metadata_df.iloc[temp_idx].copy()
    
    # Tính tỷ lệ nội bộ cho Val, Test, Demo từ temp_df
    total_temp = val_test_demo_ratio
    val_ratio_internal = config['splits']['validation'] / total_temp
    
    gss_2 = GroupShuffleSplit(n_splits=1, train_size=val_ratio_internal, random_state=config['dataset']['random_seed'])
    val_idx, test_demo_idx = next(gss_2.split(temp_df, groups=temp_df[group_col]))
    
    val_df = temp_df.iloc[val_idx].copy()
    val_df['split'] = 'validation'
    test_demo_df = temp_df.iloc[test_demo_idx].copy()
    
    # Cuối cùng tách Test và Demo
    demo_ratio_internal = config['splits']['demo'] / (config['splits']['test'] + config['splits']['demo'])
    gss_3 = GroupShuffleSplit(n_splits=1, test_size=demo_ratio_internal, random_state=config['dataset']['random_seed'])
    test_idx, demo_idx = next(gss_3.split(test_demo_df, groups=test_demo_df[group_col]))
    
    test_df = test_demo_df.iloc[test_idx].copy()
    test_df['split'] = 'test'
    demo_df = test_demo_df.iloc[demo_idx].copy()
    demo_df['split'] = 'demo'
    
    final_df = pd.concat([train_df, val_df, test_df, demo_df])
    
    # ASSERTION: Đảm bảo khoa học dữ liệu chặt chẽ (Fail fast nếu Leakage xảy ra)
    train_groups = set(train_df[group_col])
    test_groups = set(test_df[group_col])
    assert train_groups.isdisjoint(test_groups), f"DATA LEAKAGE DETECTED on {group_col}!"
    
    return final_df