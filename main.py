import argparse
import sys
import os
import pandas as pd
from src.config import Config
from src.environment_augmentation import augment_dataset
from src.playback_export import export_test_playback
from src.report import generate_report
from src.validation import validate_dataset
from src.local_source import load_local_metadata

def load_db(path):
    if os.path.exists(path):
        return pd.read_csv(path, dtype=str, keep_default_na=False)
    return pd.DataFrame()

def save_db(df, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)

def main():
    parser = argparse.ArgumentParser(description="CoTMix Private Dataset Pipeline")
    parser.add_argument('--config', default='config.yaml', help='Đường dẫn config')
    parser.add_argument('--stage', required=True, choices=['search', 'download', 'overlap', 'preprocess', 'split', 'augment', 'export_playback', 'ingest_local', 'validate', 'report', 'all'])
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args()

    cfg = Config.load(args.config)
    cfg.ensure_dirs()
    
    meta_path = os.path.join(cfg.paths['interim'], "metadata.csv")
    df = load_db(meta_path)
    
    stages = ['search', 'download', 'overlap', 'preprocess', 'split', 'augment', 'validate', 'report'] if args.stage == 'all' else [args.stage]
    
    try:
        for stage in stages:
            print(f"[{stage.upper()}] Bắt đầu...")
            
            if stage == 'augment':
                if len(df) == 0:
                    print("Dữ liệu rỗng, bỏ qua augment.")
                    continue
                df = augment_dataset(df, cfg, os.path.join(cfg.paths['interim'], 'augmented'))
                save_db(df, meta_path)
                
            elif stage == 'export_playback':
                export_test_playback(df, cfg, os.path.join(cfg.paths['local_audio'], 'playback_export'))
                
            elif stage == 'ingest_local':
                local_csv = os.path.join(cfg.paths['local_audio'], "local_metadata.csv")
                if os.path.exists(local_csv):
                    res = load_local_metadata(local_csv, cfg.paths['local_audio'], cfg)
                    # Gộp metadata thật (inherit_from_parent)
                    if not res.accepted.empty:
                        df = pd.concat([df, res.accepted], ignore_index=True)
                        save_db(df, meta_path)
                else:
                    print(f"File {local_csv} không tồn tại.")
                    
            elif stage == 'validate':
                val_result = validate_dataset(df, cfg)
                print(f"Validation Status: {val_result.status}")
                if val_result.status == 'DATASET NOT READY':
                    for r in val_result.reasons:
                        print(f" - {r}")
                    print("\nHướng dẫn: Chạy `--stage export_playback`, thu âm thật, và chạy `--stage ingest_local`.")
                    
            elif stage == 'report':
                val_result = validate_dataset(df, cfg)
                generate_report(df, cfg, val_result, cfg.paths['logs'])
                
        sys.exit(0)
    except Exception as e:
        print(f"Lỗi hệ thống: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()