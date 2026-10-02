import os
import json
import pandas as pd
import matplotlib.pyplot as plt
from typing import Any

def generate_report(df: pd.DataFrame, cfg: Any, validation_result: Any, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    
    # Tính toán các chỉ số
    stats = {
        "total_segments": len(df),
        "clean_segments": len(df[df['parent_segment_id'].isna() | (df['parent_segment_id'] == "")]),
        "augmented_segments": len(df[df['environment_type'].str.startswith('simulated_')]),
        "real_segments": len(df[df['environment_type'].str.startswith('real_')]),
        "dataset_status": validation_result.status,
        "validation_reasons": validation_result.reasons
    }
    
    # 1. Báo cáo JSON
    with open(os.path.join(out_dir, "dataset_report.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=4, ensure_ascii=False)
        
    # 2. Đồ thị phân bố
    plt.figure(figsize=(10, 6))
    df['genre'].value_counts().plot(kind='bar', color='skyblue')
    plt.title("Distribution by Genre")
    plt.xlabel("Genre")
    plt.ylabel("Number of Segments")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "genre_distribution.png"))
    plt.close()
    
    plt.figure(figsize=(10, 6))
    df['split'].value_counts().plot(kind='pie', autopct='%1.1f%%')
    plt.title("Split Distribution")
    plt.ylabel("")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "split_distribution.png"))
    plt.close()
    
    # 3. Báo cáo Markdown
    md_content = f"""# Private Dataset Report
**Status:** {stats['dataset_status']}

## Lỗi/Cảnh báo Validation:
"""
    if stats['validation_reasons']:
        for reason in stats['validation_reasons']:
            md_content += f"- {reason}\n"
    else:
        md_content += "- Không có lỗi.\n"
        
    md_content += f"""
## Thống kê chung:
- Tổng số segment: {stats['total_segments']}
- Segment gốc (clean): {stats['clean_segments']}
- Segment mô phỏng (augmented): {stats['augmented_segments']}
- Segment thu âm thật (real): {stats['real_segments']}

*(Xem thêm các biểu đồ `.png` trong thư mục này)*
"""
    with open(os.path.join(out_dir, "dataset_report.md"), "w", encoding="utf-8") as f:
        f.write(md_content)