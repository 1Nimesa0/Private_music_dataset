import os
import json
import logging
import subprocess
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List
from rapidfuzz import fuzz

logger = logging.getLogger(__name__)

class GTZANOverlapDetector:
    def __init__(self, config):
        self.cfg = config
        # Hỗ trợ trực tiếp thư mục genres_original của bạn nếu gtzan_reference chưa cập nhật
        ref_path_str = self.cfg.paths.get('gtzan_reference', 'data/genres_original')
        self.ref_dir = Path(ref_path_str)
        self.cache_path = self.cfg.paths.get('interim', Path('data/interim')) / 'gtzan_fingerprints_cache.json'
        
        self.reference_fps = {}
        self.is_reference_complete = False
        self.metadata_db = [] # Chứa các chuỗi normalize "artist title" nếu có metadata bổ sung
        
        self.has_fpcalc = self._check_fpcalc()

    def _check_fpcalc(self) -> bool:
        try:
            subprocess.run(['fpcalc', '-version'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
            return True
        except (FileNotFoundError, subprocess.CalledProcessError):
            return False

    def _get_raw_fingerprint(self, filepath: str) -> List[int]:
        if not self.has_fpcalc:
            raise RuntimeError("Lệnh fpcalc không tồn tại trong hệ thống")
        cmd = ['fpcalc', '-raw', '-json', str(filepath)]
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding='utf-8')
        if res.returncode != 0:
            raise RuntimeError(f"Lỗi chạy fpcalc: {res.stderr}")
        data = json.loads(res.stdout)
        return data.get('fingerprint', [])

    def load_references(self):
        if not self.ref_dir.exists():
            logger.warning(f"Không tìm thấy thư mục GTZAN tham chiếu tại {self.ref_dir}. Tất cả đánh giá sẽ trả về 'unknown'.")
            return

        audio_files = []
        # Tìm đệ quy trong tất cả các thư mục con (blues, pop,...)
        for ext in ('*.wav', '*.au'):
            audio_files.extend(self.ref_dir.rglob(ext))

        if len(audio_files) == 0:
            logger.warning(f"Thư mục {self.ref_dir} tồn tại nhưng không có file âm thanh. Tất cả đánh giá sẽ trả về 'unknown'.")
            return

        if len(audio_files) < self.cfg.min_reference_files:
            logger.warning(f"Số lượng file tham chiếu ({len(audio_files)}) nhỏ hơn yêu cầu ({self.cfg.min_reference_files}). Đánh dấu là chưa hoàn chỉnh.")
            self.is_reference_complete = False
        else:
            self.is_reference_complete = True

        cache = {}
        if self.cache_path.exists():
            try:
                with open(self.cache_path, 'r', encoding='utf-8') as f:
                    cache = json.load(f)
            except Exception as e:
                logger.error(f"Lỗi đọc cache: {e}")

        if not self.has_fpcalc:
            logger.warning("Không tìm thấy lệnh `fpcalc`. Bỏ qua tạo fingerprint cho GTZAN.")
            self.reference_fps = cache
            return

        new_cache = {}
        cache_updated = False

        for fp in audio_files:
            path_str = str(fp.resolve())
            try:
                stat = fp.stat()
                size = stat.st_size
                mtime = stat.st_mtime

                if path_str in cache and cache[path_str]['size'] == size and cache[path_str]['mtime'] == mtime:
                    new_cache[path_str] = cache[path_str]
                else:
                    fp_array = self._get_raw_fingerprint(path_str)
                    new_cache[path_str] = {
                        'size': size,
                        'mtime': mtime,
                        'fp': fp_array
                    }
                    cache_updated = True
            except Exception as e:
                logger.error(f"Lỗi tạo fingerprint cho {path_str}: {e}")

        self.reference_fps = new_cache

        if cache_updated:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.cache_path, 'w', encoding='utf-8') as f:
                json.dump(new_cache, f)

    def calculate_ber(self, fp_candidate: List[int], fp_reference: List[int]) -> float:
        if not fp_candidate or not fp_reference:
            return 1.0
        
        c_len = len(fp_candidate)
        r_len = len(fp_reference)
        
        short_fp, long_fp = (fp_candidate, fp_reference) if c_len < r_len else (fp_reference, fp_candidate)
        
        min_ber = 1.0
        window_size = len(short_fp)
        if window_size == 0: return 1.0
        
        for i in range(len(long_fp) - window_size + 1):
            window = long_fp[i:i+window_size]
            error_bits = 0
            for a, b in zip(short_fp, window):
                diff = a ^ b
                # Đếm số bit 1 của phép XOR (tốc độ cao)
                error_bits += bin(diff).count('1')
            ber = error_bits / (window_size * 32.0)
            if ber < min_ber:
                min_ber = ber
        return min_ber

    def check_overlap(self, cand_audio_path: str, cand_artist: str, cand_title: str) -> Tuple[str, str, bool]:
        """Trả về: (gtzan_overlap, basis, review_required)"""
        fp_match = False
        best_ber = 1.0
        
        if self.has_fpcalc and cand_audio_path and os.path.exists(cand_audio_path):
            try:
                cand_fp = self._get_raw_fingerprint(cand_audio_path)
                for ref_path, ref_data in self.reference_fps.items():
                    ber = self.calculate_ber(cand_fp, ref_data['fp'])
                    if ber < best_ber:
                        best_ber = ber
                        
                if best_ber <= self.cfg.ber_threshold:
                    fp_match = True
            except Exception as e:
                logger.warning(f"Lỗi giải mã fingerprint ứng viên {cand_audio_path}: {e}")

        if fp_match:
            return "true", "fingerprint", False

        meta_status = "none"
        cand_norm = f"{str(cand_artist).lower()} {str(cand_title).lower()}".strip()
        
        if cand_norm and self.metadata_db:
            best_score = 0
            for ref_meta in self.metadata_db:
                # B-05: Dùng fuzz.ratio thay vì fuzz.token_set_ratio
                score = fuzz.ratio(cand_norm, ref_meta)
                if score > best_score:
                    best_score = score
            
            if best_score >= self.cfg.metadata_match_threshold:
                meta_status = "high"
            elif best_score >= self.cfg.metadata_review_threshold:
                meta_status = "review"

        if meta_status == "high":
            return "true", "metadata", False

        if self.is_reference_complete:
            return "false", "fingerprint", False
        else:
            if meta_status == "review":
                return "unknown", "none", True
            else:
                return "unknown", "none", True