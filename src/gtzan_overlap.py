import os
import logging
from typing import Tuple, Dict, Optional, List
import pandas as pd
from rapidfuzz import fuzz
import pyacoustid

logger = logging.getLogger(__name__)

class GTZANOverlapChecker:
    def __init__(self, gtzan_audio_dir: str, gtzan_meta_path: Optional[str] = None):
        """
        Khởi tạo checker. 
        Giả định bắt buộc: Người dùng phải tự cung cấp bản GTZAN hợp pháp tại gtzan_audio_dir.
        """
        self.gtzan_audio_dir = gtzan_audio_dir
        self.gtzan_metadata = self._load_metadata(gtzan_meta_path)
        self.reference_fingerprints = self._build_audio_fingerprints()

    def _load_metadata(self, meta_path: Optional[str]) -> Optional[pd.DataFrame]:
        if meta_path and os.path.exists(meta_path):
            try:
                df = pd.read_csv(meta_path)
                logger.info(f"Đã load {len(df)} records từ metadata GTZAN bổ sung.")
                return df
            except Exception as e:
                logger.warning(f"Lỗi đọc GTZAN metadata: {e}")
        logger.warning("Không tìm thấy GTZAN metadata bổ sung. Mọi so khớp văn bản sẽ trả về UNKNOWN.")
        return None

    def _build_audio_fingerprints(self) -> Dict[str, bytes]:
        """
        Quét thư mục GTZAN gốc (nếu có) để tạo fingerprint.
        Chỉ chạy 1 lần khi khởi tạo pipeline để tối ưu tốc độ.
        """
        fingerprints = {}
        if not os.path.exists(self.gtzan_audio_dir):
            logger.warning(f"Thư mục GTZAN {self.gtzan_audio_dir} không tồn tại. So khớp audio = UNKNOWN.")
            return fingerprints

        valid_exts = ('.wav', '.au')
        for root, _, files in os.walk(self.gtzan_audio_dir):
            for file in files:
                if file.lower().endswith(valid_exts):
                    path = os.path.join(root, file)
                    try:
                        # Trả về duration và fingerprint (mảng bytes)
                        duration, fp = pyacoustid.fingerprint_file(path)
                        fingerprints[file] = fp
                    except Exception as e:
                        logger.debug(f"Không thể fingerprint file {file}: {e}")
                        
        logger.info(f"Đã tạo {len(fingerprints)} audio fingerprints từ GTZAN reference.")
        return fingerprints

    def check_overlap(self, artist: str, title: str, audio_path: Optional[str] = None) -> Tuple[str, str, bool]:
        """
        Kiểm tra trùng lặp.
        Returns:
            Tuple[overlap_status (true/false/unknown), reason, review_required]
        """
        # 1. Kiểm tra bằng Audio Fingerprint (Độ tin cậy cao nhất)
        if audio_path and os.path.exists(audio_path) and self.reference_fingerprints:
            try:
                _, candidate_fp = pyacoustid.fingerprint_file(audio_path)
                for ref_name, ref_fp in self.reference_fingerprints.items():
                    # So sánh byte similarity cơ bản. 
                    # Nếu giống nhau > 90% (hoặc identical), coi như trùng file audio.
                    similarity = fuzz.ratio(candidate_fp, ref_fp)
                    if similarity > 90:
                        return "true", f"Audio fingerprint trùng với GTZAN gốc ({ref_name})", False
            except pyacoustid.FingerprintGenerationError:
                pass # Bỏ qua nếu không parse được file, rớt xuống check metadata

        # 2. Kiểm tra bằng Metadata (Fuzzy Matching)
        if self.gtzan_metadata is not None:
            # Giả định file metadata bổ sung có cột 'artist' và 'title'
            if 'artist' in self.gtzan_metadata.columns and 'title' in self.gtzan_metadata.columns:
                for _, row in self.gtzan_metadata.iterrows():
                    ref_artist = str(row.get('artist', ''))
                    ref_title = str(row.get('title', ''))
                    
                    artist_match = fuzz.token_set_ratio(artist.lower(), ref_artist.lower())
                    title_match = fuzz.token_set_ratio(title.lower(), ref_title.lower())
                    
                    if artist_match > 85 and title_match > 85:
                        return "true", f"Trùng metadata: {ref_artist} - {ref_title}", False
                
                # Nếu có metadata đầy đủ để đối chiếu và không thấy trùng
                return "false", "Không trùng artist/title trong GTZAN reference metadata", False

        # 3. Không có dữ liệu để kết luận
        return "unknown", "Thiếu GTZAN metadata/audio để xác nhận", True