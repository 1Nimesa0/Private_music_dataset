import os
import logging
import requests
import librosa
import soundfile as sf
import numpy as np
from pathlib import Path
from typing import Dict, Optional

logger = logging.getLogger(__name__)

class AudioDownloader:
    def __init__(self, output_dir: str, config):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.config = config

    def download_and_validate(self, track: Dict) -> Optional[str]:
        """
        Tải file audio từ URL, kiểm tra các ngưỡng chất lượng (silence, clipping, RMS)
        và lưu dưới định dạng chuẩn (WAV, mono, 22050Hz).
        """
        url = track.get('source_url')
        if not url:
            logger.warning(f"Track {track.get('title')} không có source_url hợp lệ.")
            return None

        track_id = track.get('source_id')
        genre = track.get('genre', 'unknown')
        
        genre_dir = self.output_dir / genre
        genre_dir.mkdir(parents=True, exist_ok=True)
        local_filename = genre_dir / f"{track_id}.wav"

        # Nếu đã tải rồi thì bỏ qua
        if local_filename.exists():
            return str(local_filename)

        try:
            # Tải file từ URL
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()
            
            temp_file = genre_dir / f"temp_{track_id}.mp3"
            with open(temp_file, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)

            # Load và chuẩn hóa bằng librosa (mono, 22050Hz)
            target_sr = 22050
            y, sr = librosa.load(str(temp_file), sr=target_sr, mono=True)
            
            # --- KIỂM TRA CHẤT LƯỢNG (Quality Control) ---
            # 1. Kiểm tra silence ratio
            silence_threshold = 0.01
            silence_ratio = np.sum(np.abs(y) < silence_threshold) / len(y)
            if silence_ratio > self.config.quality.max_silence_ratio:
                logger.warning(f"Reject {track_id}: Tỷ lệ khoảng lặng quá cao ({silence_ratio:.2f})")
                temp_file.unlink(missing_ok=True)
                return None

            # 2. Kiểm tra clipping ratio (biên độ chạm ngưỡng 1.0)
            clipping_ratio = np.sum(np.abs(y) >= 0.99) / len(y)
            if clipping_ratio > self.config.quality.max_clipping_ratio:
                logger.warning(f"Reject {track_id}: Tỷ lệ clipping quá cao ({clipping_ratio:.2f})")
                temp_file.unlink(missing_ok=True)
                return None

            # 3. Kiểm tra năng lượng RMS (dBFS)
            rms = librosa.feature.rms(y=y)[0]
            dbfs = 20 * np.log10(np.maximum(np.mean(rms), 1e-5))
            if dbfs < self.config.quality.min_rms_dbfs:
                logger.warning(f"Reject {track_id}: Năng lượng RMS quá thấp ({dbfs:.2f} dBFS)")
                temp_file.unlink(missing_ok=True)
                return None

            # Lưu file chuẩn WAV
            sf.write(str(local_filename), y, target_sr)
            
            # Xóa file tạm
            if temp_file.exists():
                temp_file.unlink()

            logger.info(f"Đã tải và vượt qua kiểm định chất lượng: {local_filename}")
            return str(local_filename)

        except Exception as e:
            logger.error(f"Lỗi khi tải hoặc xử lý track {track_id}: {e}")
            return None