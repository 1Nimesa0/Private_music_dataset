import os
import logging
import requests
from typing import List, Dict, Optional
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

class JamendoSource:
    BASE_URL = "https://api.jamendo.com/v3.0/tracks/"

    def __init__(self, client_id: Optional[str] = None):
        # Lấy client_id từ biến môi trường hoặc tham số truyền vào
        self.client_id = client_id or os.getenv("JAMENDO_CLIENT_ID")
        if not self.client_id:
            logger.warning("Chưa cấu hình JAMENDO_CLIENT_ID. JamendoSource sẽ không thể gọi API thực tế.")

    def search_tracks(self, genre: str, limit: int = 50) -> List[Dict]:
        """
        Tìm kiếm candidate tracks từ Jamendo theo genre với cơ chế oversampling.
        """
        if not self.client_id:
            raise ValueError("Thiếu JAMENDO_CLIENT_ID trong biến môi trường (.env) hoặc config.")

        params = {
            'client_id': self.client_id,
            'format': 'jsonjson',
            'limit': limit,
            'tags': genre,
            'include': 'licenses+musicinfo',
            'audioformat': 'mp32' # Hoặc flac/ogg nếu API hỗ trợ tốt
        }

        try:
            response = requests.get(self.BASE_URL, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            tracks = []
            results = data.get('results', [])
            for item in results:
                # Kiểm tra giấy phép CC cơ bản (tránh bản quyền thương mại khắt khe)
                license_url = item.get('license_ccurl', '')
                if 'creativecommons.org' in license_url.lower():
                    track_info = {
                        'source': 'jamendo',
                        'source_id': item.get('id'),
                        'artist': item.get('artist_name'),
                        'title': item.get('name'),
                        'album': item.get('album_name', 'Unknown Album'),
                        'source_url': item.get('audiodownload') or item.get('audio'),
                        'license': license_url,
                        'genre': genre,
                        'environment_type': 'clean' # Mặc định nguồn tải về là clean, sẽ được augmentation xử lý sau
                    }
                    tracks.append(track_info)
                    
            logger.info(f"Jamendo API: Tìm thấy {len(tracks)} track hợp lệ cho genre '{genre}' (sau khi lọc license).")
            return tracks

        except requests.exceptions.RequestException as e:
            logger.error(f"Lỗi kết nối tới Jamendo API cho genre {genre}: {e}")
            return []