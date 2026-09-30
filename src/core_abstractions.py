from abc import ABC, abstractmethod
from typing import List, Dict

class MusicSource(ABC):
    """Abstract Base Class cho mọi nguồn dữ liệu âm thanh."""
    
    @abstractmethod
    def search_candidates(self, genre: str, limit: int) -> List[Dict]:
        """Tìm kiếm metadata của ứng viên dựa trên thể loại."""
        pass

    @abstractmethod
    def download_audio(self, source_id: str, save_path: str) -> bool:
        """Tải file âm thanh về máy."""
        pass