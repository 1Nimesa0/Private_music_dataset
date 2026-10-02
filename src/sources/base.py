from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List

@dataclass
class Candidate:
    source: str
    source_id: str
    source_url: str
    license_url: str
    artist: str
    title: str
    album: str
    tags: List[str]
    duration: int
    download_url: str
    download_allowed: bool

class MusicSource(ABC):
    @abstractmethod
    def search_candidates(self, genre: str, limit: int, offset: int) -> List[Candidate]:
        pass