from src.sources.base import MusicSource, Candidate
from typing import List

class FMASource(MusicSource):
    def search_candidates(self, genre: str, limit: int, offset: int) -> List[Candidate]:
        raise NotImplementedError(
            "FMA API is currently unstable/offline for public programmatic access. "
            "Please use Jamendo or LocalSource."
        )