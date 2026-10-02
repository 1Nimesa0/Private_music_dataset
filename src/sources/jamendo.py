import os
import logging
import requests
from typing import List
from dotenv import load_dotenv
from src.sources.base import MusicSource, Candidate

load_dotenv()
logger = logging.getLogger(__name__)

class JamendoSource(MusicSource):
    BASE_URL = "https://api.jamendo.com/v3.0/tracks/"

    def __init__(self):
        self.client_id = os.getenv("JAMENDO_CLIENT_ID")
        if not self.client_id:
            logger.warning("JAMENDO_CLIENT_ID is missing in .env. Jamendo searches will fail.")

    def search_candidates(self, genre: str, limit: int, offset: int) -> List[Candidate]:
        if not self.client_id:
            return []

        params = {
            "client_id": self.client_id,
            "format": "json",
            "limit": limit,
            "offset": offset,
            "tags": genre,
            "include": "musicinfo+licenses"
        }

        try:
            response = requests.get(self.BASE_URL, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            candidates = []
            for track in data.get("results", []):
                # Theo quy tắc P3-1: Chỉ nhận track có download_allowed=True và có url
                audiodownload_allowed = track.get("audiodownload_allowed", False)
                download_url = track.get("audiodownload", "")

                if not audiodownload_allowed or not download_url:
                    continue

                candidates.append(Candidate(
                    source="jamendo",
                    source_id=f"jamendo_{track['id']}",
                    source_url=track.get("shareurl", ""),
                    license_url=track.get("license_ccurl", ""),
                    artist=track.get("artist_name", ""),
                    title=track.get("name", ""),
                    album=track.get("album_name", ""),
                    tags=track.get("musicinfo", {}).get("tags", []),
                    duration=int(track.get("duration", 0)),
                    download_url=download_url,
                    download_allowed=audiodownload_allowed
                ))
            return candidates
        except requests.RequestException as e:
            logger.error(f"Jamendo API error: {e}")
            return []