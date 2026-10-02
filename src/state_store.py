import sqlite3
from pathlib import Path
from dataclasses import asdict
import json

class StateStore:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS candidates (
                    source_id TEXT PRIMARY KEY,
                    source TEXT,
                    artist TEXT,
                    title TEXT,
                    genre TEXT,
                    data JSON
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS downloads (
                    source_id TEXT PRIMARY KEY,
                    file_path TEXT,
                    sha256 TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS rejections (
                    source_id TEXT PRIMARY KEY,
                    reason TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)

    def save_candidate(self, candidate, genre: str):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO candidates (source_id, source, artist, title, genre, data) VALUES (?, ?, ?, ?, ?, ?)",
                (candidate.source_id, candidate.source, candidate.artist, candidate.title, genre, json.dumps(asdict(candidate)))
            )

    def log_rejection(self, source_id: str, reason: str):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO rejections (source_id, reason) VALUES (?, ?)",
                (source_id, reason)
            )

    def is_rejected(self, source_id: str) -> bool:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT 1 FROM rejections WHERE source_id = ?", (source_id,))
            return cursor.fetchone() is not None

    def mark_downloaded(self, source_id: str, file_path: str, sha256_hash: str):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO downloads (source_id, file_path, sha256) VALUES (?, ?, ?)",
                (source_id, file_path, sha256_hash)
            )

    def is_downloaded(self, source_id: str) -> bool:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT 1 FROM downloads WHERE source_id = ?", (source_id,))
            return cursor.fetchone() is not None