import pytest
import sqlite3
import json
from unittest.mock import patch, MagicMock
from pathlib import Path

from src.license_filter import LicenseFilter
from src.state_store import StateStore
from src.sources.jamendo import JamendoSource
from src.sources.fma import FMASource
from src.sources.base import Candidate

class DummyConfig:
    licensing = {'allowed_cc': ['by', 'by-nc-sa'], 'allow_nd': False}
    download = {'max_retries': 2, 'backoff_base_sec': 0.1, 'min_interval_sec': 0.0, 'timeout_sec': 5, 'oversample_candidates': 2}
    dataset = {'segments_per_song': 2, 'segment_duration_sec': 10.0}

def test_license_filter():
    filt = LicenseFilter(DummyConfig.licensing['allowed_cc'], DummyConfig.licensing['allow_nd'])
    
    # Valid
    val, cc, _ = filt.evaluate("https://creativecommons.org/licenses/by-nc-sa/4.0/")
    assert val and cc == "by-nc-sa"
    
    # B-08: Block ND when configured
    val, cc, _ = filt.evaluate("https://creativecommons.org/licenses/by-nd/4.0/")
    assert not val
    
    # Empty/Invalid
    val, _, _ = filt.evaluate("")
    assert not val

def test_state_store_idempotency(tmp_path):
    db_path = tmp_path / "test.db"
    store = StateStore(db_path)
    
    cand = Candidate("src", "id1", "url", "lic", "art", "tit", "alb", ["tag"], 100, "dl", True)
    store.save_candidate(cand, "pop")
    store.save_candidate(cand, "pop") # Idempotent test
    
    with sqlite3.connect(db_path) as conn:
        cursor = conn.execute("SELECT COUNT(*) FROM candidates")
        assert cursor.fetchone()[0] == 1

@patch('requests.get')
def test_jamendo_source_skip_no_download(mock_get):
    # B-06, B-07: Jamendo skips tracks without download
    mock_resp = MagicMock()
    mock_resp.json.return_value = {"results": [
        {"id": "1", "audiodownload_allowed": False, "audiodownload": "http://dl", "musicinfo": {}},
        {"id": "2", "audiodownload_allowed": True, "audiodownload": "", "musicinfo": {}},
        {"id": "3", "audiodownload_allowed": True, "audiodownload": "http://dl3", "musicinfo": {}}
    ]}
    mock_get.return_value = mock_resp
    
    jsource = JamendoSource()
    jsource.client_id = "dummy"
    cands = jsource.search_candidates("pop", 10, 0)
    
    assert len(cands) == 1
    assert cands[0].source_id == "jamendo_3"

def test_fma_source_not_implemented():
    fma = FMASource()
    with pytest.raises(NotImplementedError):
        fma.search_candidates("pop", 10, 0)