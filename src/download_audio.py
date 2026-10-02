import time
import hashlib
import logging
import requests
import librosa
from pathlib import Path
from src.state_store import StateStore

logger = logging.getLogger(__name__)

def sha256_file(filepath: Path) -> str:
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

def download_candidates(cfg, state_store: StateStore, candidates_list: list, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    
    max_retries = cfg.download['max_retries']
    base_backoff = cfg.download['backoff_base_sec']
    min_interval = cfg.download['min_interval_sec']
    timeout = cfg.download['timeout_sec']
    req_duration = cfg.dataset['segments_per_song'] * cfg.dataset['segment_duration_sec']

    for cand_dict in candidates_list:
        source_id = cand_dict['source_id']
        download_url = cand_dict['download_url']
        dest_path = out_dir / f"{source_id}.mp3"
        part_path = dest_path.with_suffix(".part")

        if state_store.is_rejected(source_id):
            continue

        if state_store.is_downloaded(source_id) and dest_path.exists():
            continue

        success = False
        for attempt in range(max_retries):
            try:
                time.sleep(min_interval)
                response = requests.get(download_url, stream=True, timeout=timeout)
                response.raise_for_status()

                with open(part_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                
                # Verify duration & decodability
                try:
                    duration = librosa.get_duration(path=part_path)
                    if duration < req_duration:
                        raise ValueError(f"Duration {duration}s < required {req_duration}s")
                except Exception as e:
                    raise ValueError(f"Audio decode/duration error: {e}")

                # Atomic rename
                part_path.rename(dest_path)
                file_hash = sha256_file(dest_path)
                state_store.mark_downloaded(source_id, str(dest_path), file_hash)
                logger.info(f"Downloaded {source_id}")
                success = True
                break

            except requests.RequestException as e:
                logger.warning(f"Download failed for {source_id} (Attempt {attempt+1}): {e}")
                if part_path.exists():
                    part_path.unlink()
                time.sleep(base_backoff * (2 ** attempt))
            except ValueError as e:
                logger.warning(f"Verification failed for {source_id}: {e}")
                if part_path.exists():
                    part_path.unlink()
                state_store.log_rejection(source_id, str(e))
                break # Don't retry decoding errors
        
        if not success and not state_store.is_rejected(source_id):
            state_store.log_rejection(source_id, "Max retries exceeded")