import logging
from typing import Dict, List
from src.sources import JamendoSource
from src.license_filter import LicenseFilter
from src.genre_mapping import GenreMapper
from src.metadata import normalize_text, make_artist_id
from src.state_store import StateStore

logger = logging.getLogger(__name__)

def search_all_genres(cfg, state_store: StateStore):
    sources = [JamendoSource()]
    license_filter = LicenseFilter(cfg.licensing['allowed_cc'], cfg.licensing['allow_nd'])
    genre_mapper = GenreMapper(cfg.genre_mapping['min_genre_confidence'], {})

    oversample = cfg.download['oversample_candidates']
    target_count = cfg.dataset['target_songs_per_genre']
    max_per_artist = cfg.dataset['max_songs_per_artist']

    for target_genre in cfg.dataset['genres']:
        logger.info(f"Searching candidates for genre: {target_genre}")
        
        candidates_found = 0
        artist_counts: Dict[str, int] = {}
        seen_tracks: set = set()
        offset = 0
        limit = 50
        
        needed = target_count + oversample
        
        while candidates_found < needed:
            batch_added = False
            for source in sources:
                try:
                    results = source.search_candidates(target_genre, limit=limit, offset=offset)
                except NotImplementedError as e:
                    logger.error(str(e))
                    continue

                if not results:
                    continue

                for cand in results:
                    if state_store.is_rejected(cand.source_id):
                        continue

                    # 1. Deduplication
                    norm_artist = normalize_text(cand.artist)
                    norm_title = normalize_text(cand.title)
                    track_sig = f"{norm_artist}_{norm_title}"
                    if track_sig in seen_tracks:
                        continue

                    # 2. License check
                    is_valid_lic, lic_type, lic_reason = license_filter.evaluate(cand.license_url)
                    if not is_valid_lic:
                        state_store.log_rejection(cand.source_id, f"License: {lic_reason}")
                        continue

                    # 3. Genre mapping check
                    match = genre_mapper.evaluate(cand.tags, target_genre)
                    if not match.accepted:
                        state_store.log_rejection(cand.source_id, f"Genre: {match.reason}")
                        continue

                    # 4. Max artist limit
                    try:
                        artist_id = make_artist_id(cand.artist)
                    except ValueError:
                        state_store.log_rejection(cand.source_id, "Artist empty")
                        continue
                        
                    if artist_counts.get(artist_id, 0) >= max_per_artist:
                        state_store.log_rejection(cand.source_id, "Max artist limit reached")
                        continue

                    # Accept candidate
                    seen_tracks.add(track_sig)
                    artist_counts[artist_id] = artist_counts.get(artist_id, 0) + 1
                    state_store.save_candidate(cand, target_genre)
                    candidates_found += 1
                    batch_added = True

                    if candidates_found >= needed:
                        break
            
            if not batch_added:
                # API exhausted or no valid candidates found in this page
                break
            offset += limit
            
        if candidates_found < target_count:
            logger.warning(f"[INCOMPLETE] Genre '{target_genre}' only found {candidates_found}/{target_count}. "
                           "Please manually ingest via LocalSource.")