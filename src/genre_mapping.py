from dataclasses import dataclass
from typing import List, Dict

@dataclass
class GenreMatch:
    matched_genres: List[str]
    confidence: float
    accepted: bool
    reason: str

class GenreMapper:
    def __init__(self, min_confidence: float, extra_synonyms: Dict[str, str] = None):
        self.min_confidence = min_confidence
        self.mapping = {
            'hiphop': 'hiphop', 'hip-hop': 'hiphop', 'rap': 'hiphop',
            'blues': 'blues', 'classical': 'classical', 'classic': 'classical',
            'country': 'country', 'disco': 'disco',
            'jazz': 'jazz', 'metal': 'metal', 'heavy metal': 'metal',
            'pop': 'pop', 'reggae': 'reggae', 'rock': 'rock'
        }
        if extra_synonyms:
            for k, v in extra_synonyms.items():
                if k in self.mapping and self.mapping[k] != v:
                    raise ValueError(f"Synonym conflict for {k}: {self.mapping[k]} vs {v}")
                self.mapping[k] = v

    def evaluate(self, tags: List[str], expected_genre: str = None) -> GenreMatch:
        normalized_tags = [str(t).lower().strip() for t in tags]
        matched = set()
        
        for t in normalized_tags:
            if t in self.mapping:
                matched.add(self.mapping[t])
                
        if not matched:
            return GenreMatch([], 0.0, False, "No recognized genres found in tags.")
            
        confidence = 1.0 / len(matched)
        
        if expected_genre and expected_genre not in matched:
            return GenreMatch(list(matched), confidence, False, f"Expected {expected_genre} but got {list(matched)}")
            
        accepted = confidence >= self.min_confidence
        reason = "Accepted." if accepted else f"Confidence {confidence} below threshold {self.min_confidence}."
        
        return GenreMatch(list(matched), confidence, accepted, reason)