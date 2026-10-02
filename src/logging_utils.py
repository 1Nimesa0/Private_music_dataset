import logging
from pathlib import Path
import re

class APIKeyFilter(logging.Filter):
    def filter(self, record):
        record.msg = re.sub(r'([A-Za-z0-9_-]{20,})', '***', str(record.msg))
        return True

_logger_setup = False

def setup_logging(log_dir: str | Path, level: int = logging.INFO):
    global _logger_setup
    if _logger_setup:
        return
        
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    logger = logging.getLogger()
    logger.setLevel(level)
    
    fmt = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    fh = logging.FileHandler(log_dir / 'pipeline.log', encoding='utf-8')
    fh.setFormatter(fmt)
    fh.addFilter(APIKeyFilter())
    
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    ch.addFilter(APIKeyFilter())
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    _logger_setup = True