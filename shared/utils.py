"""
Shared utilities — logging, retries, text normalization.
"""

import logging
import time
import functools
import re
from typing import Callable


def setup_logger(name: str, level: str = "INFO") -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level))
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s | %(name)-20s | %(levelname)-7s | %(message)s',
            datefmt='%H:%M:%S'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def retry(max_attempts: int = 3, delay: float = 1.0, backoff: float = 2.0):
    """Retry decorator with exponential backoff.
    Used heavily in API calls and email sending."""
    def decorator(func: Callable):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            current_delay = delay
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_attempts:
                        time.sleep(current_delay)
                        current_delay *= backoff
            raise last_exception
        return wrapper
    return decorator


def normalize_text(text: str) -> str:
    """Normalize text for comparison — lowercase, strip extra whitespace, 
    remove special characters but keep meaningful punctuation."""
    if not text:
        return ""
    text = text.lower().strip()
    text = re.sub(r'\s+', ' ', text)
    # Keep alphanumeric, spaces, and basic punctuation
    text = re.sub(r'[^\w\s.,;:!?\'-]', '', text)
    return text


def word_count(text: str) -> int:
    """Count words in text, handling edge cases."""
    if not text or not text.strip():
        return 0
    return len(text.strip().split())


def unique_words_ratio(text: str) -> float:
    """Ratio of unique words to total words. 
    Low ratio → repetitive/padded content."""
    words = text.lower().split()
    if not words:
        return 0.0
    return len(set(words)) / len(words)


def hapax_legomena_ratio(text: str) -> float:
    """Ratio of words that appear exactly once. 
    High ratio in short text → likely human-written.
    Consistently moderate ratio → possibly AI-generated."""
    words = text.lower().split()
    if not words:
        return 0.0
    freq = {}
    for w in words:
        freq[w] = freq.get(w, 0) + 1
    hapax = sum(1 for count in freq.values() if count == 1)
    return hapax / len(words)
