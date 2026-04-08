import re
import numpy as np


# ─────────────────────────────────────────────
# TEXT PROCESSING
# ─────────────────────────────────────────────

def split_into_sentences(text: str):
    """
    Splits text into meaningful sentences.
    Removes very short/noisy fragments.
    """
    sentences = re.split(r'(?<=[.?!])\s+(?=[A-Z\"\(])', text.strip())
    return [s.strip() for s in sentences if len(s.strip()) > 30]


def extract_title(text: str):
    """
    Extracts title from first line.
    """
    first_line = text.strip().splitlines()[0]
    return first_line.replace("Title:", "").strip()


# ─────────────────────────────────────────────
# NORMALIZATION
# ─────────────────────────────────────────────

def normalize_vectors(x: np.ndarray):
    """
    L2 normalize vectors for cosine similarity.
    """
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    norm = np.where(norm == 0, 1, norm)
    return x / norm


def normalize_vector(v: np.ndarray):
    """
    Normalize single vector.
    """
    norm = np.linalg.norm(v)
    return v / (norm + 1e-8)


# ─────────────────────────────────────────────
# TEXT CLEANING
# ─────────────────────────────────────────────

def clean_text(text: str):
    """
    Basic cleaning:
    - lowercase
    - remove extra spaces
    """
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


# ─────────────────────────────────────────────
# DEBUGGING HELPERS
# ─────────────────────────────────────────────

def preview_text(text: str, length=200):
    """
    Short preview of long text (for logging/debugging).
    """
    return text[:length] + "..." if len(text) > length else text