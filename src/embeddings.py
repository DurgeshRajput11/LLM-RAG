import numpy as np
from abc import ABC, abstractmethod
from typing import List


try:
    from sentence_transformers import SentenceTransformer
    SBERT_AVAILABLE = True
except:
    SBERT_AVAILABLE = False

from sklearn.feature_extraction.text import TfidfVectorizer



class BaseEmbedder(ABC):

    @abstractmethod
    def encode(self, texts: List[str], **kwargs) -> np.ndarray:
        pass

    @abstractmethod
    def get_dimension(self) -> int:
        pass


class SBERTEmbedder(BaseEmbedder):

    def __init__(self, model_name="all-MiniLM-L6-v2"):
        print(f"🔹 Loading SBERT model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.dim = self.model.get_sentence_embedding_dimension()

    def encode(self, texts: List[str], **kwargs) -> np.ndarray:
        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        return embeddings.astype("float32")

    def get_dimension(self) -> int:
        return self.dim


class TfidfEmbedder(BaseEmbedder):

    def __init__(self):
        print("🔹 Using TF-IDF fallback")
        self.vectorizer = TfidfVectorizer()
        self._fitted = False

    def fit(self, texts: List[str]):
        self.vectorizer.fit(texts)
        self._fitted = True

    def encode(self, texts: List[str], **kwargs) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("TF-IDF embedder must be fitted before encoding.")

        vecs = self.vectorizer.transform(texts).toarray()
        return vecs.astype("float32")

    def get_dimension(self) -> int:
        if not self._fitted:
            return 0
        return len(self.vectorizer.get_feature_names_out())



def get_embedder(prefer_sbert=True) -> BaseEmbedder:
    """
    Returns the best available embedder.

    Priority:
    1. SBERT (semantic embeddings)
    2. TF-IDF (fallback if SBERT unavailable)
    """

    if prefer_sbert and SBERT_AVAILABLE:
        print("✅ Using SBERT embeddings")
        return SBERTEmbedder()
    else:
        print("⚠️ SBERT not available, falling back to TF-IDF")
        return TfidfEmbedder()
