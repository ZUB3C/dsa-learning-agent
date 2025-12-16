"""
TF-IDF Retriever for keyword-based document search.
Uses sklearn TfidfVectorizer with Russian language support.
"""

import logging
import pickle
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.config import get_settings
from src.tools.base_tool import Document

logger = logging.getLogger(__name__)


class TFIDFRetriever:
    """
    TF-IDF based document retriever.

    Uses sklearn TfidfVectorizer with:
    - Russian stop words
    - Character n-grams (2-4) for better recall
    - Cosine similarity for ranking
    """

    def __init__(self, model_path: str | None = None) -> None:
        """
        Initialize TF-IDF retriever.

        Args:
            model_path: Path to saved model (default: from settings)
        """
        self.settings = get_settings()

        # Model path
        if model_path:
            self.model_path = Path(model_path)
        else:
            self.model_path = Path(self.settings.adaptive_rag.adaptive_tfidf_model_path)

        # Create directory if not exists
        self.model_path.parent.mkdir(parents=True, exist_ok=True)

        # Initialize vectorizer
        self.vectorizer: TfidfVectorizer | None = None
        self.tfidf_matrix: np.ndarray | None = None
        self.documents: list[Document] = []

        # Try to load existing model
        self._load_model()

    def _load_model(self) -> bool:
        """
        Load TF-IDF model from disk.

        Returns:
            True if loaded successfully, False otherwise
        """

        if not self.model_path.exists():
            logger.info("📂 TF-IDF model not found, will need to build")
            return False

        try:
            with Path(self.model_path).open("rb") as f:
                data = pickle.load(f)

            self.vectorizer = data["vectorizer"]
            self.tfidf_matrix = data["tfidf_matrix"]
            self.documents = data["documents"]

            logger.info(
                f"✅ Loaded TF-IDF model: {len(self.documents)} documents, "
                f"vocab size: {len(self.vectorizer.vocabulary_)}"
            )

            return True

        except Exception as e:
            logger.exception(f"❌ Failed to load TF-IDF model: {e}")
            return False

    def _save_model(self) -> None:
        """Save TF-IDF model to disk."""

        try:
            data = {
                "vectorizer": self.vectorizer,
                "tfidf_matrix": self.tfidf_matrix,
                "documents": self.documents,
            }

            with Path(self.model_path).open("wb") as f:
                pickle.dump(data, f)

            logger.info(f"💾 Saved TF-IDF model to {self.model_path}")

        except Exception as e:
            logger.exception(f"❌ Failed to save TF-IDF model: {e}")

    async def build_index(self, documents: list[Document]) -> bool:
        """
        Build TF-IDF index from documents.

        Args:
            documents: List of documents to index

        Returns:
            True if successful
        """

        if not documents:
            logger.warning("⚠️ No documents provided to build TF-IDF index")
            return False

        logger.info(f"🔨 Building TF-IDF index from {len(documents)} documents...")

        try:
            # Extract texts
            texts = [doc.page_content for doc in documents]

            # Create vectorizer with Russian support
            self.vectorizer = TfidfVectorizer(
                max_features=10000,  # Limit vocabulary size
                min_df=2,  # Ignore terms that appear in less than 2 documents
                max_df=0.8,  # Ignore terms that appear in more than 80% of documents
                ngram_range=(1, 2),  # Use unigrams and bigrams
                analyzer="char_wb",  # Character n-grams (better for Russian)
                lowercase=True,
                strip_accents="unicode",
                stop_words=self._get_russian_stopwords(),
            )

            # Fit and transform
            self.tfidf_matrix = self.vectorizer.fit_transform(texts)
            self.documents = documents

            logger.info(
                f"✅ TF-IDF index built: {len(documents)} documents, "
                f"vocab size: {len(self.vectorizer.vocabulary_)}"
            )

            # Save model
            self._save_model()

            return True

        except Exception as e:
            logger.exception(f"❌ Failed to build TF-IDF index: {e}")
            return False

    async def search(self, query: str, k: int = 5) -> list[Document]:
        """
        Search documents using TF-IDF.

        Args:
            query: Search query
            k: Number of documents to return

        Returns:
            List of top-k documents
        """

        if not self.vectorizer or self.tfidf_matrix is None:
            logger.error("❌ TF-IDF model not initialized")
            return []

        if not self.documents:
            logger.warning("⚠️ No documents in TF-IDF index")
            return []

        try:
            # Vectorize query
            query_vector = self.vectorizer.transform([query])

            # Calculate cosine similarity
            similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()

            # Get top-k indices
            top_indices = np.argsort(similarities)[::-1][:k]

            # Get top-k documents
            results = []
            for idx in top_indices:
                if similarities[idx] > 0:  # Only return documents with non-zero similarity
                    doc = self.documents[idx]
                    # Add similarity score to metadata
                    doc.metadata["tfidf_score"] = float(similarities[idx])
                    results.append(doc)

            logger.info(
                f"🔍 TF-IDF search: query='{query[:50]}...', found {len(results)} documents"
            )

            return results

        except Exception as e:
            logger.exception(f"❌ TF-IDF search failed: {e}")
            return []

    def _get_russian_stopwords(self) -> list[str]:
        """
        Get Russian stop words.

        Returns:
            List of stop words
        """

        # Common Russian stop words
        return [
            "и",
            "в",
            "во",
            "не",
            "что",
            "он",
            "на",
            "я",
            "с",
            "со",
            "как",
            "а",
            "то",
            "все",
            "она",
            "так",
            "его",
            "но",
            "да",
            "ты",
            "к",
            "у",
            "же",
            "вы",
            "за",
            "бы",
            "по",
            "только",
            "ее",
            "мне",
            "было",
            "вот",
            "от",
            "меня",
            "еще",
            "нет",
            "о",
            "из",
            "ему",
            "теперь",
            "когда",
            "даже",
            "ну",
            "вдруг",
            "ли",
            "если",
            "уже",
            "или",
            "ни",
            "быть",
            "был",
            "него",
            "до",
            "вас",
            "нибудь",
            "опять",
            "уж",
            "вам",
            "ведь",
            "там",
            "потом",
            "себя",
            "ничего",
            "ей",
            "может",
            "они",
            "тут",
            "где",
            "есть",
            "надо",
            "ней",
            "для",
            "мы",
            "тебя",
            "их",
            "чем",
            "была",
            "сам",
            "чтоб",
            "без",
            "будто",
            "чего",
            "раз",
            "тоже",
            "себе",
            "под",
            "будет",
            "ж",
            "тогда",
            "кто",
            "этот",
            "того",
            "потому",
            "этого",
            "какой",
            "совсем",
            "ним",
            "здесь",
            "этом",
            "один",
            "почти",
            "мой",
            "тем",
            "чтобы",
            "нее",
        ]

    def is_ready(self) -> bool:
        """
        Check if retriever is ready to search.

        Returns:
            True if model is loaded and ready
        """
        return (
            self.vectorizer is not None
            and self.tfidf_matrix is not None
            and len(self.documents) > 0
        )


# Global retriever instance
_retriever: TFIDFRetriever | None = None


def get_tfidf_retriever() -> TFIDFRetriever:
    """Get global TF-IDF retriever instance."""
    global _retriever

    if _retriever is None:
        _retriever = TFIDFRetriever()

    return _retriever
