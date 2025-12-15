"""
ChromaDB fallback strategies.
"""

import logging
import operator
import pickle
from pathlib import Path

from langchain_core.documents import Document

logger = logging.getLogger(__name__)


class ChromaDBFallbackHandler:
    """
    Handle ChromaDB failures with fallback to pickle files.
    """

    def __init__(self, fallback_dir: str = "./data/fallback/chromadb") -> None:
        """
        Initialize ChromaDB fallback handler.

        Args:
            fallback_dir: Directory for fallback pickle files
        """
        self.fallback_dir = Path(fallback_dir)
        self.fallback_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"📁 ChromaDB fallback directory: {self.fallback_dir}")

    def save_documents_fallback(self, documents: list[Document], collection_name: str) -> bool:
        """
        Save documents to fallback pickle file.

        Args:
            documents: List of documents
            collection_name: Collection name

        Returns:
            True if saved successfully
        """

        try:
            fallback_file = self.fallback_dir / f"{collection_name}_documents.pkl"

            # Load existing if present
            existing = []
            if fallback_file.exists():
                with Path(fallback_file).open("rb") as f:
                    existing = pickle.load(f)

            # Append new documents
            existing.extend(documents)

            # Save
            with Path(fallback_file).open("wb") as f:
                pickle.dump(existing, f)

            logger.info(f"✅ Saved {len(documents)} documents to fallback: {fallback_file}")
            return True

        except Exception as e:
            logger.exception(f"❌ Failed to save fallback: {e}")
            return False

    def load_documents_fallback(self, collection_name: str) -> list[Document]:
        """
        Load documents from fallback pickle file.

        Args:
            collection_name: Collection name

        Returns:
            List of documents
        """

        fallback_file = self.fallback_dir / f"{collection_name}_documents.pkl"

        if not fallback_file.exists():
            logger.debug(f"📂 No fallback file: {fallback_file}")
            return []

        try:
            with Path(fallback_file).open("rb") as f:
                documents = pickle.load(f)

            logger.info(f"✅ Loaded {len(documents)} documents from fallback")
            return documents

        except Exception as e:
            logger.exception(f"❌ Failed to load fallback: {e}")
            return []

    def search_fallback(self, query: str, collection_name: str, k: int = 5) -> list[Document]:
        """
        Search in fallback documents (simple keyword matching).

        Args:
            query: Search query
            collection_name: Collection name
            k: Number of results

        Returns:
            List of matching documents
        """

        documents = self.load_documents_fallback(collection_name)

        if not documents:
            return []

        # Simple keyword-based scoring
        query_lower = query.lower()
        query_words = set(query_lower.split())

        scored_docs = []

        for doc in documents:
            content_lower = doc.page_content.lower()
            content_words = set(content_lower.split())

            # Jaccard similarity
            intersection = len(query_words & content_words)
            union = len(query_words | content_words)
            score = intersection / union if union > 0 else 0.0

            scored_docs.append((doc, score))

        # Sort by score
        scored_docs.sort(key=operator.itemgetter(1), reverse=True)

        # Return top k
        results = [doc for doc, score in scored_docs[:k]]

        logger.info(f"✅ Fallback search returned {len(results)} documents")

        return results

    def clear_fallback(self, collection_name: str) -> bool:
        """
        Clear fallback file for collection.

        Args:
            collection_name: Collection name

        Returns:
            True if cleared successfully
        """

        fallback_file = self.fallback_dir / f"{collection_name}_documents.pkl"

        if not fallback_file.exists():
            return True

        try:
            fallback_file.unlink()
            logger.info(f"🗑️ Cleared fallback: {fallback_file}")
            return True
        except Exception as e:
            logger.exception(f"❌ Failed to clear fallback: {e}")
            return False
