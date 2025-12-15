"""
Build TF-IDF corpus for Adaptive RAG.

This pre-computes TF-IDF vectors for the knowledge base,
enabling fast keyword-based retrieval.
"""

import asyncio
import logging
import pickle
from pathlib import Path
from typing import Any

from langchain_core.documents import Document
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer

from src.config import get_settings
from src.core.vector_store import vector_store_manager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TFIDFCorpusBuilder:
    """
    Build TF-IDF corpus from ChromaDB documents.
    """

    def __init__(self, output_dir: str = "./data/tfidf") -> None:
        """
        Initialize corpus builder.

        Args:
            output_dir: Directory to save TF-IDF models
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.settings = get_settings()

    async def fetch_all_documents(self) -> list[Document]:
        """
        Fetch all documents from ChromaDB.

        Returns:
            List of documents
        """

        logger.info("📚 Fetching all documents from ChromaDB...")

        try:
            # Access collection through _client
            collection = vector_store_manager.client.get_collection(
                name=self.settings.vector_store.collection_name
            )

            # Get all documents (in batches)
            all_results = collection.get()

            if not all_results or not all_results["documents"]:
                logger.warning("⚠️ No documents found in ChromaDB")
                return []

            # Convert to Document objects
            documents = []

            for _i, (doc_text, metadata, doc_id) in enumerate(
                zip(
                    all_results["documents"],
                    all_results["metadatas"],
                    all_results["ids"],
                    strict=False,
                )
            ):
                documents.append(
                    Document(page_content=doc_text, metadata=metadata or {}, id=doc_id)
                )

            logger.info(f"✅ Fetched {len(documents)} documents")

            return documents

        except Exception as e:
            logger.exception(f"❌ Failed to fetch documents: {e}")
            return []

    def build_tfidf_model(
        self, documents: list[Document]
    ) -> tuple[TfidfVectorizer, Any, list[str]]:
        """
        Build TF-IDF vectorizer and transform documents.

        Args:
            documents: List of documents

        Returns:
            Tuple of (vectorizer, document_vectors, document_ids)
        """

        logger.info("🔧 Building TF-IDF model...")

        # Extract text
        corpus = [doc.page_content for doc in documents]
        doc_ids = [doc.metadata.get("id", f"doc_{i}") for i, doc in enumerate(documents)]

        # Build vectorizer
        vectorizer = TfidfVectorizer(
            max_features=10000,
            min_df=2,
            max_df=0.8,
            ngram_range=(1, 2),
            stop_words="english",  # Add Russian stopwords if needed
        )

        # Fit and transform
        logger.info("📊 Fitting TF-IDF vectorizer...")
        document_vectors = vectorizer.fit_transform(corpus)

        logger.info("✅ Built TF-IDF model:")
        logger.info(f"   - Vocabulary size: {len(vectorizer.vocabulary_)}")
        if hasattr(document_vectors, "shape"):
            logger.info(f"   - Document vectors shape: {document_vectors.shape}")

        return vectorizer, document_vectors, doc_ids

    def save_model(
        self,
        vectorizer: TfidfVectorizer,
        document_vectors: csr_matrix,
        doc_ids: list[str],
    ) -> None:
        """
        Save TF-IDF model to disk.

        Args:
            vectorizer: TfidfVectorizer
            document_vectors: Sparse matrix of document vectors
            doc_ids: List of document IDs
        """

        logger.info(f"💾 Saving TF-IDF model to {self.output_dir}...")

        # Save vectorizer
        vectorizer_path = self.output_dir / "tfidf_vectorizer.pkl"
        with Path(vectorizer_path).open("wb") as f:
            pickle.dump(vectorizer, f)
        logger.info(f"   ✅ Saved vectorizer: {vectorizer_path}")

        # Save document vectors
        vectors_path = self.output_dir / "tfidf_document_vectors.pkl"
        with Path(vectors_path).open("wb") as f:
            pickle.dump(document_vectors, f)
        logger.info(f"   ✅ Saved vectors: {vectors_path}")

        # Save document IDs
        ids_path = self.output_dir / "tfidf_document_ids.pkl"
        with Path(ids_path).open("wb") as f:
            pickle.dump(doc_ids, f)
        logger.info(f"   ✅ Saved document IDs: {ids_path}")

    async def build(self) -> None:
        """Build complete TF-IDF corpus."""

        logger.info("=" * 80)
        logger.info("BUILD TF-IDF CORPUS")
        logger.info("=" * 80)
        logger.info("")

        # Fetch documents
        documents = await self.fetch_all_documents()

        if not documents:
            logger.error("❌ No documents to process")
            return

        # Build model
        vectorizer, document_vectors, doc_ids = self.build_tfidf_model(documents)

        # Save model
        self.save_model(vectorizer, document_vectors, doc_ids)

        logger.info("")
        logger.info("=" * 80)
        logger.info("BUILD COMPLETE")
        logger.info("=" * 80)
        logger.info("")
        logger.info("Usage in Adaptive RAG:")
        logger.info(
            "1. Load vectorizer: pickle.load(open('data/tfidf/tfidf_vectorizer.pkl', 'rb'))"
        )
        logger.info(
            "2. Load vectors: pickle.load(open('data/tfidf/tfidf_document_vectors.pkl', 'rb'))"
        )
        logger.info("3. Transform query: vectorizer.transform([query])")
        logger.info("4. Calculate similarity: cosine_similarity(query_vector, document_vectors)")


async def main() -> None:
    """Main entry point."""

    builder = TFIDFCorpusBuilder()
    await builder.build()


if __name__ == "__main__":
    asyncio.run(main())
