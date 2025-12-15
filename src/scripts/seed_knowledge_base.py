"""
Seed knowledge base with AISD materials.
Code from Section 10.3 of architecture.
"""

import logging
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import get_settings
from src.core.vector_store import vector_store_manager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_documents_from_directory(directory: str) -> list[Document]:
    """
    Load documents from directory (PDFs and TXT).

    Args:
        directory: Path to directory with documents

    Returns:
        List of Document objects
    """

    documents = []
    dir_path = Path(directory)

    if not dir_path.exists():
        logger.warning(f"Directory not found: {directory}")
        return documents

    # Load PDFs
    for pdf_file in dir_path.glob("**/*.pdf"):
        logger.info(f"📄 Loading PDF: {pdf_file.name}")
        try:
            loader = PyPDFLoader(str(pdf_file))
            docs = loader.load()

            # Add metadata
            for doc in docs:
                doc.metadata["source"] = "pdf"
                doc.metadata["filename"] = pdf_file.name

            documents.extend(docs)
            logger.info(f"   ✅ Loaded {len(docs)} pages")

        except Exception as e:
            logger.exception(f"   ❌ Failed to load {pdf_file.name}: {e}")

    # Load TXT files
    for txt_file in dir_path.glob("**/*.txt"):
        logger.info(f"📄 Loading TXT: {txt_file.name}")
        try:
            loader = TextLoader(str(txt_file), encoding="utf-8")
            docs = loader.load()

            # Add metadata
            for doc in docs:
                doc.metadata["source"] = "txt"
                doc.metadata["filename"] = txt_file.name

            documents.extend(docs)
            logger.info(f"   ✅ Loaded {len(docs)} documents")

        except Exception as e:
            logger.exception(f"   ❌ Failed to load {txt_file.name}: {e}")

    return documents


def chunk_documents(
    documents: list[Document], chunk_size: int = 1000, chunk_overlap: int = 200
) -> list[Document]:
    """
    Split documents into chunks.

    Args:
        documents: List of documents
        chunk_size: Size of each chunk
        chunk_overlap: Overlap between chunks

    Returns:
        List of chunked documents
    """

    logger.info(f"🔪 Chunking {len(documents)} documents...")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""],
    )

    chunks = []

    for doc in documents:
        doc_chunks = text_splitter.split_documents([doc])
        chunks.extend(doc_chunks)

    logger.info(f"   ✅ Created {len(chunks)} chunks")

    return chunks


def seed_knowledge_base(data_directory: str = "data/knowledge_base") -> None:
    """
    Seed knowledge base with documents.

    Args:
        data_directory: Path to data directory
    """

    logger.info("🌱 Starting knowledge base seeding...")

    settings = get_settings()

    # ═══════════════════════════════════════════════════════════════
    # STEP 1: LOAD DOCUMENTS
    # ═══════════════════════════════════════════════════════════════

    documents = load_documents_from_directory(data_directory)

    if not documents:
        logger.error("❌ No documents found!")
        return

    logger.info(f"📚 Loaded {len(documents)} documents")

    # ═══════════════════════════════════════════════════════════════
    # STEP 2: CHUNK DOCUMENTS
    # ═══════════════════════════════════════════════════════════════

    chunks = chunk_documents(
        documents,
        chunk_size=settings.vector_store.chunk_size,
        chunk_overlap=settings.vector_store.chunk_overlap,
    )

    # ═══════════════════════════════════════════════════════════════
    # STEP 3: ADD TO VECTOR STORE
    # ═══════════════════════════════════════════════════════════════

    logger.info("💾 Adding chunks to vector store...")

    try:
        # Add in batches
        batch_size = 100

        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]

            vector_store_manager.add_documents(batch)

            logger.info(
                f"   ✅ Added batch {i // batch_size + 1}/{(len(chunks) - 1) // batch_size + 1}"
            )

        logger.info(f"🎉 Successfully seeded {len(chunks)} chunks to knowledge base!")

    except Exception as e:
        logger.exception(f"❌ Failed to seed knowledge base: {e}")


if __name__ == "__main__":
    import sys

    data_dir = sys.argv[1] if len(sys.argv) > 1 else "data/knowledge_base"

    seed_knowledge_base(data_dir)
