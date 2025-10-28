import chromadb
from chromadb.config import Settings as ChromaSettings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

from ..config import settings


class VectorStoreManager:
    """Менеджер для работы с ChromaDB"""

    def __init__(self) -> None:
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        )

        self.client = chromadb.PersistentClient(
            path=settings.chroma_persist_directory,
            settings=ChromaSettings(anonymized_telemetry=False),
        )

        self.vectorstore = Chroma(
            client=self.client,
            collection_name=settings.chroma_collection_name,
            embedding_function=self.embeddings,
        )

    def add_documents(self, documents: list[Document]) -> list[str]:
        """Добавить документы в векторное хранилище"""
        return self.vectorstore.add_documents(documents)

    def similarity_search(
        self, query: str, k: int = settings.rag_top_k, filter_dict: dict | None = None
    ) -> list[Document]:
        """Поиск похожих документов"""
        return self.vectorstore.similarity_search(query=query, k=k, filter=filter_dict)

    def delete_collection(self) -> None:
        """Удалить коллекцию"""
        self.client.delete_collection(settings.chroma_collection_name)


# Глобальный экземпляр
vector_store_manager = VectorStoreManager()
