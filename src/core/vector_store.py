import chromadb
from chromadb.config import Settings as ChromaSettings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

from ..config import settings


class VectorStoreManager:
    """Менеджер для работы с ChromaDB."""

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
        """Добавить документы в векторное хранилище."""
        # Фильтруем сложные метаданные (списки, словари и т.д.)
        filtered_documents = []
        for doc in documents:
            filtered_doc = Document(
                page_content=doc.page_content, metadata=self._clean_metadata(doc.metadata)
            )
            filtered_documents.append(filtered_doc)

        return self.vectorstore.add_documents(filtered_documents)

    @staticmethod
    def _clean_metadata(metadata: dict) -> dict:
        """Очистить метаданные от неподдерживаемых типов."""
        cleaned = {}
        for key, value in metadata.items():
            # Конвертируем списки в строки
            if isinstance(value, list):
                cleaned[key] = ", ".join(str(v) for v in value)
            # Конвертируем словари в строки
            elif isinstance(value, dict):
                cleaned[key] = str(value)
            # Оставляем только примитивные типы
            elif isinstance(value, (str, int, float, bool)) or value is None:
                cleaned[key] = value
            # Все остальное конвертируем в строку
            else:
                cleaned[key] = str(value)

        return cleaned

    def similarity_search(
        self, query: str, k: int = settings.rag_top_k, filter_dict: dict | None = None
    ) -> list[Document]:
        """Поиск похожих документов."""
        return self.vectorstore.similarity_search(query=query, k=k, filter=filter_dict)

    def delete_collection(self) -> None:
        """Удалить коллекцию."""
        self.client.delete_collection(settings.chroma_collection_name)

    def get_collection_info(self) -> dict:
        """Получить информацию о коллекции."""
        try:
            collection = self.client.get_collection(settings.chroma_collection_name)
            return {
                "name": collection.name,
                "count": collection.count(),
                "metadata": collection.metadata,
            }
        except Exception as e:
            return {"error": str(e), "count": 0}

    def collection_exists(self) -> bool:
        """Проверить существование коллекции."""
        try:
            self.client.get_collection(settings.chroma_collection_name)
        except Exception:
            return False
        else:
            return True


# Глобальный экземпляр
vector_store_manager = VectorStoreManager()
