import chromadb
from chromadb.config import Settings as ChromaSettings
from langchain_chroma.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

from src.config import get_settings

settings = get_settings()


class VectorStoreManager:
    """Менеджер для работы с ChromaDB."""

    def __init__(self) -> None:
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        )
        self.client = chromadb.PersistentClient(
            path=settings.memory.chroma_persist_directory,
            settings=ChromaSettings(anonymized_telemetry=False),
        )

        self.vectorstore = Chroma(
            client=self.client,
            collection_name=settings.vector_store.collection_name,
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
        self, query: str, k: int = settings.adaptive_rag.rag_top_k, filter_dict: dict | None = None
    ) -> list[Document]:
        """Поиск похожих документов."""
        return self.vectorstore.similarity_search(query=query, k=k, filter=filter_dict)

    def similarity_search_with_score(
        self, query: str, k: int = settings.adaptive_rag.rag_top_k, filter_dict: dict | None = None
    ) -> list[tuple[Document, float]]:
        """Поиск похожих документов."""
        return self.vectorstore.similarity_search_with_score(query=query, k=k, filter=filter_dict)

    def get_collection(self, collection_name: str | None = None) -> chromadb.Collection:
        """
        Получить ChromaDB коллекцию по имени.

        Args:
            collection_name: Имя коллекции (optional, default: from settings)

        Returns:
            ChromaDB Collection object

        Raises:
            ValueError: If collection doesn't exist

        Example:
            >>> collection = vector_store.get_collection("aisd_materials")
            >>> results = collection.get(limit=100)
        """

        if collection_name is None:
            collection_name = settings.vector_store.collection_name

        try:
            return self.client.get_collection(collection_name)
        except Exception as e:
            raise ValueError(f"Collection '{collection_name}' not found: {e}")

    def get_or_create_collection(
        self, collection_name: str, embedding_function=None
    ) -> chromadb.Collection:
        """
        Получить существующую или создать новую коллекцию.

        Args:
            collection_name: Имя коллекции
            embedding_function: Функция эмбеддингов (optional)

        Returns:
            ChromaDB Collection object
        """

        try:
            return self.client.get_collection(collection_name)
        except Exception:
            # Collection doesn't exist, create it
            return self.client.create_collection(
                name=collection_name,
                embedding_function=embedding_function,
            )

    def list_collections(self) -> list[str]:
        """
        Получить список всех коллекций.

        Returns:
            List of collection names
        """

        collections = self.client.list_collections()
        return [col.name for col in collections]

    def delete_collection(self, collection_name: str | None = None) -> None:
        """
        Удалить коллекцию.

        Args:
            collection_name: Имя коллекции (optional, default: from settings)
        """

        if collection_name is None:
            collection_name = settings.vector_store.collection_name

        self.client.delete_collection(collection_name)

    def get_collection_info(self, collection_name: str | None = None) -> dict:
        """
        Получить информацию о коллекции.

        Args:
            collection_name: Имя коллекции (optional, default: from settings)

        Returns:
            Dict with collection info
        """

        if collection_name is None:
            collection_name = settings.vector_store.collection_name

        try:
            collection = self.client.get_collection(collection_name)
            return {
                "name": collection.name,
                "count": collection.count(),
                "metadata": collection.metadata,
            }
        except Exception as e:
            return {"error": str(e), "count": 0}

    def collection_exists(self, collection_name: str | None = None) -> bool:
        """
        Проверить существование коллекции.

        Args:
            collection_name: Имя коллекции (optional, default: from settings)

        Returns:
            True if collection exists
        """

        if collection_name is None:
            collection_name = settings.vector_store.collection_name

        try:
            self.client.get_collection(collection_name)
            return True
        except Exception:
            return False


# Глобальный экземпляр
vector_store_manager = VectorStoreManager()
