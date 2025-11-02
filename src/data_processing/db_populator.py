import logging
from pathlib import Path
from typing import TYPE_CHECKING

from langchain_community.vectorstores.utils import filter_complex_metadata

from ..core.vector_store import VectorStoreManager, vector_store_manager
from .pdf_parser import PDFParser
from .text_splitter import SmartTextSplitter

if TYPE_CHECKING:
    from langchain_core.documents import Document

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DatabasePopulator:
    """Класс для заполнения ChromaDB данными из PDF."""

    def __init__(
        self,
        pdf_path: str | Path,
        vector_store_manager: VectorStoreManager,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ) -> None:
        self.pdf_path = Path(pdf_path)
        self.vector_store = vector_store_manager
        self.text_splitter = SmartTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    def populate(self, *, clear_existing: bool = False) -> dict:
        """Заполнить базу данных материалами из PDF."""
        if clear_existing:
            logger.info("Очистка существующей коллекции...")
            try:
                self.vector_store.delete_collection()
                # Пересоздаем vector store после удаления
                self.vector_store = VectorStoreManager()
            except Exception as e:
                logger.warning("Не удалось удалить коллекцию: %s", e)

        logger.info(f"Начало обработки PDF: {self.pdf_path}")

        # Парсинг PDF
        with PDFParser(self.pdf_path) as parser:
            # Извлекаем оглавление
            logger.info("Извлечение оглавления...")
            toc = parser.extract_toc()
            logger.info(f"Найдено разделов: {len(toc)}")

            # Извлекаем содержимое по разделам
            logger.info("Извлечение содержимого по разделам...")
            sections = parser.extract_content_by_toc()
            logger.info(f"Извлечено разделов с содержимым: {len(sections)}")

        # Разбиваем на чанки и создаем документы
        all_documents: list[Document] = []

        logger.info("Разбиение на чанки...")
        for i, section in enumerate(sections):
            logger.info(
                f"Обработка раздела {i + 1}/{len(sections)}: "
                f"{section['title']} (страница {section['start_page']})"
            )

            # Создаем иерархию для вложенных разделов
            hierarchy = self._build_hierarchy(sections, i)

            # Разбиваем раздел на чанки
            documents = self.text_splitter.split_section(section, hierarchy)
            all_documents.extend(documents)

            logger.info(f"  Создано чанков: {len(documents)}")

        logger.info(f"Всего создано документов: {len(all_documents)}")

        # Фильтруем сложные метаданные перед добавлением
        logger.info("Фильтрация сложных метаданных...")
        filtered_documents = filter_complex_metadata(all_documents)
        logger.info(f"Документов после фильтрации: {len(filtered_documents)}")

        # Добавляем в векторное хранилище
        logger.info("Добавление документов в ChromaDB...")
        try:
            doc_ids = self.vector_store.add_documents(filtered_documents)
            logger.info(f"Успешно добавлено документов: {len(doc_ids)}")

            return {
                "status": "success",
                "total_sections": len(sections),
                "total_documents": len(filtered_documents),
                "document_ids": doc_ids,
            }
        except Exception as e:
            logger.exception("Ошибка при добавлении документов: %s", e)
            return {
                "status": "error",
                "error": str(e),
                "total_sections": len(sections),
                "total_documents": len(filtered_documents),
            }

    @staticmethod
    def _build_hierarchy(sections: list[dict], current_index: int) -> list[str]:
        """Построить иерархию для текущего раздела."""
        current_section = sections[current_index]
        current_level = current_section["level"]

        hierarchy = []

        # Ищем родительские разделы
        for i in range(current_index - 1, -1, -1):
            section = sections[i]
            if section["level"] < current_level:
                hierarchy.insert(0, section["title"])
                current_level = section["level"]
                if current_level == 1:
                    break

        return hierarchy

    def get_statistics(self) -> dict:
        """Получить статистику по базе данных."""
        return self.vector_store.get_collection_info()


def populate_from_pdf(
    pdf_path: str = "algobook.pdf",
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    *,
    clear_existing: bool = False,
) -> dict:
    """
    Удобная функция для заполнения БД из PDF.

    Args:
        pdf_path: Путь к PDF файлу
        clear_existing: Очистить существующие данные
        chunk_size: Размер чанка
        chunk_overlap: Перекрытие чанков

    Returns:
        Словарь со статистикой заполнения

    """
    populator = DatabasePopulator(
        pdf_path=pdf_path,
        vector_store_manager=vector_store_manager,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    return populator.populate(clear_existing=clear_existing)
