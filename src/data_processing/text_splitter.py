import re

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


class SmartTextSplitter:
    """Умный разделитель текста с учетом структуры LaTeX документов."""

    def __init__(
        self, chunk_size: int = 1000, chunk_overlap: int = 200, separators: list[str] | None = None
    ) -> None:
        # Сепараторы с учетом структуры LaTeX документов
        if separators is None:
            separators = [
                "\n\n\n",  # Разделы
                "\n\n",  # Параграфы
                "\n",  # Строки
                ". ",  # Предложения
                ", ",  # Фразы
                " ",  # Слова
                "",  # Символы
            ]

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=separators,
            length_function=len,
            is_separator_regex=False,
        )

    def split_section(
        self, section_data: dict, parent_hierarchy: list[str] | None = None
    ) -> list[Document]:
        """Разделить раздел на чанки с метаданными."""
        content = section_data.get("content", "")
        title = section_data.get("title", "Untitled")
        level = section_data.get("level", 1)
        section_type = section_data.get("type", "unknown")
        start_page = section_data.get("start_page", 0)

        if not content:
            return []

        # Разделяем текст на чанки
        chunks = self.splitter.split_text(content)

        # Создаем иерархию для метаданных
        hierarchy = (parent_hierarchy or []) + [title]

        # Создаем документы с метаданными
        documents = []
        for i, chunk in enumerate(chunks):
            # Извлекаем ключевые концепции из чанка
            concepts = self._extract_concepts(chunk)

            metadata = {
                "source": "algobook.pdf",
                "title": title,
                "level": level,
                "type": section_type,
                "page": start_page,
                "chunk_index": i,
                "total_chunks": len(chunks),
                "hierarchy": " > ".join(hierarchy),
                "concepts": ", ".join(concepts),
            }

            doc = Document(page_content=chunk, metadata=metadata)
            documents.append(doc)

        return documents

    @staticmethod
    def _extract_concepts(text: str) -> list[str]:
        """Извлечь ключевые концепции из текста."""
        concepts = []

        # Паттерны для распространенных терминов АиСД
        concept_patterns = [
            r"\b(алгоритм[а-я]*)\b",
            r"\b(сложност[ьи])\b",
            r"\b(O\([^\)]+\))",
            r"\b(дерев[ао])\b",
            r"\b(граф[а-я]*)\b",
            r"\b(массив[а-я]*)\b",
            r"\b(список[а-я]*)\b",
            r"\b(стек[а-я]*)\b",
            r"\b(очеред[ьия])\b",
            r"\b(сортиров[а-я]+)\b",
            r"\b(поиск[а-я]*)\b",
            r"\b(рекурси[яи])\b",
            r"\b(динамическ[а-я]+ программирован[а-я]+)\b",
            r"\b(жадн[а-я]+ алгоритм[а-я]*)\b",
            r"\b(хеш[а-я]*)\b",
            r"\b(куч[аи])\b",
            r"\b(обход[а-я]*)\b",
        ]

        for pattern in concept_patterns:
            matches = re.findall(pattern, text.lower())
            concepts.extend(matches)

        # Удаляем дубликаты и возвращаем первые 10
        return list(set(concepts))[:10]
