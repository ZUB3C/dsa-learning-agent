import re
from pathlib import Path
from types import TracebackType

import pymupdf  # PyMuPDF


class PDFParser:
    """Парсер для PDF документов, скомпилированных из LaTeX."""

    def __init__(self, pdf_path: str | Path) -> None:
        self.pdf_path = Path(pdf_path)
        if not self.pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {self.pdf_path}")

        self.doc = pymupdf.open(str(self.pdf_path))
        self.toc: list[tuple[int, str, int]] = []  # (level, title, page)

    def extract_toc(self) -> list[dict[str, any]]:
        """Извлечь оглавление из PDF."""
        # Попытка получить TOC из метаданных PDF
        raw_toc = self.doc.get_toc()

        if raw_toc:
            # PyMuPDF возвращает список [level, title, page]
            self.toc = [(level, title, page) for level, title, page in raw_toc]
        else:
            # Если TOC не встроен, пытаемся извлечь вручную
            self.toc = self._extract_toc_manually()

        # Конвертируем в структурированный формат
        structured_toc = []
        for level, title, page in self.toc:
            structured_toc.append({
                "level": level,
                "title": title.strip(),
                "page": page,
                "type": self._classify_heading(level, title),
            })

        return structured_toc

    def _extract_toc_manually(self) -> list[tuple[int, str, int]]:
        """Извлечь оглавление вручную из первых страниц."""
        toc = []
        # Обычно оглавление на первых 5-10 страницах
        toc_pages = min(10, len(self.doc))

        for page_num in range(toc_pages):
            page = self.doc[page_num]
            text = page.get_text()

            # Ищем паттерны оглавления
            # Формат: "Номер Название ... Страница"
            # Пример: "1.1 Введение в алгоритмы . . . . . . . . 5"
            lines = text.split("\n")

            for line in lines:
                # Паттерн для лекций (уровень 1)
                lecture_match = re.match(
                    r"^Лекция\s+(\d+)[\.\s]+(.*?)\s+\.{2,}\s*(\d+)", line, re.IGNORECASE
                )
                if lecture_match:
                    title = f"Лекция {lecture_match.group(1)}: {lecture_match.group(2).strip()}"
                    page = int(lecture_match.group(3))
                    toc.append((1, title, page))
                    continue

                # Паттерн для разделов (уровень 2)
                section_match = re.match(r"^(\d+)\.(\d+)[\.\s]+(.*?)\s+\.{2,}\s*(\d+)", line)
                if section_match:
                    title = f"{section_match.group(1)}.{section_match.group(2)} {section_match.group(3).strip()}"  # noqa: E501
                    page = int(section_match.group(4))
                    toc.append((2, title, page))
                    continue

                # Паттерн для подразделов (уровень 3)
                subsection_match = re.match(
                    r"^(\d+)\.(\d+)\.(\d+)[\.\s]+(.*?)\s+\.{2,}\s*(\d+)", line
                )
                if subsection_match:
                    title = (
                        f"{subsection_match.group(1)}.{subsection_match.group(2)}."
                        f"{subsection_match.group(3)} {subsection_match.group(4).strip()}"
                    )
                    page = int(subsection_match.group(5))
                    toc.append((3, title, page))
                    continue

        return toc

    @staticmethod
    def _classify_heading(level: int, title: str) -> str:
        """Классифицировать тип заголовка."""
        title_lower = title.lower()

        if "лекция" in title_lower or level == 1:
            return "lecture"
        if level == 2 or re.match(r"^\d+\.", title):
            return "section"
        if level >= 3:
            return "subsection"
        return "unknown"

    def extract_content_by_toc(self) -> list[dict[str, any]]:
        """Извлечь содержимое по разделам из оглавления."""
        if not self.toc:
            self.extract_toc()

        sections_content = []

        for i, (level, title, start_page) in enumerate(self.toc):
            # Определяем конечную страницу (начало следующего раздела того же уровня)
            end_page = None

            for j in range(i + 1, len(self.toc)):
                next_level, _, next_page = self.toc[j]
                if next_level <= level:
                    end_page = next_page - 1
                    break

            if end_page is None:
                end_page = len(self.doc)

            # Извлекаем текст
            content = self._extract_text_from_pages(start_page - 1, end_page)

            sections_content.append({
                "level": level,
                "title": title.strip(),
                "start_page": start_page,
                "end_page": end_page,
                "content": content,
                "type": self._classify_heading(level, title),
            })

        return sections_content

    def _extract_text_from_pages(self, start_page: int, end_page: int) -> str:
        """Извлечь текст со страниц."""
        text_parts = []

        for page_num in range(start_page, min(end_page, len(self.doc))):
            page = self.doc[page_num]
            text = page.get_text()

            # Очистка текста
            text = self._clean_text(text)
            text_parts.append(text)

        return "\n\n".join(text_parts)

    @staticmethod
    def _clean_text(text: str) -> str:
        """Очистить текст от артефактов."""
        # Удаляем множественные пробелы
        text = re.sub(r"\s+", " ", text)

        # Удаляем номера страниц (обычно одиночные числа в конце строк)
        text = re.sub(r"\n\s*\d+\s*\n", "\n", text)

        # Восстанавливаем параграфы
        text = re.sub(r"\n\s*\n", "\n\n", text)

        return text.strip()

    def close(self) -> None:
        """Закрыть PDF документ."""
        if self.doc:
            self.doc.close()

    def __enter__(self) -> None:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        exc_traceback: TracebackType | None,
    ) -> None:
        self.close()
