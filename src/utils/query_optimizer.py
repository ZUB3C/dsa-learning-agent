"""
Query optimization utilities.
"""

import logging
import re

logger = logging.getLogger(__name__)


class QueryOptimizer:
    """
    Optimize user queries for better retrieval.
    """

    # Russian stopwords
    STOPWORDS_RU = {
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
        "сейчас",
        "были",
        "куда",
        "зачем",
        "всех",
        "никогда",
        "можно",
        "при",
        "наконец",
        "два",
        "об",
        "другой",
        "хоть",
        "после",
        "над",
        "больше",
        "тот",
        "через",
        "эти",
        "нас",
        "про",
        "всего",
        "них",
        "какая",
        "много",
        "разве",
        "три",
        "эту",
        "моя",
        "впрочем",
        "хорошо",
        "свою",
        "этой",
        "перед",
        "иногда",
        "лучше",
        "чуть",
        "том",
        "нельзя",
        "такой",
        "им",
        "более",
        "всегда",
        "конечно",
        "всю",
        "между",
    }

    # English stopwords
    STOPWORDS_EN = {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "for",
        "from",
        "has",
        "he",
        "in",
        "is",
        "it",
        "its",
        "of",
        "on",
        "that",
        "the",
        "to",
        "was",
        "will",
        "with",
        "this",
        "but",
        "they",
        "have",
        "had",
        "what",
        "when",
        "where",
        "who",
        "which",
        "why",
        "how",
    }

    @classmethod
    def optimize(cls, query: str, language: str = "ru") -> str:
        """
        Optimize query for better retrieval.

        Steps:
        1. Expand common abbreviations
        2. Add context keywords
        3. Remove stopwords (optional)
        4. Normalize

        Args:
            query: Original query
            language: Language (ru/en)

        Returns:
            Optimized query
        """

        # Expand abbreviations
        query = cls._expand_abbreviations(query, language)

        # Add context
        query = cls._add_context(query, language)

        # Normalize
        return cls._normalize(query)

    @classmethod
    def extract_key_terms(cls, query: str, language: str = "ru") -> list[str]:
        """
        Extract key terms from query.

        Args:
            query: Query text
            language: Language (ru/en)

        Returns:
            List of key terms
        """

        stopwords = cls.STOPWORDS_RU if language == "ru" else cls.STOPWORDS_EN

        # Tokenize
        words = re.findall(r"\b\w+\b", query.lower())

        # Filter stopwords
        return [w for w in words if w not in stopwords and len(w) > 2]

    @classmethod
    def expand_query(cls, query: str, synonyms: dict | None = None) -> str:
        """
        Expand query with synonyms.

        Args:
            query: Original query
            synonyms: Dict of {term: [synonyms]}

        Returns:
            Expanded query
        """

        if not synonyms:
            # Default synonyms for AISD domain
            synonyms = {
                "сортировка": ["упорядочивание", "сортинг"],
                "поиск": ["search", "нахождение"],
                "граф": ["graph", "сеть"],
                "дерево": ["tree"],
                "алгоритм": ["algorithm"],
                "сложность": ["complexity", "асимптотика"],
            }

        expanded_terms = [query]

        for term, syns in synonyms.items():
            if term.lower() in query.lower():
                expanded_terms.extend(syns)

        return " ".join(expanded_terms)

    @classmethod
    def _expand_abbreviations(cls, query: str, language: str) -> str:
        """Expand common abbreviations."""

        abbreviations = {
            "ru": {
                "АиСД": "алгоритмы и структуры данных",
                "ДП": "динамическое программирование",
                "БФС": "поиск в ширину",
                "ДФС": "поиск в глубину",
            },
            "en": {
                "BFS": "breadth-first search",
                "DFS": "depth-first search",
                "DP": "dynamic programming",
                "DS": "data structures",
            },
        }

        abbr_dict = abbreviations.get(language, {})

        for abbr, expansion in abbr_dict.items():
            # Case-insensitive replacement
            pattern = re.compile(re.escape(abbr), re.IGNORECASE)
            query = pattern.sub(expansion, query)

        return query

    @classmethod
    def _add_context(cls, query: str, language: str) -> str:
        """Add context keywords if missing."""

        context_keywords = {
            "ru": ["алгоритм", "структура данных", "программирование"],
            "en": ["algorithm", "data structure", "programming"],
        }

        keywords = context_keywords.get(language, [])

        # Check if query already has context
        query_lower = query.lower()
        has_context = any(kw.lower() in query_lower for kw in keywords)

        if not has_context:
            # Add first context keyword
            query = f"{query} {keywords[0]}"

        return query

    @classmethod
    def _normalize(cls, query: str) -> str:
        """Normalize query text."""

        # Remove extra whitespace
        query = re.sub(r"\s+", " ", query)

        # Strip
        return query.strip()
