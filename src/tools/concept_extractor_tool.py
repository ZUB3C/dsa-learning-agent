"""
Concept Extractor Tool using KeyBERT and spaCy.
"""

import logging
from typing import Any

from src.config import get_settings
from src.exceptions import ToolExecutionError
from src.tools.base_tool import BaseTool, Document, ToolResult

logger = logging.getLogger(__name__)


class ConceptExtractorTool(BaseTool):
    """
    Extract key concepts from text using KeyBERT and/or spaCy.

    Methods:
    - KeyBERT: Keyword extraction using BERT embeddings
    - spaCy: Named Entity Recognition (NER)
    - Hybrid: Combination of both
    """

    name = "extract_concepts"
    description = """
    Извлечение ключевых концепций из текста.

    Params:
      text (str): текст для анализа
      method (str): "auto" | "keybert" | "spacy" | "hybrid"
      top_n (int): количество концепций (default: 10)
      language (str): язык текста (default: "ru")

    Returns:
      ToolResult with concepts as metadata
    """

    def __init__(self) -> None:
        super().__init__()
        self.settings = get_settings()
        self._keybert = None
        self._spacy = None

    async def execute(self, params: dict[str, Any]) -> ToolResult:
        """Execute concept extraction."""
        text = params.get("text", "")
        method = params.get("method", "auto")
        top_n = params.get("top_n", self.settings.concept_extraction.concept_keybert_top_n)
        params.get("language", "ru")

        if not text:
            return ToolResult(success=False, documents=[], error="Text parameter is required")

        logger.info(f"🔍 Extracting concepts: method={method}, top_n={top_n}")

        # Auto-select method
        if method == "auto":
            if self.settings.concept_extraction.concept_keybert_enabled:
                method = "keybert"
            elif self.settings.concept_extraction.concept_spacy_enabled:
                method = "spacy"
            else:
                method = "heuristic"

        # Extract concepts
        try:
            if method == "keybert":
                concepts = await self._extract_keybert(text, top_n)
            elif method == "spacy":
                concepts = await self._extract_spacy(text, top_n)
            elif method == "hybrid":
                keybert_concepts = await self._extract_keybert(text, top_n)
                spacy_concepts = await self._extract_spacy(text, top_n)
                concepts = self._merge_concepts(keybert_concepts, spacy_concepts, top_n)
            else:
                # Fallback: heuristic
                concepts = self._extract_heuristic(text, top_n)

            logger.info(f"✅ Extracted {len(concepts)} concepts")

        except Exception as e:
            logger.warning(f"⚠️ Concept extraction failed: {e}, using heuristic fallback")
            concepts = self._extract_heuristic(text, top_n)

        # Create document with concepts
        doc = Document(
            page_content=text[:500],  # Snippet
            metadata={"concepts": concepts, "method": method, "count": len(concepts)},
            source="concept_extraction",
        )

        return ToolResult(
            success=True,
            documents=[doc],
            metadata={"concepts": concepts, "method_used": method, "count": len(concepts)},
        )

    async def _extract_keybert(self, text: str, top_n: int) -> list[str]:
        """
        Extract concepts using KeyBERT.

        Fallback: Install keybert if not available
        """

        if not self._keybert:
            try:
                from keybert import KeyBERT

                model_name = self.settings.concept_extraction.concept_keybert_model
                self._keybert = KeyBERT(model=model_name)
                logger.info(f"✅ KeyBERT initialized with {model_name}")

            except ImportError:
                logger.exception("❌ KeyBERT not installed. Install: pip install keybert")
                msg = "extract_concepts"
                raise ToolExecutionError(msg, "KeyBERT not available", 0)

        # Extract keywords
        keywords = self._keybert.extract_keywords(
            text,
            keyphrase_ngram_range=(1, 2),
            stop_words="russian",
            top_n=top_n,
            use_mmr=True,
            diversity=0.5,
        )

        # Return just the words (not scores)
        return [kw[0] for kw in keywords]

    async def _extract_spacy(self, text: str, top_n: int) -> list[str]:
        """
        Extract concepts using spaCy NER.

        Fallback: Install spacy if not available
        """

        if not self._spacy:
            try:
                import spacy

                model_name = self.settings.concept_extraction.concept_spacy_model

                try:
                    self._spacy = spacy.load(model_name)
                    logger.info(f"✅ spaCy initialized with {model_name}")
                except OSError:
                    logger.warning(f"⚠️ spaCy model {model_name} not found, downloading...")
                    import subprocess

                    subprocess.run(["python", "-m", "spacy", "download", model_name], check=False)
                    self._spacy = spacy.load(model_name)

            except ImportError:
                logger.exception("❌ spaCy not installed. Install: pip install spacy")
                msg = "extract_concepts"
                raise ToolExecutionError(msg, "spaCy not available", 0)

        # Process text
        doc = self._spacy(text[:1000000])  # spaCy has text length limits

        # Extract entities
        entity_types = self.settings.concept_extraction.concept_spacy_entity_types

        concepts = []
        seen = set()

        for ent in doc.ents:
            if ent.label_ in entity_types and ent.text not in seen:
                concepts.append(ent.text)
                seen.add(ent.text)

                if len(concepts) >= top_n:
                    break

        # Also extract noun chunks if not enough entities
        if len(concepts) < top_n:
            for chunk in doc.noun_chunks:
                if chunk.text not in seen and len(chunk.text.split()) <= 3:
                    concepts.append(chunk.text)
                    seen.add(chunk.text)

                    if len(concepts) >= top_n:
                        break

        return concepts[:top_n]

    def _merge_concepts(
        self, keybert_concepts: list[str], spacy_concepts: list[str], top_n: int
    ) -> list[str]:
        """
        Merge concepts from KeyBERT and spaCy.

        Strategy:
        - Keep unique concepts
        - Use fuzzy matching to avoid duplicates
        - Prioritize KeyBERT concepts
        """

        merged = []
        seen_lower = set()

        # Add KeyBERT concepts first
        for concept in keybert_concepts:
            concept_lower = concept.lower()
            if concept_lower not in seen_lower:
                merged.append(concept)
                seen_lower.add(concept_lower)

        # Add spaCy concepts (if not duplicate)
        for concept in spacy_concepts:
            concept_lower = concept.lower()

            # Check for fuzzy duplicates
            is_duplicate = False
            for seen in seen_lower:
                if self._is_similar(concept_lower, seen):
                    is_duplicate = True
                    break

            if not is_duplicate:
                merged.append(concept)
                seen_lower.add(concept_lower)

        return merged[:top_n]

    def _is_similar(self, s1: str, s2: str, threshold: float = 0.85) -> bool:
        """Check if two strings are similar (fuzzy match)."""

        threshold = self.settings.concept_extraction.concept_fuzzy_threshold

        # Simple Jaccard similarity
        words1 = set(s1.split())
        words2 = set(s2.split())

        if not words1 or not words2:
            return False

        intersection = len(words1 & words2)
        union = len(words1 | words2)

        similarity = intersection / union if union > 0 else 0.0

        return similarity >= threshold

    def _extract_heuristic(self, text: str, top_n: int) -> list[str]:
        """
        Heuristic concept extraction (fallback).

        Strategy:
        - Look for known algorithm/DS terms
        - Extract capitalized phrases
        - Remove stopwords
        """

        # Common algorithm/data structure terms
        known_concepts = [
            "сортировка",
            "поиск",
            "дерево",
            "граф",
            "хеш",
            "таблица",
            "стек",
            "очередь",
            "список",
            "массив",
            "алгоритм",
            "сложность",
            "O(n)",
            "рекурсия",
            "итерация",
            "динамическое программирование",
            "жадный алгоритм",
            "BFS",
            "DFS",
            "Дейкстра",
            "Прим",
            "Краскал",
            "быстрая сортировка",
            "пирамидальная сортировка",
            "двоичное дерево",
            "AVL",
            "красно-черное дерево",
            "хеш-таблица",
            "связный список",
        ]

        text_lower = text.lower()

        found_concepts = []

        for concept in known_concepts:
            if concept.lower() in text_lower:
                found_concepts.append(concept)

                if len(found_concepts) >= top_n:
                    break

        # If not enough, extract capitalized phrases
        if len(found_concepts) < top_n:
            import re

            # Find phrases starting with capital letter
            capitalized = re.findall(r"\b[А-ЯA-Z][а-яa-z]+(?:\s+[А-ЯA-Z][а-яa-z]+)*\b", text)

            for phrase in capitalized:
                if phrase not in found_concepts and len(phrase.split()) <= 3:
                    found_concepts.append(phrase)

                    if len(found_concepts) >= top_n:
                        break

        return found_concepts[:top_n]
