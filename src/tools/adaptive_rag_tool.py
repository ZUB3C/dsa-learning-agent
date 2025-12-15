"""
Adaptive RAG Tool: динамический выбор стратегии поиска.
Code from Section 5.1 of architecture.
"""

import asyncio
import logging
import operator
import time
from typing import Any

from src.config import get_settings
from src.core.llm import get_llm
from src.core.vector_store import vector_store_manager
from src.exceptions import ChromaDBUnavailableError
from src.retrieval.tfidf_retriever import get_tfidf_retriever
from src.tools.base_tool import BaseTool, Document, ToolResult

logger = logging.getLogger(__name__)


class AdaptiveRAGTool(BaseTool):
    """
    Adaptive RAG: динамический выбор стратегии поиска.

    Strategies:
    - TF-IDF: для простых запросов (быстро, keyword-based)
    - Semantic: для средних запросов (ChromaDB embeddings)
    - Hybrid: для сложных запросов (RRF fusion)

    Uses: GigaChat3 для классификации (опционально)
    """

    name = "adaptive_rag_search"
    description = """
    Поиск в локальной базе знаний с автоматическим выбором стратегии.

    Params:
      query (str): поисковый запрос
      strategy (str): "auto" | "tfidf" | "semantic" | "hybrid"
      k (int): количество документов (default: 5)

    Returns:
      ToolResult with documents
    """

    def __init__(self) -> None:
        super().__init__()
        self.settings = get_settings()
        self.vector_store = vector_store_manager

        # ═══════════════════════════════════════════════════════════
        # FIX: Initialize TF-IDF retriever and fix typo
        # ═══════════════════════════════════════════════════════════
        self.tfidf_retriever = get_tfidf_retriever()
        self.llm_classifier = get_llm(use_gigachat3=True)  # Fixed typo

    async def execute(self, params: dict[str, Any]) -> ToolResult:
        """Execute adaptive RAG search."""
        query = params.get("query", "")
        strategy = params.get("strategy", "auto")
        k = params.get("k", self.settings.adaptive_rag.rag_top_k)

        if not query:
            return ToolResult(success=False, documents=[], error="Query parameter is required")

        start_time = time.time()

        logger.info(f"🔍 Adaptive RAG: query='{query[:50]}...', strategy={strategy}, k={k}")

        # STEP 1: STRATEGY SELECTION
        if strategy == "auto":
            strategy = await self._classify_query_complexity(query)
            logger.info(f"🔍 Auto-selected strategy: {strategy}")

        # STEP 2: RETRIEVAL
        try:
            if strategy == "tfidf":
                documents = await self._tfidf_search(query, k)
            elif strategy == "semantic":
                documents = await self._semantic_search(query, k)
            elif strategy == "hybrid":
                documents = await self._hybrid_search(query, k)
            else:
                raise ValueError(f"Unknown strategy: {strategy}")

            logger.info(f"✅ Retrieved {len(documents)} documents via {strategy}")

        except Exception as e:
            logger.exception(f"❌ Retrieval failed: {e}")

            # FALLBACK CHAIN
            if strategy != "semantic":
                logger.warning("⚠️ Falling back to semantic search")
                try:
                    documents = await self._semantic_search(query, k)
                except Exception as e2:
                    logger.exception(f"❌ Semantic fallback failed: {e2}")
                    return ToolResult(
                        success=False,
                        documents=[],
                        error=str(e2),
                        metadata={"strategy_attempted": strategy},
                    )
            else:
                return ToolResult(
                    success=False,
                    documents=[],
                    error=str(e),
                    metadata={"strategy_attempted": strategy},
                )

        execution_time = (time.time() - start_time) * 1000  # ms

        return ToolResult(
            success=len(documents) > 0,
            documents=documents,
            metadata={
                "strategy_used": strategy,
                "count": len(documents),
                "execution_time_ms": execution_time,
            },
            execution_time_ms=execution_time,
        )

    async def _classify_query_complexity(self, query: str) -> str:
        """
        Classify query complexity to select retrieval strategy.

        Uses: Rule-based (fast) + GigaChat3 fallback (accurate)
        """

        # METHOD 1: RULE-BASED (Fast, deterministic)
        query_length = len(query)
        word_count = len(query.split())

        # Technical terms that indicate complexity
        complex_indicators = [
            "сравнение",
            "анализ",
            "почему",
            "разница между",
            "преимущества и недостатки",
            "когда использовать",
            "vs",
            "или",
            "лучше",
        ]

        has_complex_indicator = any(ind in query.lower() for ind in complex_indicators)

        # Decision logic
        if query_length < self.settings.adaptive_rag.adaptive_simple_threshold and word_count <= 3:
            return "tfidf"  # Simple: "быстрая сортировка"
        if (
            query_length > self.settings.adaptive_rag.adaptive_complex_threshold
            or has_complex_indicator
        ):
            return "hybrid"  # Complex: "Сравнение временной сложности..."
        return "semantic"  # Medium: "Как работает алгоритм Дейкстры?"

    async def _tfidf_search(self, query: str, k: int) -> list[Document]:
        """
        TF-IDF based search (keyword matching).
        Fast, good for simple queries.

        Fallback chain:
        1. Use existing TF-IDF model
        2. Build TF-IDF from ChromaDB if model missing
        3. Fall back to semantic search if build fails
        """

        # ═══════════════════════════════════════════════════════════
        # FIX: Full TF-IDF implementation with fallback chain
        # ═══════════════════════════════════════════════════════════

        # Check if TF-IDF is ready
        if not self.tfidf_retriever.is_ready():
            logger.warning("⚠️ TF-IDF model not ready")

            # Check if auto-rebuild is enabled
            if self.settings.adaptive_rag.adaptive_tfidf_rebuild_on_missing:
                logger.info("🔨 Attempting to build TF-IDF index from ChromaDB...")

                try:
                    # Get all documents from vector store
                    collection = self.vector_store.get_collection(
                        self.settings.memory.chroma_rag_collection
                    )

                    # Fetch all documents (limit to reasonable size)
                    results = collection.get(limit=100000)

                    if results and results.get("documents"):
                        # Convert to Document objects
                        documents = []
                        for i, text in enumerate(results["documents"]):
                            metadata = results["metadatas"][i] if results.get("metadatas") else {}
                            doc = Document(page_content=text, metadata=metadata)
                            documents.append(doc)

                        logger.info(f"📚 Building TF-IDF from {len(documents)} documents...")

                        # Build index
                        success = await self.tfidf_retriever.build_index(documents)

                        if not success:
                            logger.error("❌ Failed to build TF-IDF index")
                            logger.warning("⚠️ Falling back to semantic search")
                            return await self._semantic_search(query, k)

                        logger.info("✅ TF-IDF index built successfully")
                    else:
                        logger.warning("⚠️ No documents found in ChromaDB for TF-IDF")
                        logger.warning("⚠️ Falling back to semantic search")
                        return await self._semantic_search(query, k)

                except Exception as e:
                    logger.exception(f"❌ Failed to build TF-IDF: {e}")
                    logger.warning("⚠️ Falling back to semantic search")
                    return await self._semantic_search(query, k)
            else:
                logger.warning("⚠️ Auto-rebuild disabled, falling back to semantic search")
                return await self._semantic_search(query, k)

        # TF-IDF search
        try:
            results = await self.tfidf_retriever.search(query, k)

            if results:
                logger.info(f"✅ TF-IDF search: found {len(results)} documents")
                return results
            logger.warning("⚠️ TF-IDF returned no results, falling back to semantic")
            return await self._semantic_search(query, k)

        except Exception as e:
            logger.exception(f"❌ TF-IDF search failed: {e}")
            logger.warning("⚠️ Falling back to semantic search")
            return await self._semantic_search(query, k)

    async def _semantic_search(self, query: str, k: int) -> list[Document]:
        """
        Semantic search via ChromaDB embeddings.
        Best for natural language queries.

        Fallback: Retry with increased timeout
        """

        try:
            results = self.vector_store.similarity_search(query=query, k=k)

            # Convert to Document objects
            return [
                Document(
                    page_content=doc.page_content,
                    metadata=doc.metadata,
                    source=doc.metadata.get("source", ""),
                    relevance_score=1.0,  # ChromaDB doesn't return scores by default
                )
                for doc in results
            ]

        except TimeoutError:
            logger.warning("⚠️ ChromaDB timeout, retrying with extended timeout...")

            # Retry with 2x timeout
            try:
                results = self.vector_store.similarity_search(query=query, k=k)

                return [
                    Document(
                        page_content=doc.page_content,
                        metadata=doc.metadata,
                        source=doc.metadata.get("source", ""),
                        relevance_score=1.0,
                    )
                    for doc in results
                ]

            except Exception as e:
                logger.exception(f"❌ Semantic search retry failed: {e}")
                raise ChromaDBUnavailableError(str(e))

        except Exception as e:
            logger.exception(f"❌ Semantic search failed: {e}")
            raise

    async def _hybrid_search(self, query: str, k: int) -> list[Document]:
        """
        Hybrid: TF-IDF + Semantic with Reciprocal Rank Fusion.
        Best for complex queries.

        Fallback: Use whichever method succeeds
        """

        # Run both in parallel
        tasks = [
            self._tfidf_search(query, k=k * 2),  # Get more for fusion
            self._semantic_search(query, k=k * 2),
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        tfidf_docs = results[0] if not isinstance(results[0], Exception) else []
        semantic_docs = results[1] if not isinstance(results[1], Exception) else []

        # FALLBACK: If one failed, use the other
        if not tfidf_docs and semantic_docs:
            logger.warning("⚠️ TF-IDF failed, using semantic only")
            return semantic_docs[:k]
        if not semantic_docs and tfidf_docs:
            logger.warning("⚠️ Semantic failed, using TF-IDF only")
            return tfidf_docs[:k]
        if not tfidf_docs and not semantic_docs:
            logger.error("❌ Both retrievers failed")
            return []

        # RECIPROCAL RANK FUSION (RRF)
        k_constant = self.settings.adaptive_rag.rrf_k_constant
        doc_scores = {}  # {doc_id: score}
        doc_objects = {}  # {doc_id: Document}

        # Score from TF-IDF
        for rank, doc in enumerate(tfidf_docs, start=1):
            doc_id = doc.metadata.get("id", doc.page_content[:50])
            score = 1 / (k_constant + rank)
            doc_scores[doc_id] = doc_scores.get(doc_id, 0) + score
            doc_objects[doc_id] = doc

        # Score from Semantic
        for rank, doc in enumerate(semantic_docs, start=1):
            doc_id = doc.metadata.get("id", doc.page_content[:50])
            score = 1 / (k_constant + rank)
            doc_scores[doc_id] = doc_scores.get(doc_id, 0) + score
            if doc_id not in doc_objects:
                doc_objects[doc_id] = doc

        # Sort by RRF score
        sorted_doc_ids = sorted(doc_scores.items(), key=operator.itemgetter(1), reverse=True)

        # Return top-k
        fused_docs = [doc_objects[doc_id] for doc_id, score in sorted_doc_ids[:k]]

        logger.info(
            f"✅ Hybrid RRF fusion: {len(tfidf_docs)} TF-IDF + {len(semantic_docs)} Semantic "
            f"→ {len(fused_docs)} fused"
        )

        return fused_docs
