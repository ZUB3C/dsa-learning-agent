"""
Web Search Tool using 4get meta-search engine.
Code from Section 7.1 of architecture.
"""

import logging
import operator
import time
from typing import Any
from urllib.parse import quote_plus

import aiohttp

from src.config import get_settings
from src.exceptions import WebSearchUnavailableError
from src.tools.base_tool import BaseTool, Document, ToolResult

logger = logging.getLogger(__name__)


class WebSearchTool(BaseTool):
    """
    Web search using 4get meta-search engine.

    Features:
    - Multiple scraper support (google, bing, duckduckgo)
    - Fallback instances
    - Domain filtering and prioritization
    - Deduplication
    - Optional content scraping
    """

    name = "web_search"
    description = """
    Поиск в интернете через 4get meta-search engine.

    Params:
      query (str): поисковый запрос
      num_results (int): количество результатов (default: 5)
      scrape_content (bool): загружать полный контент страниц (default: True)

    Returns:
      ToolResult with search results (and optionally full content)
    """

    def __init__(self) -> None:
        super().__init__()
        self.settings = get_settings()

        # Build instance list
        self.instances = [
            (
                self.settings.web_search.web_search_base_url,
                self.settings.web_search.web_search_scraper,
            )
        ]

        # Add fallbacks
        for url, scraper in zip(
            self.settings.web_search.web_search_fallback_urls,
            self.settings.web_search.web_search_fallback_scrapers,
            strict=False,
        ):
            self.instances.append((url, scraper))

        logger.info(f"📡 Web Search initialized with {len(self.instances)} instances")

    async def execute(self, params: dict[str, Any]) -> ToolResult:
        """Execute web search."""
        query = params.get("query", "")
        num_results = params.get("num_results", self.settings.web_search.web_search_results_limit)
        scrape_content = params.get("scrape_content", True)

        if not query:
            return ToolResult(success=False, documents=[], error="Query parameter is required")

        start_time = time.time()

        # Optimize query
        optimized_query = self._optimize_query(query)

        logger.info(f"🔍 Web Search: query='{optimized_query[:50]}...', results={num_results}")

        # ═══════════════════════════════════════════════════════════
        # STEP 1: SEARCH WITH FALLBACKS
        # ═══════════════════════════════════════════════════════════

        search_results = None
        used_instance = None

        for instance_url, scraper in self.instances:
            try:
                search_results = await self._search_instance(
                    instance_url=instance_url,
                    scraper=scraper,
                    query=optimized_query,
                    limit=num_results,
                )
                used_instance = (instance_url, scraper)
                logger.info(f"✅ Search successful via {instance_url} ({scraper})")
                break

            except Exception as e:
                logger.warning(f"⚠️ Search failed on {instance_url}: {e}")
                continue

        if not search_results:
            return ToolResult(
                success=False,
                documents=[],
                error="All search instances failed",
                metadata={"attempted_instances": len(self.instances)},
            )

        # ═══════════════════════════════════════════════════════════
        # STEP 2: FILTER & PRIORITIZE
        # ═══════════════════════════════════════════════════════════

        filtered_results = self._filter_and_prioritize(search_results)

        # ═══════════════════════════════════════════════════════════
        # STEP 3: DEDUPLICATE
        # ═══════════════════════════════════════════════════════════

        deduped_results = self._deduplicate_results(filtered_results)

        logger.info(
            f"📊 Search: {len(search_results)} → {len(filtered_results)} → {len(deduped_results)} results"
        )

        # ═══════════════════════════════════════════════════════════
        # STEP 4: SCRAPE CONTENT (Optional)
        # ═══════════════════════════════════════════════════════════

        documents = []

        if scrape_content and deduped_results:
            # Import scraper tool
            from src.tools.web_scraper_tool import WebScraperTool

            scraper_tool = WebScraperTool()

            urls = [r["url"] for r in deduped_results[:num_results]]

            try:
                scrape_result = await scraper_tool.execute({"urls": urls})

                if scrape_result.success:
                    documents = scrape_result.documents
                    logger.info(f"✅ Scraped {len(documents)} pages")
                else:
                    logger.warning("⚠️ Scraping failed, using snippets only")
                    # Fallback: use snippets
                    documents = self._create_documents_from_snippets(deduped_results[:num_results])

            except Exception as e:
                logger.exception(f"❌ Scraping error: {e}, using snippets")
                documents = self._create_documents_from_snippets(deduped_results[:num_results])
        else:
            # No scraping, use snippets
            documents = self._create_documents_from_snippets(deduped_results[:num_results])

        execution_time = (time.time() - start_time) * 1000  # ms

        return ToolResult(
            success=len(documents) > 0,
            documents=documents,
            metadata={
                "instance_used": used_instance[0] if used_instance else None,
                "scraper_used": used_instance[1] if used_instance else None,
                "results_count": len(documents),
                "content_scraped": scrape_content,
                "execution_time_ms": execution_time,
            },
            execution_time_ms=execution_time,
        )

    async def _search_instance(
        self, instance_url: str, scraper: str, query: str, limit: int
    ) -> list[dict[str, Any]]:
        """
        Execute search on a specific 4get instance.

        Args:
            instance_url: Base URL of 4get instance
            scraper: Scraper to use (google, bing, duckduckgo)
            query: Search query
            limit: Number of results

        Returns:
            List of search results
        """

        # Build search URL
        encoded_query = quote_plus(query)
        search_url = f"{instance_url}/api/v1/search?s={scraper}&q={encoded_query}&limit={limit}"

        timeout = aiohttp.ClientTimeout(total=self.settings.web_search.web_search_timeout_s)

        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(search_url) as response:
                if response.status != 200:
                    raise WebSearchUnavailableError(f"HTTP {response.status}")

                data = await response.json()

                # Parse 4get response format
                # Format: {"web": [{"title": "...", "url": "...", "description": "..."}]}
                return data.get("web", [])

    def _optimize_query(self, query: str) -> str:
        """
        Optimize search query.

        - Add context suffix if enabled
        - Clean special characters
        """

        optimized = query.strip()

        if self.settings.web_search.web_search_add_context:
            # Add context for better results
            context = self.settings.web_search.web_search_context_suffix
            optimized = f"{optimized} {context}"

        return optimized

    def _filter_and_prioritize(self, results: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """
        Filter and prioritize search results by domain.

        - Remove blacklisted domains
        - Boost priority domains
        """

        filtered = []

        for result in results:
            url = result.get("url", "")

            # Check blacklist
            if any(
                blacklisted in url for blacklisted in self.settings.web_search.web_search_blacklist
            ):
                logger.debug(f"⚫ Blacklisted: {url}")
                continue

            # Calculate priority score
            priority = 1.0

            if ".edu" in url:
                priority *= self.settings.web_search.web_search_priority_edu
            elif ".org" in url:
                priority *= self.settings.web_search.web_search_priority_org
            elif ".gov" in url:
                priority *= self.settings.web_search.web_search_priority_gov
            elif "wikipedia.org" in url:
                priority *= self.settings.web_search.web_search_priority_wiki
            elif "habr.com" in url:
                priority *= self.settings.web_search.web_search_priority_habr
            elif "vc.ru" in url:
                priority *= self.settings.web_search.web_search_priority_vc
            elif "stackoverflow.com" in url:
                priority *= self.settings.web_search.web_search_priority_stackoverflow
            elif ".com" in url:
                priority *= self.settings.web_search.web_search_priority_com

            result["_priority"] = priority
            filtered.append(result)

        # Sort by priority
        filtered.sort(key=operator.itemgetter("_priority"), reverse=True)

        return filtered

    def _deduplicate_results(self, results: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """
        Deduplicate search results by URL and title similarity.
        """

        seen_urls = set()
        seen_titles = set()
        deduped = []

        for result in results:
            url = result.get("url", "")
            title = result.get("title", "").lower()

            # Skip exact URL duplicates
            if url in seen_urls:
                continue

            # Skip very similar titles
            is_duplicate = False
            for seen_title in seen_titles:
                # Simple similarity check
                if (
                    self._calculate_similarity(title, seen_title)
                    > self.settings.web_search.web_search_deduplicate_threshold
                ):
                    is_duplicate = True
                    break

            if is_duplicate:
                continue

            seen_urls.add(url)
            seen_titles.add(title)
            deduped.append(result)

        return deduped

    def _calculate_similarity(self, s1: str, s2: str) -> float:
        """Simple Jaccard similarity."""
        words1 = set(s1.split())
        words2 = set(s2.split())

        if not words1 or not words2:
            return 0.0

        intersection = len(words1 & words2)
        union = len(words1 | words2)

        return intersection / union if union > 0 else 0.0

    def _create_documents_from_snippets(self, results: list[dict[str, Any]]) -> list[Document]:
        """Create documents from search result snippets."""

        documents = []

        for result in results:
            doc = Document(
                page_content=result.get("description", ""),
                metadata={
                    "title": result.get("title", ""),
                    "url": result.get("url", ""),
                    "source": "web_search_snippet",
                },
                source=result.get("url", ""),
            )
            documents.append(doc)

        return documents
