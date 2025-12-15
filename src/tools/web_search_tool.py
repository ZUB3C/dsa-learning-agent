"""
Web Search Tool: поиск через 4get metasearch engine.
Code from Section 7.1 of architecture.
"""

import logging
import time
from typing import Any
from urllib.parse import urlencode

import aiohttp

from src.config import get_settings
from src.tools.base_tool import BaseTool, ToolResult

logger = logging.getLogger(__name__)


class WebSearchTool(BaseTool):
    """
    Web Search через 4get metasearch.

    Uses: No LLM (pure API calls)
    Fallback chain: Primary → Fallback instances → Cache
    """

    name = "web_search"
    description = """
    Поиск в интернете через 4get metasearch engine.

    Params:
      query (str): поисковый запрос
      num_results (int): количество результатов (default: 5)
      scrape_content (bool): скачивать полный контент страниц (default: True)

    Returns:
      ToolResult with search results (title, url, snippet)
    """

    def __init__(self) -> None:
        super().__init__()
        self.settings = get_settings()

        # ═══════════════════════════════════════════════════════════
        # FIX: Get instances from settings
        # ═══════════════════════════════════════════════════════════

        # Primary instance from settings
        self.primary_url = self.settings.web_search.web_search_base_url

        # Fallback instances from settings
        self.fallback_urls = self.settings.web_search.web_search_fallback_urls

        # Domain priorities
        self.domain_priorities = {
            ".edu": self.settings.web_search.web_search_priority_edu,
            ".org": self.settings.web_search.web_search_priority_org,
            ".gov": self.settings.web_search.web_search_priority_gov,
            "wikipedia.org": self.settings.web_search.web_search_priority_wiki,
            "habr.com": self.settings.web_search.web_search_priority_habr,
            "stackoverflow.com": self.settings.web_search.web_search_priority_stackoverflow,
            ".com": self.settings.web_search.web_search_priority_com,
            ".ru": self.settings.web_search.web_search_priority_ru,
        }

        # Blacklist
        self.blacklist = set(self.settings.web_search.web_search_blacklist)

        logger.info("🔍 Web Search Tool initialized:")
        logger.info(f"   - Primary: {self.primary_url}")
        logger.info(f"   - Fallbacks: {self.fallback_urls}")

    async def execute(self, params: dict[str, Any]) -> ToolResult:
        """Execute web search."""

        query = params.get("query", "")
        num_results = params.get(
            "num_results",
            self.settings.web_search.web_search_results_limit
        )
        scrape_content = params.get("scrape_content", True)

        if not query:
            return ToolResult(success=False, documents=[], error="Query is required")

        start_time = time.time()

        # Trim query to max length
        query = query[:200]

        # Add context if enabled
        if self.settings.web_search.web_search_add_context:
            context_suffix = self.settings.web_search.web_search_context_suffix
            if context_suffix and context_suffix not in query:
                query = f"{query} {context_suffix}"

        logger.info(f"🔍 Web Search: query='{query[:80]}...', results={num_results}")

        # ═══════════════════════════════════════════════════════════
        # STEP 1: Try search with fallback chain
        # ═══════════════════════════════════════════════════════════

        search_results = await self._search_with_fallback(query, num_results)

        if not search_results:
            execution_time = (time.time() - start_time) * 1000
            return ToolResult(
                success=False,
                documents=[],
                error="All search instances failed",
                execution_time_ms=execution_time,
            )

        logger.info(f"✅ Found {len(search_results)} search results")

        # ═══════════════════════════════════════════════════════════
        # STEP 2: Filter blacklisted domains
        # ═══════════════════════════════════════════════════════════

        filtered_results = []
        for result in search_results:
            url = result.get("url", "")

            # Check blacklist
            if any(bl in url for bl in self.blacklist):
                logger.debug(f"⛔ Blacklisted: {url}")
                continue

            # Calculate priority score
            priority = 1.0
            for domain, score in self.domain_priorities.items():
                if domain in url:
                    priority = score
                    break

            result["priority_score"] = priority
            filtered_results.append(result)

        # Sort by priority
        filtered_results.sort(key=lambda r: r.get("priority_score", 1.0), reverse=True)

        logger.info(
            f"📊 Filtered: {len(search_results)} → {len(filtered_results)} "
            f"(removed {len(search_results) - len(filtered_results)} blacklisted)"
        )

        # ═══════════════════════════════════════════════════════════
        # STEP 3: Scrape content if requested
        # ═══════════════════════════════════════════════════════════

        documents = []

        if scrape_content:
            # Import scraper
            from src.tools.web_scraper_tool import WebScraperTool

            scraper = WebScraperTool()

            # Extract URLs
            urls = [result["url"] for result in filtered_results]

            # Scrape
            scrape_result = await scraper.execute(
                {"urls": urls, "extract_text": True, "extract_metadata": True}
            )

            documents = scrape_result.documents

            logger.info(f"📄 Scraped {len(documents)}/{len(urls)} pages")

        else:
            # Return search snippets as documents
            from src.tools.base_tool import Document

            for result in filtered_results:
                doc = Document(
                    page_content=result.get("description", ""),
                    metadata={
                        "source": "web_search",
                        "url": result.get("url"),
                        "title": result.get("title"),
                        "priority_score": result.get("priority_score", 1.0),
                    },
                )
                documents.append(doc)

        execution_time = (time.time() - start_time) * 1000

        return ToolResult(
            success=len(documents) > 0,
            documents=documents,
            metadata={
                "query": query,
                "total_results": len(search_results),
                "filtered_results": len(filtered_results),
                "scraped_pages": len(documents),
            },
            execution_time_ms=execution_time,
        )

    async def _search_with_fallback(
        self, query: str, limit: int = 5
    ) -> list[dict[str, Any]]:
        """
        Search with fallback chain.

        Tries:
        1. Primary instance
        2. Fallback instances (in order)

        Returns:
            List of search results or empty list
        """

        # Try primary instance
        try:
            results = await self._search_4get(
                base_url=self.primary_url, query=query, limit=limit
            )

            if results:
                logger.info(f"✅ Search successful on primary: {self.primary_url}")
                return results

        except Exception as e:
            logger.warning(f"⚠️ Primary search failed on {self.primary_url}: {e}")

        # Try fallback instances
        for fallback_url in self.fallback_urls:
            try:
                results = await self._search_4get(
                    base_url=fallback_url, query=query, limit=limit
                )

                if results:
                    logger.info(f"✅ Search successful on fallback: {fallback_url}")
                    return results

            except Exception as e:
                logger.warning(f"⚠️ Fallback search failed on {fallback_url}: {e}")
                continue

        # All failed
        logger.error("❌ All search instances failed")
        return []

    async def _search_4get(
        self, base_url: str, query: str, limit: int = 5
    ) -> list[dict[str, Any]]:
        """
        Search using 4get API.

        Args:
            base_url: 4get instance URL
            query: Search query
            limit: Max results

        Returns:
            List of search results

        Raises:
            Exception: If search fails
        """

        # Build API URL - 4get uses /api/v1/web endpoint
        params = {
            "s": query,  # Search query parameter
            "nsfw": "no",  # Filter NSFW content
        }

        # Construct URL
        url = f"{base_url}/api/v1/web?{urlencode(params)}"

        timeout = aiohttp.ClientTimeout(
            total=self.settings.web_search.web_search_timeout_s
        )

        # Retry logic
        retry_count = self.settings.web_search.web_search_retry_count

        for attempt in range(retry_count + 1):
            try:
                async with aiohttp.ClientSession(timeout=timeout) as session:
                    headers = {
                        "User-Agent": (
                            "Mozilla/5.0 (compatible; MaterialsAgent/2.0; "
                            "+https://example.com/bot)"
                        ),
                        "Accept": "application/json",
                    }

                    async with session.get(url, headers=headers) as response:
                        if response.status == 404:
                            logger.error(
                                f"Search endpoint not found - HTTP {response.status}"
                            )
                            raise Exception(
                                f"4get API endpoint unavailable (HTTP {response.status})"
                            )

                        if response.status != 200:
                            text = await response.text()
                            logger.error(
                                f"Search request failed - HTTP {response.status}: "
                                f"{text[:200]}"
                            )

                            if attempt < retry_count:
                                logger.info(f"🔄 Retrying... (attempt {attempt + 1}/{retry_count})")
                                continue

                            raise Exception(f"HTTP {response.status}")

                        data = await response.json()

                        # Parse 4get response
                        results = []

                        # 4get returns results in "web" key
                        if "web" in data and isinstance(data["web"], list):
                            for item in data["web"][:limit]:
                                result = {
                                    "title": item.get("title", ""),
                                    "url": item.get("url", ""),
                                    "description": item.get("description", ""),
                                }
                                results.append(result)

                        logger.info(f"📊 Parsed {len(results)} results from 4get")

                        return results

            except aiohttp.ClientError as e:
                logger.exception(f"Search request failed - {e}")

                if attempt < retry_count:
                    logger.info(f"🔄 Retrying... (attempt {attempt + 1}/{retry_count})")
                    continue

                raise Exception(f"Web search service unavailable: {e}")

            except Exception as e:
                logger.exception(f"Search failed: {e}")

                if attempt < retry_count:
                    logger.info(f"🔄 Retrying... (attempt {attempt + 1}/{retry_count})")
                    continue

                raise

        # Should not reach here
        msg = "All retry attempts failed"
        raise Exception(msg)
