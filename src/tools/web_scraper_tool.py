"""
Web Scraper Tool for fetching full page content.
Code from Section 7.2 of architecture.
"""

import asyncio
import logging
import random
import time
from typing import Any

import aiohttp
from selectolax.parser import HTMLParser

from src.config import get_settings
from src.tools.base_tool import BaseTool, Document, ToolResult

logger = logging.getLogger(__name__)


class WebScraperTool(BaseTool):
    """
    Web scraper for fetching and parsing HTML content.

    Features:
    - Async batch fetching
    - User-Agent rotation
    - Content extraction with selectolax
    - Retry with fallbacks
    - Timeout handling
    """

    name = "web_scraper"
    description = """
    Загрузка и парсинг HTML контента с веб-страниц.

    Params:
      urls (list[str]): список URL для загрузки
      timeout_s (float): таймаут на запрос (default: 5.0)

    Returns:
      ToolResult with extracted text content
    """

    def __init__(self) -> None:
        super().__init__()
        self.settings = get_settings()
        self.user_agents = self.settings.web_scraper.web_content_user_agents

    async def execute(self, params: dict[str, Any]) -> ToolResult:
        """Execute web scraping."""
        urls = params.get("urls", [])
        timeout_s = params.get("timeout_s", self.settings.web_scraper.web_content_timeout_s)

        if not urls:
            return ToolResult(success=False, documents=[], error="URLs parameter is required")

        start_time = time.time()

        logger.info(f"🕷️ Web Scraper: fetching {len(urls)} URLs")

        # ═══════════════════════════════════════════════════════════
        # STEP 1: BATCH FETCH
        # ═══════════════════════════════════════════════════════════

        documents = []

        # Process in batches
        batch_size = self.settings.web_scraper.web_content_batch_size

        for i in range(0, len(urls), batch_size):
            batch_urls = urls[i : i + batch_size]

            batch_results = await asyncio.gather(
                *[self._fetch_and_parse(url, timeout_s) for url in batch_urls],
                return_exceptions=True,
            )

            for url, result in zip(batch_urls, batch_results, strict=False):
                if isinstance(result, Exception):
                    logger.warning(f"⚠️ Failed to fetch {url}: {result}")
                    continue

                if result:
                    documents.append(result)

        logger.info(f"✅ Scraped {len(documents)}/{len(urls)} pages")

        execution_time = (time.time() - start_time) * 1000  # ms

        return ToolResult(
            success=len(documents) > 0,
            documents=documents,
            metadata={
                "requested_urls": len(urls),
                "successful_scrapes": len(documents),
                "failed_scrapes": len(urls) - len(documents),
                "execution_time_ms": execution_time,
            },
            execution_time_ms=execution_time,
        )

    async def _fetch_and_parse(self, url: str, timeout_s: float) -> Document:
        """
        Fetch and parse a single URL.

        Fallback chain:
        1. Standard fetch (5s)
        2. Extended timeout (10s)
        3. Different User-Agent
        4. Cache (if available)
        """

        # Try standard fetch
        try:
            html_content = await self._fetch_url(url, timeout_s)
            text_content = self._extract_text(html_content)

            return Document(
                page_content=text_content,
                metadata={"url": url, "source": "web_scraper", "length": len(text_content)},
                source=url,
            )

        except TimeoutError:
            logger.warning(f"⚠️ Timeout on {url}, retrying with extended timeout")

            # Fallback: Extended timeout
            try:
                extended_timeout = self.settings.web_scraper.web_content_extended_timeout_s
                html_content = await self._fetch_url(url, extended_timeout)
                text_content = self._extract_text(html_content)

                return Document(
                    page_content=text_content,
                    metadata={
                        "url": url,
                        "source": "web_scraper",
                        "length": len(text_content),
                        "fallback": "extended_timeout",
                    },
                    source=url,
                )

            except Exception as e:
                logger.exception(f"❌ Extended timeout failed for {url}: {e}")
                raise

        except Exception as e:
            logger.exception(f"❌ Fetch failed for {url}: {e}")
            raise

    async def _fetch_url(self, url: str, timeout_s: float) -> str:
        """Fetch URL with random User-Agent."""

        headers = {
            "User-Agent": random.choice(self.user_agents),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "ru-RU,ru;q=0.9,en-US;q=0.8,en;q=0.7",
        }

        timeout = aiohttp.ClientTimeout(total=timeout_s)

        async with aiohttp.ClientSession(timeout=timeout, headers=headers) as session:
            async with session.get(url) as response:
                if response.status != 200:
                    raise Exception(f"HTTP {response.status}")

                return await response.text()

    def _extract_text(self, html: str) -> str:
        """
        Extract text content from HTML using selectolax.

        Steps:
        1. Parse HTML
        2. Remove unwanted tags (script, style, nav, etc.)
        3. Extract text from content selectors
        4. Clean and normalize
        """

        parser = HTMLParser(html)

        # Remove unwanted tags
        for tag in self.settings.web_scraper.web_content_remove_tags:
            for node in parser.css(tag):
                node.decompose()

        # Try to find main content
        text_content = ""

        for selector in self.settings.web_scraper.web_content_selectors:
            nodes = parser.css(selector)
            if nodes:
                text_content = " ".join(node.text(deep=True) for node in nodes)
                break

        # Fallback: entire body
        if not text_content:
            body = parser.body
            if body:
                text_content = body.text(deep=True)

        # Clean text
        text_content = self._clean_text(text_content)

        # Truncate if too long
        max_length = self.settings.web_scraper.web_content_max_length
        if len(text_content) > max_length:
            text_content = text_content[:max_length] + "..."

        return text_content

    def _clean_text(self, text: str) -> str:
        """Clean and normalize text."""

        # Remove excessive whitespace
        lines = [line.strip() for line in text.split("\n")]
        lines = [line for line in lines if line]  # Remove empty lines

        text = "\n".join(lines)

        # Replace multiple spaces
        import re

        return re.sub(r" +", " ", text)
