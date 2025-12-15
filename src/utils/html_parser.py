"""
HTML parsing utilities.
"""

import logging
import re
from urllib.parse import urljoin

from selectolax.parser import HTMLParser

logger = logging.getLogger(__name__)


class HTMLContentExtractor:
    """
    Extract clean text content from HTML.
    """

    def __init__(
        self, remove_tags: list[str] | None = None, content_selectors: list[str] | None = None
    ) -> None:
        """
        Initialize HTML extractor.

        Args:
            remove_tags: Tags to remove (e.g., script, style)
            content_selectors: CSS selectors for main content
        """
        self.remove_tags = remove_tags or [
            "script",
            "style",
            "nav",
            "footer",
            "header",
            "aside",
            "iframe",
            "noscript",
        ]
        self.content_selectors = content_selectors or [
            "article",
            "main",
            ".content",
            ".post-content",
            "#content",
            ".entry-content",
        ]

    def extract_text(self, html: str, base_url: str | None = None) -> str:
        """
        Extract clean text from HTML.

        Args:
            html: HTML string
            base_url: Base URL for resolving relative links

        Returns:
            Clean text content
        """

        parser = HTMLParser(html)

        # Remove unwanted tags
        for tag in self.remove_tags:
            for node in parser.css(tag):
                node.decompose()

        # Try content selectors
        text_content = ""

        for selector in self.content_selectors:
            nodes = parser.css(selector)
            if nodes:
                text_content = " ".join(node.text(deep=True) for node in nodes)
                break

        # Fallback to body
        if not text_content:
            body = parser.body
            if body:
                text_content = body.text(deep=True)

        # Clean text
        return self._clean_text(text_content)

    def extract_links(self, html: str, base_url: str | None = None) -> list[str]:
        """
        Extract all links from HTML.

        Args:
            html: HTML string
            base_url: Base URL for resolving relative links

        Returns:
            List of absolute URLs
        """

        parser = HTMLParser(html)
        links = []

        for link in parser.css("a[href]"):
            href = link.attributes.get("href", "")

            if not href or href.startswith("#"):
                continue

            # Resolve relative URLs
            if base_url:
                href = urljoin(base_url, href)

            links.append(href)

        return links

    def extract_metadata(self, html: str) -> dict:
        """
        Extract metadata from HTML (title, description, etc.).

        Args:
            html: HTML string

        Returns:
            Dict with metadata
        """

        parser = HTMLParser(html)
        metadata = {}

        # Title
        title = parser.css_first("title")
        if title:
            metadata["title"] = title.text()

        # Meta description
        description = parser.css_first('meta[name="description"]')
        if description:
            metadata["description"] = description.attributes.get("content", "")

        # Open Graph
        og_title = parser.css_first('meta[property="og:title"]')
        if og_title:
            metadata["og_title"] = og_title.attributes.get("content", "")

        og_desc = parser.css_first('meta[property="og:description"]')
        if og_desc:
            metadata["og_description"] = og_desc.attributes.get("content", "")

        return metadata

    def _clean_text(self, text: str) -> str:
        """Clean extracted text."""
        # Remove excessive whitespace
        text = re.sub(r"\s+", " ", text)

        # Remove multiple newlines
        text = re.sub(r"\n\n+", "\n\n", text)

        # Strip
        return text.strip()
