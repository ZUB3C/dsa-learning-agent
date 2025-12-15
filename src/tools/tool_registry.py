"""
Tool Registry for managing and providing tools to orchestrator.
"""

import logging

from src.tools.adaptive_rag_tool import AdaptiveRAGTool
from src.tools.base_tool import BaseTool
from src.tools.concept_extractor_tool import ConceptExtractorTool
from src.tools.corrective_rag_tool import CorrectiveRAGTool
from src.tools.memory_retrieval_tool import MemoryRetrievalTool
from src.tools.web_scraper_tool import WebScraperTool
from src.tools.web_search_tool import WebSearchTool

logger = logging.getLogger(__name__)


class ToolRegistry:
    """
    Registry for all available tools.

    Provides:
    - Tool lookup by name
    - Tool initialization
    - Tool metadata
    """

    def __init__(self) -> None:
        """Initialize tool registry."""
        self._tools: dict[str, BaseTool] = {}
        self._initialize_tools()

    def _initialize_tools(self) -> None:
        """Initialize all tools."""

        tool_classes = [
            AdaptiveRAGTool,
            CorrectiveRAGTool,
            WebSearchTool,
            WebScraperTool,
            ConceptExtractorTool,
            MemoryRetrievalTool,
        ]

        for tool_class in tool_classes:
            try:
                tool = tool_class()
                self._tools[tool.name] = tool
                logger.info(f"✅ Registered tool: {tool.name}")
            except Exception as e:
                logger.exception(f"❌ Failed to initialize {tool_class.__name__}: {e}")

    def get(self, tool_name: str) -> BaseTool | None:
        """
        Get tool by name.

        Args:
            tool_name: Name of the tool

        Returns:
            Tool instance or None if not found
        """

        tool = self._tools.get(tool_name)

        if not tool:
            logger.warning(f"⚠️ Tool not found: {tool_name}")
            logger.info(f"Available tools: {list(self._tools.keys())}")

        return tool

    def list_tools(self) -> dict[str, str]:
        """
        List all available tools.

        Returns:
            Dict of {tool_name: description}
        """

        return {name: tool.description for name, tool in self._tools.items()}

    def has_tool(self, tool_name: str) -> bool:
        """Check if tool exists."""
        return tool_name in self._tools
