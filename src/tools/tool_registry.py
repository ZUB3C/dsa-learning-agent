"""
Tool registry for managing available tools.
"""

import logging
from collections.abc import Callable

logger = logging.getLogger(__name__)


class ToolRegistry:
    """
    Registry for managing available tools.
    """

    def __init__(self) -> None:
        """Initialize tool registry."""
        self._tools: dict[str, Callable] = {}
        self._tool_aliases: dict[str, str] = {}
        self._initialized = False
        logger.info("🔧 Tool registry initialized")

    def register_tool(self, name: str, tool: Callable, aliases: list | None = None) -> None:
        """
        Register a tool.

        Args:
            name: Tool name (usually class name)
            tool: Tool callable
            aliases: List of alternative names (snake_case versions)
        """
        self._tools[name] = tool
        logger.info(f"✅ Registered tool: {name}")

        # Register aliases
        if aliases:
            for alias in aliases:
                self._tool_aliases[alias] = name
                logger.debug(f"   - Alias: {alias} -> {name}")

    def get_tool(self, name: str) -> Callable | None:
        """
        Get tool by name or alias.

        Args:
            name: Tool name or alias

        Returns:
            Tool callable or None
        """
        # Lazy initialization
        if not self._initialized:
            self._register_all_tools()

        # Try direct name first
        if name in self._tools:
            return self._tools[name]

        # Try alias
        if name in self._tool_aliases:
            real_name = self._tool_aliases[name]
            return self._tools.get(real_name)

        # Not found
        logger.warning(f"self._tool_aliases={self._tool_aliases}")
        logger.warning(f"self._tools={self._tools}")
        logger.warning(f"⚠️ Tool not found: {name}")
        logger.info(f"Available tools: {list(self._tools.keys())}")
        logger.info(f"Available aliases: {list(self._tool_aliases.keys())}")
        return None

    def get(self, name: str) -> Callable | None:
        """
        Alias for get_tool() for backward compatibility.

        Args:
            name: Tool name or alias

        Returns:
            Tool callable or None
        """
        return self.get_tool(name)

    def list_tools(self) -> list[str]:
        """List all registered tools."""
        if not self._initialized:
            self._register_all_tools()
        return list(self._tools.keys())

    def list_aliases(self) -> dict[str, str]:
        """List all tool aliases."""
        if not self._initialized:
            self._register_all_tools()
        return self._tool_aliases.copy()

    def _register_all_tools(self) -> None:
        """
        Register all available tools with their aliases.
        Called automatically on first use (lazy initialization).
        """
        if self._initialized:
            return

        logger.info("🔄 Auto-registering tools...")

        try:
            from src.tools.adaptive_rag_tool import AdaptiveRAGTool
            from src.tools.concept_extractor_tool import ConceptExtractorTool
            from src.tools.corrective_rag_tool import CorrectiveRAGTool
            from src.tools.memory_retrieval_tool import MemoryRetrievalTool
            from src.tools.web_scraper_tool import WebScraperTool
            from src.tools.web_search_tool import WebSearchTool

            # Register tools with snake_case aliases
            self.register_tool(
                "AdaptiveRAGTool",
                AdaptiveRAGTool(),
                aliases=["adaptive_rag", "adaptive_rag_search", "rag_adaptive"]
            )

            self.register_tool(
                "CorrectiveRAGTool",
                CorrectiveRAGTool(),
                aliases=["corrective_rag", "corrective_check", "rag_corrective"]
            )

            self.register_tool(
                "WebSearchTool",
                WebSearchTool(),
                aliases=["web_search", "search_web", "search"]
            )

            self.register_tool(
                "WebScraperTool",
                WebScraperTool(),
                aliases=["web_scraper", "scrape_web", "fetch_content"]
            )

            self.register_tool(
                "ConceptExtractorTool",
                ConceptExtractorTool(),
                aliases=["concept_extractor", "extract_concepts", "concepts"]
            )

            self.register_tool(
                "MemoryRetrievalTool",
                MemoryRetrievalTool(),
                aliases=["memory_retrieval", "retrieve_memory", "memory_search"]
            )

            self._initialized = True
            logger.info(f"✅ Auto-registered {len(self._tools)} tools with {len(self._tool_aliases)} aliases")

        except Exception as e:
            logger.exception(f"❌ Failed to auto-register tools: {e}")
            raise


# Global registry instance
_registry = ToolRegistry()


def get_tool_registry() -> ToolRegistry:
    """Get global tool registry."""
    return _registry


def register_all_tools() -> None:
    """
    Manually register all available tools.
    (Kept for backward compatibility, but not required due to lazy init)
    """
    registry = get_tool_registry()
    registry._register_all_tools()
