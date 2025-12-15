import asyncio
import logging
from typing import Any

import aiohttp
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from pydantic import ValidationError
from selectolax.parser import HTMLParser

from ..config import settings
from ..core.llm import get_llm
from ..core.vector_store import vector_store_manager
from ..models.fourget_models import FourGetResponse

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

MATERIALS_SYSTEM_PROMPT = (
    "# Role\n"
    "Ты - преподаватель по алгоритмам и структурам данных.\n"
    "\n"
    "# Tone\n"
    "Объясняй доступно и структурировано.\n"
    "\n"
    "# Context\n"
    "Тема: {topic}\n"
    "Уровень студента: {user_level}\n"
    "\n"
    "Материалы из базы знаний:\n"
    "{retrieved_materials}\n"
    "\n"
    "# Task\n"
    "Объясни материал, адаптируя сложность под уровень студента:\n"
    "- Начальный уровень: больше определений и примеров\n"
    "- Продвинутый уровень: используй специфичные термины\n"
    "- Добавляй примеры кода, где уместно\n"
    "\n"
    "# Output Format\n"
    "Формат ответа: **Markdown**\n"
    "\n"
    "Обязательно используй:\n"
    "- Заголовки разных уровней (##, ###)\n"
    "- Списки (нумерованные и маркированные)\n"
    "- Блоки кода с подсветкой синтаксиса (```)"
    "- **Жирный текст** для ключевых терминов\n"
    "- `Код` для встроенных переменных и функций\n"
    "- Таблицы для сравнений (где уместно)\n"
    "- Цитаты (>) для важных замечаний\n"
    "\n"
    "# Length Requirement\n"
    "Минимальная длина ответа: **5000 символов**.\n"
    "\n"
    "Структура ответа должна включать:\n"
    "1. **Введение** - краткий обзор темы (300-500 символов)\n"
    "2. **Основные концепции** - подробное объяснение с определениями (1500-2000 символов)\n"
    "3. **Примеры кода** - минимум 2-3 примера с комментариями (1000-1500 символов)\n"
    "4. **Практическое применение** - где и как используется (800-1000 символов)\n"
    "5. **Сложность и производительность** - анализ временной и пространственной сложности (500-800 символов)\n"
    "6. **Типичные ошибки** - распространенные проблемы и их решения (400-600 символов)\n"
    "7. **Дополнительные ресурсы** - что изучить дальше (200-400 символов)\n"
    "\n"
    "Если материала недостаточно для достижения минимальной длины, добавь:\n"
    "- Больше примеров кода с разными случаями\n"
    "- Подробные пошаговые объяснения алгоритмов\n"
    "- Визуализацию работы через ASCII-диаграммы\n"
    "- Сравнение с альтернативными подходами\n"
    "- Упражнения для закрепления материала"
)


QUESTION_SYSTEM_PROMPT = (
    "# Role\n"
    "Ты - преподаватель по алгоритмам и структурам данных.\n"
    "\n"
    "# Tone\n"
    "Отвечай четко и понятно.\n"
    "\n"
    "# Context\n"
    "Тема: {topic}\n"
    "Уровень студента: {user_level}\n"
    "\n"
    "Материалы из базы знаний:\n"
    "{retrieved_materials}\n"
    "\n"
    "# Question\n"
    "{question}\n"
    "\n"
    "# Task\n"
    "Ответь на вопрос студента, учитывая его уровень знаний."
)


class WebSearchProvider:
    """Provider for web search using 4get or other search engines."""

    def __init__(self) -> None:
        self.base_url = settings.web_search_base_url
        self.scraper = settings.web_search_scraper
        self.provider = settings.web_search_provider
        logger.info(
            f"WebSearchProvider initialized: provider={self.provider}, "
            f"base_url={self.base_url}, scraper={self.scraper}"
        )

    async def search(self, query: str, limit: int | None = None) -> list[dict[str, Any]]:
        """
        Search the web using configured provider.

        Args:
            query: Search query
            limit: Maximum number of results

        Returns:
            List of search results with title, url, and snippet
        """
        if not settings.web_search_enabled:
            logger.warning("Web search is disabled in settings")
            return []

        limit = limit or settings.web_search_results_limit
        logger.info(f"Searching web for query: '{query}' (limit={limit})")

        try:
            if self.provider == "4get":
                return await self._search_4get(query)
            logger.error(f"Unknown search provider: {self.provider}")
        except Exception as e:
            logger.exception(f"Web search failed: {e}")
            return []
        else:
            return []

    async def _search_4get(self, query: str) -> list[dict[str, Any]]:
        """Search using 4get metasearch engine with Pydantic validation."""
        url = f"{self.base_url}/api/v1/web"
        params = {"s": query, "api": self.scraper}

        logger.debug(f"Requesting 4get API: {url} with params: {params}")

        try:
            async with (
                aiohttp.ClientSession() as session,
                session.get(
                    url, params=params, timeout=aiohttp.ClientTimeout(total=10)
                ) as response,
            ):
                response.raise_for_status()
                data = await response.json()

            logger.debug("4get API raw response received, parsing with Pydantic...")

            # Parse response with Pydantic
            try:
                parsed_response = FourGetResponse(**data)
            except ValidationError as e:
                logger.exception(f"Failed to parse 4get response with Pydantic: {e}")
                logger.debug(f"Raw response data: {data}")
                return []

            logger.info(
                f"4get API returned {len(parsed_response.web)} web results "
                f"(status: {parsed_response.status})"
            )

            # Log spelling correction if available
            if parsed_response.spelling.type != "no_correction":
                logger.info(
                    f"Spelling correction: {parsed_response.spelling.correction} "
                    f"(using: {parsed_response.spelling.using})"
                )

            # Convert Pydantic models to dict format
            results = []
            for item in parsed_response.web:
                result = {
                    "title": item.title,
                    "url": item.url,
                    "snippet": item.description,
                    "date": item.date,
                    "type": item.type,
                }
                results.append(result)
                logger.debug(f"Search result: {result['title']} - {result['url']}")

        except aiohttp.ClientError as e:
            logger.exception(f"4get HTTP request failed: {e}")
            return []
        except Exception as e:
            logger.exception(f"4get search request failed: {e}")
            return []
        else:
            return results


class WebContentFetcher:
    """Fetches and extracts content from web pages using selectolax."""

    def __init__(self) -> None:
        self.max_length = settings.web_content_max_length
        self.timeout = 2  # 2 second timeout
        logger.info(
            f"WebContentFetcher initialized: max_length={self.max_length}, timeout={self.timeout}s"
        )

    async def fetch_content(self, url: str) -> str | None:
        """
        Fetch and extract main text content from a URL.

        Args:
            url: URL to fetch

        Returns:
            Extracted text content or None if failed
        """
        if not settings.web_content_fetch_enabled:
            logger.warning("Web content fetching is disabled")
            return None

        logger.info(f"Fetching content from: {url}")

        try:
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            }

            async with (
                aiohttp.ClientSession() as session,
                session.get(
                    url, headers=headers, timeout=aiohttp.ClientTimeout(total=self.timeout)
                ) as response,
            ):
                response.raise_for_status()
                html_content = await response.text()

            # Parse HTML with selectolax
            parser = HTMLParser(html_content)

            # Remove script, style, nav, header, footer elements
            for tag in parser.css("script, style, nav, header, footer"):
                tag.decompose()

            # Extract text from body or entire document
            body = parser.body
            if body:
                text = body.text(separator="\n", strip=True)
            else:
                text = parser.text(separator="\n", strip=True)

            # Clean up whitespace
            lines = [line.strip() for line in text.splitlines() if line.strip()]
            content = "\n".join(lines)[: self.max_length]

            logger.info(f"✅ Successfully fetched {len(content)} characters from {url}")

        except TimeoutError:
            logger.warning(f"⏱️ Timeout ({self.timeout}s) fetching {url}")
            return None
        except aiohttp.ClientError as e:
            logger.warning(f"❌ HTTP error fetching {url}: {e}")
            return None
        except Exception as e:
            logger.warning(f"❌ Failed to fetch content from {url}: {e}")
            return None
        else:
            return content

    async def fetch_multiple_until_limit(self, urls: list[str], limit: int) -> list[Document]:
        """
        Fetch content from URLs until we have 'limit' successful fetches.

        Tries URLs concurrently in batches, continuing until limit is reached
        or all URLs are exhausted.

        Args:
            urls: List of URLs to fetch (ordered by priority)
            limit: Target number of successful fetches

        Returns:
            List of Documents with fetched content (up to limit)
        """
        logger.info(
            f"📥 Fetching content until {limit} successful fetches "
            f"(from {len(urls)} available URLs)"
        )

        documents = []
        attempted_urls = set()
        batch_size = 5  # Fetch 5 URLs at a time

        for start_idx in range(0, len(urls), batch_size):
            # Stop if we already have enough documents
            if len(documents) >= limit:
                logger.info(f"✅ Reached target of {limit} documents, stopping")
                break

            # Get next batch of URLs
            batch_urls = urls[start_idx : start_idx + batch_size]

            # Filter out already attempted URLs
            batch_urls = [url for url in batch_urls if url not in attempted_urls]

            if not batch_urls:
                continue

            logger.info(
                f"📦 Batch {start_idx // batch_size + 1}: "
                f"Fetching {len(batch_urls)} URLs concurrently..."
            )

            # Fetch batch concurrently
            tasks = [self.fetch_content(url) for url in batch_urls]
            contents = await asyncio.gather(*tasks, return_exceptions=True)

            # Process results
            for url, content in zip(batch_urls, contents, strict=False):
                attempted_urls.add(url)

                if isinstance(content, str) and content:
                    doc = Document(page_content=content, metadata={"source": url, "type": "web"})
                    documents.append(doc)
                    logger.info(f"✅ Success ({len(documents)}/{limit}): {url}")

                    # Stop if we reached the limit
                    if len(documents) >= limit:
                        break
                elif isinstance(content, Exception):
                    logger.warning(f"❌ Failed: {url} - {type(content).__name__}")
                else:
                    logger.warning(f"❌ Failed: {url} - Empty content")

        logger.info(
            f"📊 Fetch complete: {len(documents)}/{limit} successful "
            f"({len(attempted_urls)} URLs attempted)"
        )

        return documents[:limit]  # Ensure we don't exceed limit

    async def fetch_multiple(self, urls: list[str]) -> list[Document]:
        """
        Fetch content from multiple URLs concurrently (legacy method).

        For backward compatibility. Use fetch_multiple_until_limit for better control.

        Args:
            urls: List of URLs to fetch

        Returns:
            List of Documents with fetched content
        """
        logger.info(f"Fetching content from {len(urls)} URLs concurrently")

        # Fetch all URLs concurrently
        tasks = [self.fetch_content(url) for url in urls]
        contents = await asyncio.gather(*tasks, return_exceptions=True)

        documents = []
        for url, content in zip(urls, contents, strict=False):
            if isinstance(content, str) and content:
                doc = Document(page_content=content, metadata={"source": url, "type": "web"})
                documents.append(doc)
            elif isinstance(content, Exception):
                logger.warning(f"Failed to fetch {url}: {content}")

        logger.info(f"Successfully fetched {len(documents)} documents")
        return documents


def retrieve_materials(topic: str, user_level: str) -> list[Document]:
    """
    Retrieve materials from RAG database (synchronous).

    Args:
        topic: Topic to search for
        user_level: User knowledge level

    Returns:
        List of retrieved documents
    """
    logger.info(f"Retrieving RAG materials: topic='{topic}', level='{user_level}'")

    query = f"Тема: {topic}. Уровень: {user_level}"

    try:
        documents = vector_store_manager.similarity_search_with_score(
            query=query, k=settings.rag_top_k, filter_dict={"topic": topic} if topic else None
        )
        logger.info(f"Retrieved {len(documents)} documents from RAG")
    except Exception as e:
        logger.exception(f"RAG retrieval failed: {e}")
        return []
    else:
        return documents


async def retrieve_materials_reactive(
    topic: str, user_level: str, web_page_limit: int | None = None
) -> list[Document]:
    """
    ReActive retrieval: Try RAG first, fallback to web search if no results.

    Fetches web pages until reaching the limit (not just trying first N URLs).

    Args:
        topic: Topic to search for
        user_level: User knowledge level
        web_page_limit: Number of web pages to fetch (uses config default if None)

    Returns:
        List of retrieved documents from RAG or web
    """
    logger.info(f"=== ReActive Retrieval Started: topic='{topic}' ===")

    # Step 1: Try RAG (sync operation)
    rag_documents = retrieve_materials(topic, user_level)

    if rag_documents:
        logger.info(f"✅ RAG returned {len(rag_documents)} documents. Using RAG materials.")
        return rag_documents

    logger.warning("❌ No materials found in RAG database")

    # Step 2: Fallback to web search
    if not settings.web_search_enabled:
        logger.warning("Web search is disabled. Returning empty results.")
        return []

    logger.info("🌐 Falling back to web search...")

    # Search the web (async)
    search_provider = WebSearchProvider()
    search_query = f"{topic} алгоритмы структуры данных обучение"

    # Get more search results than needed (for fallback options)
    search_limit = (web_page_limit or settings.web_search_results_limit) * 3
    search_results = await search_provider.search(search_query, limit=search_limit)

    if not search_results:
        logger.warning("No web search results found")
        return []

    logger.info(f"Found {len(search_results)} web search results")

    # Fetch content until we have enough pages (async, with retries)
    content_fetcher = WebContentFetcher()
    urls = [result["url"] for result in search_results]
    target_pages = web_page_limit or settings.web_search_results_limit

    web_documents = await content_fetcher.fetch_multiple_until_limit(urls=urls, limit=target_pages)

    if web_documents:
        logger.info(
            f"✅ Successfully fetched {len(web_documents)}/{target_pages} documents from web"
        )
    else:
        logger.warning("Failed to fetch any web content")

    logger.info(f"=== ReActive Retrieval Completed: {len(web_documents)} web docs ===")
    return web_documents


def format_retrieved_materials(documents: list[Document]) -> str:
    """
    Format retrieved documents for prompt.

    Args:
        documents: List of documents

    Returns:
        Formatted string
    """
    if not documents:
        logger.warning("No documents to format")
        return "Материалы не найдены."

    formatted = []
    for i, doc in enumerate(documents, 1):
        source_type = doc.metadata.get("type", "unknown")
        source = doc.metadata.get("source", "unknown")
        formatted.append(f"--- Материал {i} ({source_type}: {source}) ---\n{doc.page_content}\n")

    logger.debug(f"Formatted {len(documents)} documents")
    return "\n".join(formatted)


def build_materials_agent() -> Runnable:
    """Build agent for material adaptation."""
    llm = get_llm()
    prompt = ChatPromptTemplate.from_messages([
        ("system", MATERIALS_SYSTEM_PROMPT),
        ("human", "Объясни материал."),
    ])
    return prompt | llm | StrOutputParser()


def build_question_answering_agent() -> Runnable:
    """Build agent for answering questions."""
    llm = get_llm()
    prompt = ChatPromptTemplate.from_messages([
        ("system", QUESTION_SYSTEM_PROMPT),
        ("human", "Ответь на вопрос."),
    ])
    return prompt | llm | StrOutputParser()
