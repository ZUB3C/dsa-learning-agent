"""
Pydantic schemas for tool parameters and results.
"""

from pydantic import BaseModel, Field


class AdaptiveRAGParams(BaseModel):
    """Parameters for Adaptive RAG tool."""

    query: str = Field(..., description="Search query")
    strategy: str = Field(
        "auto", description="Retrieval strategy: auto | tfidf | semantic | hybrid"
    )
    k: int = Field(5, ge=1, le=20, description="Number of documents to retrieve")
    user_level: str | None = Field(None, description="User level for filtering")


class CorrectiveRAGParams(BaseModel):
    """Parameters for Corrective RAG tool."""

    query: str = Field(..., description="Search query")
    documents: list[str] = Field(..., description="Documents to evaluate")
    min_relevance: float = Field(0.6, ge=0.0, le=1.0, description="Min relevance threshold")
    evaluate_coverage: bool = Field(True, description="Evaluate concept coverage")


class WebSearchParams(BaseModel):
    """Parameters for Web Search tool."""

    query: str = Field(..., description="Search query")
    num_results: int = Field(5, ge=1, le=20, description="Number of results")
    domain_filter: str | None = Field(None, description="Filter by domain")
    exclude_domains: list[str] = Field(default_factory=list, description="Exclude domains")
    scrape_content: bool = Field(True, description="Scrape full content from results")


class WebScraperParams(BaseModel):
    """Parameters for Web Scraper tool."""

    urls: list[str] = Field(..., description="URLs to scrape")
    extract_text: bool = Field(True, description="Extract text content")
    extract_metadata: bool = Field(True, description="Extract metadata")
    timeout_s: float = Field(5.0, description="Fetch timeout")


class ConceptExtractorParams(BaseModel):
    """Parameters for Concept Extractor tool."""

    text: str = Field(..., description="Text to extract concepts from")
    method: str = Field("auto", description="auto | keybert | spacy | hybrid")
    top_n: int = Field(10, ge=1, le=50, description="Number of concepts")
    language: str = Field("ru", description="Language code")


class MemoryRetrievalParams(BaseModel):
    """Parameters for Memory Retrieval tool."""

    query: str = Field(..., description="Query for memory search")
    memory_type: str = Field("procedural", description="working | procedural | all")
    limit: int = Field(3, ge=1, le=10, description="Number of patterns to retrieve")
    min_success_score: float = Field(0.8, ge=0.0, le=1.0, description="Min success threshold")


class ToolParamsUnion(BaseModel):
    """Union of all tool parameters."""

    tool_name: str = Field(..., description="Name of the tool")
    adaptive_rag: AdaptiveRAGParams | None = None
    corrective_rag: CorrectiveRAGParams | None = None
    web_search: WebSearchParams | None = None
    web_scraper: WebScraperParams | None = None
    concept_extractor: ConceptExtractorParams | None = None
    memory_retrieval: MemoryRetrievalParams | None = None
