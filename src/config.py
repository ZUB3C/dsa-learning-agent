"""
Configuration management for Materials Agent v2.
Loads from .env file with defaults.
"""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class BaseNestedSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", case_sensitive=False, extra="ignore"
    )


class LLMSettings(BaseNestedSettings):
    """LLM provider configuration."""

    # GigaChat-2-Max (Expensive, High Quality)
    gigachat_api_key: str = Field(description="GigaChat API Key")
    gigachat_model: str = Field("GigaChat/GigaChat-2-Max", description="GigaChat model name")
    gigachat_base_url: str = Field(
        "https://foundation-models.api.cloud.ru/v1", description="GigaChat API base URL"
    )
    gigachat_temperature: float = Field(0.2, description="Temperature for GigaChat")
    gigachat_max_tokens: int = Field(4096, description="Max tokens for GigaChat")
    gigachat_timeout_s: float = Field(60.0, description="GigaChat timeout in seconds")

    # GigaChat3 (Free, Fast)
    gigachat3_model: str = Field("ai-sage/GigaChat3-10B-A1.8B", description="GigaChat3 model")
    gigachat3_temperature: float = Field(0.1, description="Temperature for GigaChat3")
    gigachat3_max_tokens: int = Field(2048, description="Max tokens for GigaChat3")
    gigachat3_timeout_s: float = Field(10.0, description="GigaChat3 timeout in seconds")


class ToTSettings(BaseNestedSettings):
    """Tree-of-Thoughts configuration."""

    tot_max_depth: int = Field(5, description="Max ToT tree depth")
    tot_branching_factor: int = Field(3, description="Number of branches per node")
    tot_completeness_threshold: float = Field(0.85, description="Goal completeness threshold")
    tot_min_iterations: int = Field(2, description="Minimum iterations before termination")
    tot_promise_threshold: float = Field(0.3, description="Min promise score to continue")
    tot_dead_end_relevance: float = Field(0.4, description="Min relevance for dead-end detection")
    tot_dead_end_quality: float = Field(0.3, description="Min quality for dead-end detection")


class ContentGuardSettings(BaseNestedSettings):
    """Content Guard Layer configuration."""

    content_guard_enabled: bool = Field(True, description="Enable Content Guard")

    # Toxicity Check
    toxicity_threshold: float = Field(0.3, description="Toxicity score threshold")
    toxicity_batch_size: int = Field(5, description="Batch size for toxicity checks")
    toxicity_timeout_s: float = Field(5.0, description="Toxicity check timeout")

    # Policy Check
    policy_check_enabled: bool = Field(True, description="Enable policy compliance check")
    policy_check_timeout_s: float = Field(5.0, description="Policy check timeout")

    # Content Sanitization
    sanitize_remove_urls: bool = Field(True, description="Remove URLs from content")
    sanitize_remove_emails: bool = Field(True, description="Remove emails from content")
    sanitize_max_length_per_doc: int = Field(3000, description="Max chars per document")

    # Quality Gate
    min_content_length: int = Field(100, description="Minimum content length")
    max_content_length: int = Field(8000, description="Maximum content length")
    min_sentence_count: int = Field(3, description="Minimum sentence count")


class ChromaDBSettings:
    """ChromaDB vector store configuration."""

    persist_directory: Path = Field(
        default=Path("./data/chroma"), description="ChromaDB persistence directory"
    )
    collection_name: str = Field(default="aisd_materials", description="Main collection name")
    working_memory_collection: str = Field(
        default="agent_working_memory", description="Working memory collection name"
    )
    procedural_memory_collection: str = Field(
        default="agent_procedural_memory", description="Procedural memory collection name"
    )


class ChunkingSettings:
    """Document chunking configuration for text splitting and embedding."""

    # Default text chunking
    chunk_size: int = Field(
        default=1000,
        description="Default chunk size in characters for text splitting",
        ge=100,
        le=10000,
    )
    chunk_overlap: int = Field(
        default=200,
        description="Overlap between consecutive chunks to preserve context",
        ge=0,
        le=1000,
    )

    # PDF-specific chunking
    pdf_chunk_size: int = Field(
        default=1000, description="Chunk size for PDF documents", ge=100, le=10000
    )
    pdf_chunk_overlap: int = Field(
        default=200, description="Overlap for PDF document chunks", ge=0, le=1000
    )

    # Advanced chunking options
    chunking_strategy: Literal["fixed", "recursive", "semantic"] = Field(
        default="recursive",
        description="Chunking strategy: fixed (character-based), recursive (respects structure), semantic (meaning-based)",
    )

    separators: list[str] = Field(
        default=["\n\n", "\n", ". ", " ", ""],
        description="Separators for recursive chunking, in order of preference",
    )

    min_chunk_size: int = Field(
        default=100, description="Minimum chunk size to avoid too-small chunks", ge=50
    )

    @field_validator("chunk_overlap", "pdf_chunk_overlap")
    @classmethod
    def validate_overlap(cls, v: int, info) -> int:
        """Ensure overlap is smaller than chunk size."""
        info.field_name.replace("_overlap", "_size")
        # This will be validated after all fields are set
        return v

    def model_post_init(self, __context) -> None:
        """Validate overlap relative to chunk size after initialization."""
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(
                f"chunk_overlap ({self.chunk_overlap}) must be less than chunk_size ({self.chunk_size})"
            )
        if self.pdf_chunk_overlap >= self.pdf_chunk_size:
            raise ValueError(
                f"pdf_chunk_overlap ({self.pdf_chunk_overlap}) must be less than pdf_chunk_size ({self.pdf_chunk_size})"
            )


class VectorStoreSettings(BaseNestedSettings, ChromaDBSettings, ChunkingSettings):
    """Vector store and embedding configuration."""

    semantic_embedding_model: str = Field(
        default="sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        description="Embedding model for semantic search",
    )
    # chroma: ChromaDBSettings = Field(default_factory=ChromaDBSettings)
    # chunking: ChunkingSettings = Field(default_factory=ChunkingSettings)


class AdaptiveRAGSettings(BaseNestedSettings):
    """Adaptive RAG configuration."""

    adaptive_simple_threshold: int = Field(20, description="Simple query char threshold")
    adaptive_complex_threshold: int = Field(50, description="Complex query char threshold")

    rag_top_k: int = Field(5, description="Top K documents to retrieve")
    rag_timeout_s: float = Field(5.0, description="RAG retrieval timeout")

    # TF-IDF
    tfidf_model_path: str = Field("./models/tfidf_model.pkl", description="TF-IDF model path")
    tfidf_update_interval_s: int = Field(86400, description="TF-IDF update interval")
    tfidf_rebuild_on_missing: bool = Field(True, description="Rebuild TF-IDF if missing")

    # Semantic Search
    semantic_embedding_model: str = Field(
        "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        description="Embedding model for semantic search",
    )
    semantic_extended_timeout_s: float = Field(10.0, description="Extended semantic timeout")

    # Hybrid Search (RRF)
    rrf_k_constant: int = Field(60, description="RRF constant")


class CorrectiveRAGSettings(BaseNestedSettings):
    """Corrective RAG configuration."""

    corrective_enabled: bool = Field(True, description="Enable Corrective RAG")
    corrective_min_relevance: float = Field(0.6, description="Min relevance score")
    corrective_batch_size: int = Field(5, description="Batch size for relevance checks")
    corrective_timeout_s: float = Field(5.0, description="Corrective RAG timeout")
    corrective_min_coverage: float = Field(0.6, description="Min concept coverage")
    corrective_min_docs_after_filter: int = Field(3, description="Min docs after filtering")


class WebSearchSettings(BaseNestedSettings):
    """Web search configuration."""

    web_search_enabled: bool = Field(True, description="Enable web search")

    # Primary Instance
    web_search_base_url: str = Field("https://4get.bloat.cat", description="4get base URL")
    web_search_scraper: str = Field("google", description="Primary scraper")

    # Fallback Instances
    web_search_fallback_urls: list[str] = Field(
        ["https://4get.bloat.cat", "https://4get.lunar.icu"],
        description="Fallback 4get URLs",
    )
    web_search_fallback_scrapers: list[str] = Field(
        ["bing", "duckduckgo", "google"], description="Fallback scrapers"
    )

    # Search Parameters
    web_search_results_limit: int = Field(5, description="Max search results")
    web_search_timeout_s: float = Field(10.0, description="Web search timeout")
    web_search_retry_count: int = Field(2, description="Retry count")

    # Domain Priorities
    web_search_priority_edu: float = Field(2.0, description=".edu priority")
    web_search_priority_org: float = Field(1.5, description=".org priority")
    web_search_priority_gov: float = Field(1.5, description=".gov priority")
    web_search_priority_wiki: float = Field(1.2, description="Wikipedia priority")
    web_search_priority_habr: float = Field(1.0, description="Habr priority")
    web_search_priority_vc: float = Field(1.0, description="VC priority")
    web_search_priority_stackoverflow: float = Field(0.8, description="Stack Overflow priority")
    web_search_priority_com: float = Field(0.5, description=".com priority")

    # Blacklist
    web_search_blacklist: list[str] = Field(
        ["pinterest.com", "quora.com", "forbes.com", "medium.com"],
        description="Blacklisted domains",
    )

    # Query Optimization
    web_search_add_context: bool = Field(True, description="Add context to queries")
    web_search_context_suffix: str = Field(
        "алгоритмы обучение программирование", description="Query context suffix"
    )
    web_search_deduplicate_threshold: float = Field(0.85, description="Dedup threshold")


class WebScraperSettings(BaseNestedSettings):
    """Web scraping configuration."""

    web_content_fetch_enabled: bool = Field(True, description="Enable web content fetching")
    web_content_max_length: int = Field(8000, description="Max content length")
    web_content_min_length: int = Field(300, description="Min content length")
    web_content_timeout_s: float = Field(5.0, description="Fetch timeout")
    web_content_extended_timeout_s: float = Field(10.0, description="Extended timeout")
    web_content_batch_size: int = Field(5, description="Batch size for fetching")
    web_content_retry_count: int = Field(2, description="Retry count")

    web_content_user_agents: list[str] = Field(
        [
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36",
        ],
        description="User agents for rotation",
    )

    web_content_selectors: list[str] = Field(
        ["article", "main", ".content", ".post-content", "body"],
        description="CSS selectors for content",
    )
    web_content_remove_tags: list[str] = Field(
        ["script", "style", "nav", "header", "footer", "aside", "iframe"],
        description="Tags to remove",
    )


class MemorySettings(BaseNestedSettings):
    """Memory configuration."""

    chroma_host: str = Field("localhost", description="ChromaDB host")
    chroma_port: int = Field(8000, description="ChromaDB port")
    chroma_persist_directory: str = Field(".chromadb", description="ChromaDB persist dir")

    # Collections
    chroma_rag_collection: str = Field("aisd_materials", description="RAG collection name")
    chroma_working_memory_collection: str = Field(
        "agent_working_memory", description="Working memory collection"
    )
    chroma_procedural_memory_collection: str = Field(
        "agent_procedural_memory", description="Procedural memory collection"
    )

    # Working Memory
    memory_working_ttl_hours: int = Field(1, description="Working memory TTL")
    memory_working_cleanup_interval_s: int = Field(3600, description="Cleanup interval")

    # Procedural Memory
    memory_procedural_max_patterns: int = Field(1000, description="Max patterns")
    memory_procedural_min_success_score: float = Field(0.8, description="Min success score")
    memory_procedural_min_usage_for_save: int = Field(1, description="Min usage to save")

    # Fallback
    memory_use_sqlite_backup: bool = Field(True, description="Use SQLite backup")
    memory_use_in_memory_fallback: bool = Field(True, description="Use in-memory fallback")


class CacheSettings(BaseNestedSettings):
    """Cache configuration."""

    redis_enabled: bool = Field(False, description="Enable Redis cache")
    redis_host: str = Field("localhost", description="Redis host")
    redis_port: int = Field(6379, description="Redis port")
    redis_db: int = Field(0, description="Redis DB")
    redis_password: str | None = Field(None, description="Redis password")
    redis_timeout_s: float = Field(2.0, description="Redis timeout")

    # Cache TTLs
    cache_web_search_ttl: int = Field(86400, description="Web search cache TTL")
    cache_web_content_ttl: int = Field(604800, description="Web content cache TTL")
    cache_rag_results_ttl: int = Field(3600, description="RAG results cache TTL")


class DataBaseModel(BaseNestedSettings):
    """Database configuration."""

    database_url: str = Field("sqlite+aiosqlite:///./app.db", description="Database URL")
    database_echo: bool = Field(False, description="Echo SQL queries")

    # Backup
    db_backup_enabled: bool = Field(True, description="Enable DB backup")
    db_backup_path: str = Field("./backups/db", description="Backup path")
    db_backup_interval_s: int = Field(86400, description="Backup interval")

    # JSON Fallback
    db_json_fallback_enabled: bool = Field(True, description="Enable JSON fallback")
    db_json_fallback_path: str = Field("./backups/generations", description="JSON fallback path")

    # Retry
    db_retry_count: int = Field(2, description="Retry count")
    db_retry_delay_ms: int = Field(100, description="Retry delay")


class ValidationSettings(BaseNestedSettings):
    """Input validation configuration."""

    validation_enabled: bool = Field(True, description="Enable validation")
    validation_timeout_s: float = Field(5.0, description="Validation timeout")
    validation_max_input_length: int = Field(200, description="Max input length")
    validation_min_input_length: int = Field(5, description="Min input length")

    # Injection Detection
    validation_injection_patterns: list[str] = Field(
        ["ignore previous", "system prompt", "__import__", "eval(", "exec(", "<script>"],
        description="Injection patterns",
    )

    # SQL Injection Detection
    validation_sql_patterns: list[str] = Field(
        ["DROP TABLE", "DELETE FROM", "INSERT INTO", "UPDATE SET"],
        description="SQL injection patterns",
    )


class ConceptExtractionSettings(BaseNestedSettings):
    """Concept extraction configuration."""

    concept_keybert_enabled: bool = Field(True, description="Enable KeyBERT")
    concept_keybert_model: str = Field(
        "sentence-transformers/paraphrase-multilingual-mpnet-base-v2", description="KeyBERT model"
    )
    concept_keybert_top_n: int = Field(10, description="Top N concepts")

    concept_spacy_enabled: bool = Field(True, description="Enable spaCy")
    concept_spacy_model: str = Field("ru_core_news_lg", description="spaCy model")
    concept_spacy_entity_types: list[str] = Field(
        ["PERSON", "ORG", "GPE", "NORP", "FAC", "LOC"], description="Entity types"
    )

    concept_fuzzy_threshold: float = Field(0.85, description="Fuzzy matching threshold")


class GenerationSettings(BaseNestedSettings):
    """Final generation configuration."""

    generation_min_length: int = Field(5000, description="Min material length")
    generation_timeout_s: float = Field(60.0, description="Generation timeout")
    generation_retry_count: int = Field(2, description="Retry count")

    # Context Reduction
    generation_retry_max_docs: int = Field(5, description="Max docs on first retry")
    generation_retry_2_max_docs: int = Field(3, description="Max docs on second retry")
    generation_retry_2_timeout_s: float = Field(90.0, description="Second retry timeout")

    # Template Fallback
    generation_template_enabled: bool = Field(True, description="Enable template fallback")
    generation_template_path: str = Field(
        "./templates/material_template.md", description="Template path"
    )


class LoggingSettings(BaseNestedSettings):
    """Logging configuration."""

    log_level: str = Field("INFO", description="Log level")
    log_file: str = Field("./logs/materials_agent.log", description="Log file")
    log_max_bytes: int = Field(10485760, description="Max log file size")
    log_backup_count: int = Field(5, description="Backup count")
    log_format: str = Field(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s", description="Log format"
    )

    log_json_enabled: bool = Field(True, description="Enable JSON logging")

    # Metrics
    metrics_enabled: bool = Field(True, description="Enable metrics")
    metrics_track_tot_paths: bool = Field(True, description="Track ToT paths")
    metrics_track_tool_usage: bool = Field(True, description="Track tool usage")
    metrics_track_llm_calls: bool = Field(True, description="Track LLM calls")
    metrics_track_content_guard: bool = Field(True, description="Track content guard")
    metrics_export_interval_s: int = Field(60, description="Export interval")

    # Alerts
    alert_enabled: bool = Field(True, description="Enable alerts")
    alert_on_repeated_failures: bool = Field(True, description="Alert on failures")
    alert_failure_threshold: int = Field(3, description="Failure threshold")
    alert_email: str | None = Field(None, description="Alert email")


class RateLimitSettings(BaseNestedSettings):
    """Rate limiting configuration."""

    rate_limit_gigachat_max_calls_per_minute: int = Field(
        10, description="GigaChat-2-Max calls/min"
    )
    rate_limit_gigachat_max_tokens_per_hour: int = Field(
        100000, description="GigaChat-2-Max tokens/hour"
    )
    rate_limit_gigachat3_max_calls_per_minute: int = Field(30, description="GigaChat3 calls/min")

    # Cost Estimation
    cost_per_1k_tokens_gigachat: float = Field(0.002, description="Cost per 1K tokens")
    cost_alert_threshold_per_request: float = Field(0.10, description="Cost alert per request")
    cost_alert_threshold_per_hour: float = Field(5.0, description="Cost alert per hour")


class APISettings(BaseNestedSettings):
    """API configuration."""

    api_host: str = Field("0.0.0.0", description="API host")
    api_port: int = Field(8000, description="API port")
    api_workers: int = Field(4, description="Worker count")

    # CORS
    cors_enabled: bool = Field(True, description="Enable CORS")
    cors_origins: list[str] = Field(
        ["http://localhost:3000", "https://example.com"], description="CORS origins"
    )
    cors_methods: list[str] = Field(["GET", "POST", "PUT", "DELETE"], description="CORS methods")
    cors_headers: str = Field("*", description="CORS headers")

    # Rate Limiting
    api_rate_limit_per_minute: int = Field(10, description="Requests per minute")
    api_rate_limit_per_hour: int = Field(100, description="Requests per hour")


class SecuritySettings(BaseNestedSettings):
    """Security configuration."""

    secret_key: str = Field(..., description="Secret key for JWT")
    jwt_algorithm: str = Field("HS256", description="JWT algorithm")
    jwt_expiration_minutes: int = Field(60, description="JWT expiration")


class FeatureFlags(BaseNestedSettings):
    """Feature flags."""

    feature_tot_enabled: bool = Field(True, description="Enable ToT")
    feature_adaptive_rag_enabled: bool = Field(True, description="Enable Adaptive RAG")
    feature_corrective_rag_enabled: bool = Field(True, description="Enable Corrective RAG")
    feature_web_search_enabled: bool = Field(True, description="Enable web search")
    feature_content_guard_enabled: bool = Field(True, description="Enable content guard")
    feature_procedural_memory_enabled: bool = Field(True, description="Enable procedural memory")

    # Experimental
    feature_graph_rag_enabled: bool = Field(False, description="Enable Graph RAG")
    feature_multi_agent_enabled: bool = Field(False, description="Enable multi-agent")


class ProjectSettings(BaseNestedSettings):
    """Project metadata."""

    project_name: str = Field("Materials_Agent_v2", description="Project name")
    version: str = Field("2.0.0", description="Version")
    environment: str = Field("production", description="Environment")
    debug: bool = Field(False, description="Debug mode")


class Settings(BaseSettings):
    """
    Combined settings class that aggregates all sub-settings.
    """

    project: ProjectSettings = Field(default_factory=ProjectSettings)  # pyright: ignore[reportArgumentType]
    llm: LLMSettings = Field(default_factory=LLMSettings)  # pyright: ignore[reportArgumentType]
    tot: ToTSettings = Field(default_factory=ToTSettings)  # pyright: ignore[reportArgumentType]
    content_guard: ContentGuardSettings = Field(default_factory=ContentGuardSettings)  # pyright: ignore[reportArgumentType]
    adaptive_rag: AdaptiveRAGSettings = Field(default_factory=AdaptiveRAGSettings)  # pyright: ignore[reportArgumentType]
    corrective_rag: CorrectiveRAGSettings = Field(default_factory=CorrectiveRAGSettings)  # pyright: ignore[reportArgumentType]
    web_search: WebSearchSettings = Field(default_factory=WebSearchSettings)  # pyright: ignore[reportArgumentType]
    web_scraper: WebScraperSettings = Field(default_factory=WebScraperSettings)  # pyright: ignore[reportArgumentType]
    memory: MemorySettings = Field(default_factory=MemorySettings)  # pyright: ignore[reportArgumentType]
    cache: CacheSettings = Field(default_factory=CacheSettings)  # pyright: ignore[reportArgumentType]
    database: DataBaseModel = Field(default_factory=DataBaseModel)  # pyright: ignore[reportArgumentType]
    validation: ValidationSettings = Field(default_factory=ValidationSettings)  # pyright: ignore[reportArgumentType]
    concept_extraction: ConceptExtractionSettings = Field(
        default_factory=ConceptExtractionSettings  # pyright: ignore[reportArgumentType]
    )
    generation: GenerationSettings = Field(default_factory=GenerationSettings)  # pyright: ignore[reportArgumentType]
    logging: LoggingSettings = Field(default_factory=LoggingSettings)  # pyright: ignore[reportArgumentType]
    rate_limit: RateLimitSettings = Field(default_factory=RateLimitSettings)  # pyright: ignore[reportArgumentType]
    api: APISettings = Field(default_factory=APISettings)  # pyright: ignore[reportArgumentType]
    security: SecuritySettings = Field(default_factory=SecuritySettings)  # pyright: ignore[reportArgumentType]
    features: FeatureFlags = Field(default_factory=FeatureFlags)  # pyright: ignore[reportArgumentType]
    vector_store: VectorStoreSettings = Field(default_factory=VectorStoreSettings)  # pyright: ignore[reportArgumentType]

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        env_nested_delimiter="__",
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
