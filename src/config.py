from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # GigaChat settings (единый провайдер)
    gigachat_api_key: str = Field(alias="GIGACHAT_API_KEY", default="")
    gigachat_model: str = Field(alias="GIGACHAT_MODEL", default="GigaChat")
    gigachat3_model: str = Field(alias="GIGACHAT3_MODEL", default="GigaChat3-10B-A1.8B")
    gigachat_base_url: str = Field(
        alias="GIGACHAT_BASE_URL", default="https://gigachat.devices.sberbank.ru/api/v1"
    )

    # LLM common settings
    llm_temperature: float = Field(alias="LLM_TEMPERATURE", default=0.2)
    timeout_s: int = Field(alias="TIMEOUT_S", default=60)

    # ChromaDB settings
    chroma_persist_directory: str = Field(alias="CHROMA_PERSIST_DIRECTORY", default="./chroma_db")
    chroma_collection_name: str = Field(alias="CHROMA_COLLECTION_NAME", default="aisd_materials")

    # RAG settings
    rag_top_k: int = Field(alias="RAG_TOP_K", default=3)

    # Web Search settings (ReActive Agent)
    web_search_enabled: bool = Field(alias="WEB_SEARCH_ENABLED", default=True)
    web_search_provider: str = Field(alias="WEB_SEARCH_PROVIDER", default="4get")
    web_search_base_url: str = Field(alias="WEB_SEARCH_BASE_URL", default="https://4get.sijh.net")
    web_search_scraper: str = Field(alias="WEB_SEARCH_SCRAPER", default="google")
    web_search_results_limit: int = Field(alias="WEB_SEARCH_RESULTS_LIMIT", default=5)
    web_content_fetch_enabled: bool = Field(alias="WEB_CONTENT_FETCH_ENABLED", default=True)
    web_content_max_length: int = Field(alias="WEB_CONTENT_MAX_LENGTH", default=5000)

    # Database settings
    database_url: str = Field(alias="DATABASE_URL", default="sqlite:///./app_data.db")

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


settings = Settings()
