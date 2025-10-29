from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # GigaChat settings
    gigachat_api_key: str = Field(alias="GIGACHAT_API_KEY", default="")
    gigachat_model: str = Field(alias="GIGACHAT_MODEL", default="GigaChat")
    gigachat_base_url: str = Field(
        alias="GIGACHAT_BASE_URL", default="https://gigachat.devices.sberbank.ru/api/v1"
    )

    # DeepSeek settings
    deepseek_api_key: str = Field(alias="DEEPSEEK_API_KEY", default="")
    deepseek_model: str = Field(alias="DEEPSEEK_MODEL", default="deepseek-chat")
    deepseek_base_url: str = Field(alias="DEEPSEEK_BASE_URL", default="https://api.deepseek.com")

    # LLM common settings
    llm_temperature: float = Field(alias="LLM_TEMPERATURE", default=0.2)
    timeout_s: int = Field(alias="TIMEOUT_S", default=60)

    # ChromaDB settings
    chroma_persist_directory: str = Field(alias="CHROMA_PERSIST_DIRECTORY", default="./chroma_db")
    chroma_collection_name: str = Field(alias="CHROMA_COLLECTION_NAME", default="aisd_materials")

    # RAG settings
    rag_top_k: int = Field(alias="RAG_TOP_K", default=3)

    # Database settings
    database_url: str = Field(alias="DATABASE_URL", default="sqlite:///./app_data.db")

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


settings = Settings()
