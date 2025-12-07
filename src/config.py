from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env")

    # GigaChat (единый провайдер)
    gigachat_api_key: str = Field(alias="GIGACHAT_API_KEY")
    gigachat_model: str = Field(alias="GIGACHAT_MODEL", default="GigaChat")
    gigachat_base_url: str = Field(alias="GIGACHAT_BASE_URL", default="https://gigachat.devices.sberbank.ru/api/v1")

    # LLM
    llm_temperature: float = Field(alias="LLM_TEMPERATURE", default=0.2)
    timeout_s: int = Field(alias="TIMEOUT_S", default=60)

    # ChromaDB
    chroma_persist_directory: str = Field(alias="CHROMA_PERSIST_DIRECTORY", default=".chromadb")
    chroma_collection_name: str = Field(alias="CHROMA_COLLECTION_NAME", default="aisd_materials")

    # RAG
    rag_top_k: int = Field(alias="RAG_TOP_K", default=3)

    # Database
    database_url: str = Field(alias="DATABASE_URL", default="sqlite+aiosqlite:///./app.db")

settings = Settings()
