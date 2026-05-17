from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application configuration loaded from environment variables and .env."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    app_name: str = Field(
        default="Chat with Marine Biology Docs",
        validation_alias=AliasChoices("MARINE_RAG_APP_NAME", "APP_NAME"),
    )
    environment: str = Field(
        default="development",
        validation_alias=AliasChoices("MARINE_RAG_ENVIRONMENT", "ENVIRONMENT"),
    )
    log_level: str = Field(
        default="INFO",
        validation_alias=AliasChoices("MARINE_RAG_LOG_LEVEL", "LOG_LEVEL"),
    )

    raw_docs_dir: Path = Field(
        default=Path("data/raw_docs"),
        validation_alias="MARINE_RAG_RAW_DOCS_DIR",
    )
    storage_dir: Path = Field(
        default=Path("data/storage"),
        validation_alias="MARINE_RAG_STORAGE_DIR",
    )
    chroma_collection: str = Field(
        default="marine_biology_docs",
        validation_alias="MARINE_RAG_CHROMA_COLLECTION",
    )
    chroma_host: str | None = Field(default=None, validation_alias="MARINE_RAG_CHROMA_HOST")
    chroma_port: int = Field(default=8000, validation_alias="MARINE_RAG_CHROMA_PORT")
    chroma_ssl: bool = Field(default=False, validation_alias="MARINE_RAG_CHROMA_SSL")

    redis_url: str | None = Field(default=None, validation_alias="MARINE_RAG_REDIS_URL")
    cache_ttl_seconds: int = Field(default=900, validation_alias="MARINE_RAG_CACHE_TTL_SECONDS")

    chunk_size: int = Field(default=900, validation_alias="MARINE_RAG_CHUNK_SIZE")
    chunk_overlap: int = Field(default=150, validation_alias="MARINE_RAG_CHUNK_OVERLAP")
    similarity_top_k: int = Field(default=20, validation_alias="MARINE_RAG_SIMILARITY_TOP_K")
    final_top_k: int = Field(default=5, validation_alias="MARINE_RAG_FINAL_TOP_K")

    llm_provider: Literal["openai", "anthropic"] = Field(
        default="openai",
        validation_alias="MARINE_RAG_LLM_PROVIDER",
    )
    openai_api_key: str | None = Field(
        default=None,
        validation_alias=AliasChoices("MARINE_RAG_OPENAI_API_KEY", "OPENAI_API_KEY"),
    )
    anthropic_api_key: str | None = Field(
        default=None,
        validation_alias=AliasChoices("MARINE_RAG_ANTHROPIC_API_KEY", "ANTHROPIC_API_KEY"),
    )
    openai_model: str = Field(
        default="gpt-4o-mini",
        validation_alias=AliasChoices("MARINE_RAG_OPENAI_MODEL", "OPENAI_MODEL"),
    )
    anthropic_model: str = Field(
        default="claude-3-5-sonnet-latest",
        validation_alias="MARINE_RAG_ANTHROPIC_MODEL",
    )
    llm_temperature: float = Field(default=0.1, validation_alias="MARINE_RAG_LLM_TEMPERATURE")

    embedding_provider: Literal["openai", "local"] = Field(
        default="openai",
        validation_alias="MARINE_RAG_EMBEDDING_PROVIDER",
    )
    openai_embedding_model: str = Field(
        default="text-embedding-3-small",
        validation_alias="MARINE_RAG_OPENAI_EMBEDDING_MODEL",
    )
    local_embedding_model: str = Field(
        default="BAAI/bge-small-en-v1.5",
        validation_alias="MARINE_RAG_LOCAL_EMBEDDING_MODEL",
    )

    enable_reranking: bool = Field(default=True, validation_alias="MARINE_RAG_ENABLE_RERANKING")
    reranker_model: str = Field(
        default="cross-encoder/ms-marco-MiniLM-L-6-v2",
        validation_alias="MARINE_RAG_RERANKER_MODEL",
    )

    @property
    def chroma_path(self) -> Path:
        return self.storage_dir / "chroma"


@lru_cache
def get_settings() -> Settings:
    return Settings()
