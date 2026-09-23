from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration loaded from environment variables and `.env`."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_environment: Literal["development", "production"] = "development"
    log_level: str = "INFO"
    port: int = 8000
    allowed_origins: str = ""
    qdrant_mode: Literal["local", "url", "cloud"] = "local"
    qdrant_path: Path = Path("./qdrant_storage")
    qdrant_url: str | None = None
    qdrant_api_key: str | None = None
    qdrant_timeout_seconds: float = 10.0
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_device: str = "cpu"
    embedding_batch_size: int = 16
    llm_model_name: str = "HuggingFaceTB/SmolLM2-360M-Instruct"
    rag_llm_device: str = "cpu"
    rag_min_relevance_score: float = Field(default=0.20, ge=-1.0, le=1.0)
    hf_hub_disable_xet: str = "1"

    @property
    def cors_origins(self) -> list[str]:
        return [origin.strip() for origin in self.allowed_origins.split(",") if origin.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
