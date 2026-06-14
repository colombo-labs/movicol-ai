"""Application settings using pydantic-settings."""

from functools import lru_cache

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application configuration loaded from environment variables."""

    # Server
    host: str = "0.0.0.0"
    port: int = 8000
    env: str = "development"

    # Database
    database_url: str = "postgresql://movicol:movicol_dev@localhost:5432/movicol"

    # LLM
    openai_api_key: str = ""
    llm_model: str = "gpt-4o-mini"

    # Model paths
    model_path: str = "models/gat_best.pt"
    graph_path: str = "models/graph_clean.graphml"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache
def get_settings() -> Settings:
    """Cached settings singleton."""
    return Settings()
