import os
from dataclasses import dataclass
from typing import Optional
from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    """Central environment-backed settings object.

    Keep this file as the single location that reads the runtime
    environment so every other module imports configuration from here.
    """

    llm_model: str = os.getenv("LLM_MODEL", "gpt-4o-mini")
    llm_provider: str = os.getenv("LLM_PROVIDER", "openai")
    llm_api_key: str = os.getenv("LLM_API_KEY", "")
    llm_base_url: Optional[str] = os.getenv("LLM_BASE_URL")
    llm_max_tokens: int = int(os.getenv("LLM_MAX_TOKENS", "512"))

    neo4j_uri: Optional[str] = os.getenv("NEO4J_URI")
    neo4j_username: Optional[str] = os.getenv("NEO4J_USERNAME")
    neo4j_password: Optional[str] = os.getenv("NEO4J_PASSWORD")

    milvus_uri: Optional[str] = os.getenv("MILVUS_URI")
    milvus_token: Optional[str] = os.getenv("MILVUS_TOKEN")

    minio_endpoint: Optional[str] = os.getenv("MINIO_ENDPOINT")
    minio_access_key: Optional[str] = os.getenv("MINIO_ACCESS_KEY")
    minio_secret_key: Optional[str] = os.getenv("MINIO_SECRET_KEY")
    minio_bucket: Optional[str] = os.getenv("MINIO_BUCKET")

    log_level: str = os.getenv("LOG_LEVEL", "INFO")


settings = Settings()


def get_env(name: str, default: Optional[str] = None) -> Optional[str]:
    """Read a protected environment variable without repeating dotenv setup."""
    return os.getenv(name, default)
