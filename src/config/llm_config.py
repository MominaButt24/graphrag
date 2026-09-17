from src.config.settings import settings


def llm_config() -> dict:
    """Return normalized configuration for the shared LangChain model."""

    return {
        "model": settings.llm_model,
        "model_provider": settings.llm_provider,
        "api_key": settings.llm_api_key,
        "base_url": settings.llm_base_url,
        "max_tokens": settings.llm_max_tokens,
        "timeout": settings.llm_timeout,
        "max_retries": settings.llm_max_retries,
    }


def llm_provider() -> str:
    return settings.llm_provider