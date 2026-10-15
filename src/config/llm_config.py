from src.config.settings import settings


def llm_config() -> dict:
    """Return normalized configuration for the primary LLM."""
    return {
            "model": settings.llm_model,
            "model_provider": settings.llm_provider,
            "api_key": settings.llm_api_key,
            "base_url": settings.llm_base_url,
            "max_tokens": settings.llm_max_tokens,
            "timeout": settings.llm_timeout,
            "max_retries": settings.llm_max_retries,
        }


def fallback_llm_config() -> dict:
    """Return configuration for the OpenAI-compatible fallback."""
    return {
            "model": settings.anthropic_model,
            "model_provider": "anthropic",
            "api_key": settings.anthropic_api_key,
            "base_url": None,
            "max_tokens": settings.llm_max_tokens,
            "timeout": settings.llm_timeout,
            "max_retries": settings.llm_max_retries,
        }
    