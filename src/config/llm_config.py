from src.config.settings import settings


def llm_config() -> dict:
    """Return the normalized kwargs needed for the shared LangChain model object.

    Keep the provider/model fields in one place so every module can ask for the
    same model object instead of copying the same init_chat_model code.
    """
    return {
        "model": settings.llm_model,
        "model_provider": settings.llm_provider,
        "api_key": settings.llm_api_key,
        "base_url": settings.llm_base_url,
        "max_tokens": settings.llm_max_tokens,
    }


def llm_provider() -> str:
    return settings.llm_provider
