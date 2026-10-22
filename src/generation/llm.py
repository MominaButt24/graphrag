
from langchain.chat_models import init_chat_model

from src.config.llm_config import llm_config

_llm = None


def _create_model(cfg: dict):
    if not cfg["api_key"]:
        raise ValueError(
            "LLM_API_KEY is empty. Check your .env and environment."
        )

    return init_chat_model(
        model=cfg["model"],
        model_provider=cfg["model_provider"],
        api_key=cfg["api_key"],
        base_url=cfg["base_url"],
        max_tokens=cfg["max_tokens"],
        timeout=cfg["timeout"],
        max_retries=cfg["max_retries"],
    )


def get_llm():
    """Return the configured OpenAI-compatible LLM."""
    global _llm

    if _llm is None:
        cfg = llm_config()

        print(
            f"[LLM] Provider: {cfg['model_provider']}, "
            f"Model: {cfg['model']}, "
            f"Base URL: {cfg['base_url']}"
        )

        _llm = _create_model(cfg)

    return _llm


def reset_llm():
    """Reset the shared LLM instance."""
    global _llm
    _llm = None
