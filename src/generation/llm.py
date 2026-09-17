from langchain.chat_models import init_chat_model

from src.config.llm_config import llm_config

_llm = None


def get_llm():
    """Return the shared LangChain chat model instance.

    This is the one source of truth for LLM creation in the repo.
    All other files should ask for it here instead of constructing a
    brand-new model every time they need a reply.
    """
    global _llm
    if _llm is None:
        cfg = llm_config()
        _llm = init_chat_model(
            model=cfg["model"],
            model_provider=cfg["model_provider"],
            api_key=cfg["api_key"],
            base_url=cfg["base_url"],
            max_tokens=cfg["max_tokens"],
            timeout=cfg["timeout"],
            max_retries=cfg["max_retries"],
        )
    return _llm


def reset_llm():
    """Optional helper for tests or process restarts."""
    global _llm
    _llm = None
