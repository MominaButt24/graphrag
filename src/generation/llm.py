from langchain.chat_models import init_chat_model

from src.config.llm_config import (
    llm_config,
    fallback_llm_config,
)

_llm = None


def _create_model(cfg: dict):
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
    """Return the shared LLM.

    Anthropic Claude Haiku is the primary provider.
    OpenAI-compatible LLM is used as fallback.
    """
    global _llm

    if _llm is None:
        primary_cfg = llm_config()

        try:
            print(
                f"[LLM] Using Anthropic: "
                f"{primary_cfg['model']}"
            )

            _llm = _create_model(primary_cfg)

        except Exception as e:
            print(
                f"[LLM] Anthropic failed: {e}"
            )
            print("[LLM] Falling back to OpenAI-compatible model")

            fallback_cfg = fallback_llm_config()

            _llm = _create_model(fallback_cfg)

            print(
                f"[LLM] Using fallback: "
                f"{fallback_cfg['model']}"
            )

    return _llm


def reset_llm():
    """Helper for tests or process restarts."""
    global _llm
    _llm = None