import logging
import os
import colorlog

_configured = False


def setup_logging(level: str = None):
    """
    Call this once, at process startup. Safe to call more than once —
    it no-ops after the first call, so importing it in multiple places
    (e.g. both main.py and a script's __main__ block) won't double up
    handlers or duplicate log lines.
    """
    global _configured
    if _configured:
        return

    log_level = (level or os.getenv("LOG_LEVEL", "INFO")).upper()

    handler = colorlog.StreamHandler()
    handler.setFormatter(
        colorlog.ColoredFormatter(
            "%(log_color)s%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%H:%M:%S",
            log_colors={
                "DEBUG": "cyan",
                "INFO": "green",
                "WARNING": "yellow",
                "ERROR": "red",
                "CRITICAL": "red,bg_white",
            },
        )
    )

    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    root_logger.handlers = [handler]  # replace, don't stack, in case something else already added one
    _configured = True


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger for the calling module. Doesn't configure anything —
    assumes setup_logging() has already run at the entry point. If it
    hasn't (e.g. during a quick REPL import), Python's logging module
    falls back to a plain default so nothing crashes, it just won't be
    colorized until setup_logging() actually runs.
    """
    return logging.getLogger(name)