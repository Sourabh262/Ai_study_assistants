import logging
from pathlib import Path

from app.config import LOG_DIR


def setup_logging() -> logging.Logger:
    """
    Configure application-wide logging.

    Logs are written to:
    - Console
    - logs/app.log
    """

    LOG_DIR.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("ai_study_assistant")
    logger.setLevel(logging.INFO)

    # Prevent duplicate handlers if setup_logging() is called more than once
    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    # File handler
    file_handler = logging.FileHandler(
        LOG_DIR / "app.log",
        encoding="utf-8",
    )
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger


logger = setup_logging()