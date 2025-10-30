import logging
import sys


def init_logger(log_level: str) -> logging.Logger:
    """Initialize a simple console logger."""
    level = getattr(logging, log_level.upper(), logging.INFO)
    logger = logging.getLogger("gpt2giga")
    logger.setLevel(level)
    # Prevent upstream/uvicorn handlers from overriding our format
    logger.propagate = False

    formatter = logging.Formatter(
        fmt="%(asctime)s.%(msecs)03d [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    else:
        for h in logger.handlers:
            try:
                h.setFormatter(formatter)
            except Exception:
                pass

    return logger
