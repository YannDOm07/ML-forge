"""Gestion centralisée de la journalisation.

Fournit un système de logs lisible, traçable et configurable.
"""

import logging
import sys
from pathlib import Path
from datetime import datetime


def setup_logging(log_dir: Path = None, log_level: str = "INFO") -> logging.Logger:
    """Configure le système de logging global."""
    if log_dir is None:
        log_dir = Path(__file__).parent.parent.parent / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("ml_forge")
    logger.setLevel(getattr(logging, log_level))

    # Éviter les handlers en double
    if logger.handlers:
        return logger

    # Format lisible
    formatter = logging.Formatter(
        "%(asctime)s | %(name)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler
    log_file = log_dir / f"ml_forge_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


# Instance globale
logger = setup_logging()
