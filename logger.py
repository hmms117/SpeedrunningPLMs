import logging
import sys
from typing import Optional

try:
    from colorama import Fore, Style, init as colorama_init
    colorama_init(autoreset=True)
except Exception:  # pragma: no cover - colorama optional
    class Dummy:
        BLUE = GREEN = YELLOW = RED = ''
        RESET_ALL = BRIGHT = ''
    Fore = Style = Dummy()

_COLORS = {
    logging.DEBUG: Fore.BLUE,
    logging.INFO: Fore.GREEN,
    logging.WARNING: Fore.YELLOW,
    logging.ERROR: Fore.RED,
    logging.CRITICAL: Fore.RED + Style.BRIGHT,
}

class _ColorFormatter(logging.Formatter):
    """Formatter that injects ANSI colors based on log level."""

    def format(self, record: logging.LogRecord) -> str:
        color = _COLORS.get(record.levelno, '')
        reset = Style.RESET_ALL if hasattr(Style, 'RESET_ALL') else ''
        msg = super().format(record)
        return f"{color}{msg}{reset}"


def get_logger(name: str,
               log_file: Optional[str] = None,
               level: int = logging.INFO) -> logging.Logger:
    """Return a logger with colored console output and optional file logging."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(level)
    fmt = '%(asctime)s - %(levelname)s - %(message)s'
    datefmt = '%Y-%m-%d %H:%M:%S'

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(_ColorFormatter(fmt, datefmt=datefmt))
    logger.addHandler(console_handler)

    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(logging.Formatter(fmt, datefmt=datefmt))
        logger.addHandler(file_handler)

    return logger
