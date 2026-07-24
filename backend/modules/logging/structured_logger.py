import json
import logging
import sys
from datetime import datetime, timezone
from typing import Any

from .correlation import get_correlation_id


class StructuredLogger:
    def __init__(self, name: str, level: int = logging.INFO):
        self._name = name
        self._logger = logging.getLogger(name)
        self._logger.setLevel(level)
        if not self._logger.handlers:
            handler = logging.StreamHandler(sys.stdout)
            handler.setFormatter(logging.Formatter("%(message)s"))
            self._logger.addHandler(handler)

    def _log(self, level: int, event: str, **kwargs: Any) -> None:
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": logging.getLevelName(level).lower(),
            "logger": self._name,
            "event": event,
            "correlation_id": get_correlation_id(),
        }
        extra = {k: v for k, v in kwargs.items() if v is not None}
        if extra:
            record["data"] = extra
        self._logger.log(level, json.dumps(record, default=str))

    def info(self, event: str, **kwargs: Any) -> None:
        self._log(logging.INFO, event, **kwargs)

    def warning(self, event: str, **kwargs: Any) -> None:
        self._log(logging.WARNING, event, **kwargs)

    def error(self, event: str, **kwargs: Any) -> None:
        self._log(logging.ERROR, event, **kwargs)

    def debug(self, event: str, **kwargs: Any) -> None:
        self._log(logging.DEBUG, event, **kwargs)


_loggers: dict[str, StructuredLogger] = {}


def get_logger(name: str, level: int = logging.INFO) -> StructuredLogger:
    if name not in _loggers:
        _loggers[name] = StructuredLogger(name, level=level)
    return _loggers[name]
