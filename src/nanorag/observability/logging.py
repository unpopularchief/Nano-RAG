"""JSON logging with credential redaction for application log handlers."""

from __future__ import annotations

import json
import logging
import os
import re
from datetime import UTC, datetime

_CREDENTIAL = re.compile(
    r"(?i)\b(authorization|api[_-]?key|access[_-]?token|password|secret)"
    r"\s*[:=]\s*[^\s,}]+|\bBearer\s+[^\s,}]+"
)


def redact(message: str) -> str:
    """Mask credential assignments and configured API key values in text."""
    message = _CREDENTIAL.sub("[REDACTED]", message)
    for name, value in os.environ.items():
        if value and name.upper().endswith(("KEY", "TOKEN", "SECRET", "PASSWORD")):
            message = message.replace(value, "[REDACTED]")
    return message


class JsonFormatter(logging.Formatter):
    """Emit one JSON object per record, without exception messages or traces."""

    def format(self, record: logging.LogRecord) -> str:
        """Render a safe record for a console or file handler."""
        message = record.getMessage()
        try:
            event = json.loads(message)
        except ValueError:
            event = None
        payload: dict[str, object] = {
            "time": datetime.fromtimestamp(record.created, UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
        }
        if isinstance(event, dict) and "event" in event:
            payload.update(
                {
                    key: redact(value) if isinstance(value, str) else value
                    for key, value in event.items()
                }
            )
        else:
            payload["message"] = redact(message)
        if record.exc_info and record.exc_info[0] is not None:
            payload["error_type"] = record.exc_info[0].__name__
        return json.dumps(payload, ensure_ascii=False)
