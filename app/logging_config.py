# app/logging_config.py
from __future__ import annotations
import json, logging, os
from datetime import datetime, timezone
from logging.config import dictConfig

_STD_KEYS = {
    "name","msg","args","levelname","levelno","pathname","filename","module",
    "exc_info","exc_text","stack_info","lineno","funcName","created","msecs",
    "relativeCreated","thread","threadName","processName","process","message"
}

class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        base = {
            "ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "lvl": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        # add all extra-columns
        extras = {k: v for k, v in record.__dict__.items() if k not in _STD_KEYS}
        if record.exc_info:
            base["exc"] = self.formatException(record.exc_info)
        base.update(extras)
        return json.dumps(base, ensure_ascii=False)

def setup_logging() -> None:
    level = os.getenv("LOG_LEVEL", "INFO").upper()
    log_file = os.getenv("LOG_FILE", "logs/app.log")

    # directory for log file
    log_dir = os.path.dirname(log_file) or "."
    os.makedirs(log_dir, exist_ok=True)

    dictConfig({
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "json": {"()": JsonFormatter},
        },
        "handlers": {
            "stdout": {
                "class": "logging.StreamHandler",
                "stream": "ext://sys.stdout",
                "formatter": "json",
            },
            "file": {
                "class": "logging.handlers.RotatingFileHandler",
                "filename": log_file,
                "maxBytes": 5_000_000,
                "backupCount": 3,
                "encoding": "utf-8",
                "formatter": "json",
            },
        },
        "root": {"level": level, "handlers": ["stdout", "file"]},
        "loggers": {
            "uvicorn":        {"level": level, "handlers": ["stdout", "file"], "propagate": False},
            "uvicorn.error":  {"level": level, "handlers": ["stdout", "file"], "propagate": False},
            "uvicorn.access": {"level": level, "handlers": ["stdout", "file"], "propagate": False},
        },
    })

