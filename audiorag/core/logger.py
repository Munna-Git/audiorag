"""
audiorag/core/logger.py

Structured logging framework for AudioRAG.
Every operation is logged with context for debugging and monitoring.

Features:
- Structured JSON logging for log aggregation
- File + console output
- Performance tracking (latency, tokens, memory)
- Error tracking with context
- Log rotation

Usage:
    from audiorag.core.logger import get_logger
    
    logger = get_logger(__name__)
    logger.info("Processing document", extra={"doc_size": 1024})
    logger.error("Retrieval failed", extra={"query": "...", "error_code": "RET_001"})
"""

import logging
import logging.handlers
import json
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any
import sys


# ============================================================================
# CUSTOM FORMATTERS
# ============================================================================

class JSONFormatter(logging.Formatter):
    """Format logs as JSON for structured logging / log aggregation."""
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        log_data = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "function": record.funcName,
            "line": record.lineno,
            "message": record.getMessage(),
        }
        
        # Include extra fields if provided
        if hasattr(record, "extra") and isinstance(record.extra, dict):
            log_data.update(record.extra)
        
        # Include exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)
        
        return json.dumps(log_data)


class HumanReadableFormatter(logging.Formatter):
    """Format logs for human reading (console)."""
    
    COLORS = {
        "DEBUG": "\033[36m",      # Cyan
        "INFO": "\033[32m",       # Green
        "WARNING": "\033[33m",    # Yellow
        "ERROR": "\033[31m",      # Red
        "CRITICAL": "\033[35m",   # Magenta
        "RESET": "\033[0m"        # Reset
    }
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record for console output."""
        color = self.COLORS.get(record.levelname, "")
        reset = self.COLORS["RESET"]
        
        timestamp = datetime.fromtimestamp(record.created).strftime("%Y-%m-%d %H:%M:%S")
        
        # Base format
        log_msg = (
            f"{timestamp} - {record.name}:{record.lineno} - "
            f"{color}{record.levelname}{reset} - {record.getMessage()}"
        )
        
        # Add extra fields if provided
        if hasattr(record, "extra") and isinstance(record.extra, dict):
            extra_str = " | " + " | ".join(
                f"{k}={v}" for k, v in record.extra.items()
            )
            log_msg += extra_str
        
        # Include exception info if present
        if record.exc_info:
            log_msg += "\n" + self.formatException(record.exc_info)
        
        return log_msg


# ============================================================================
# CUSTOM LOGGER CLASS
# ============================================================================

class AudioRAGLogger:
    """Wrapper around Python logger with structured logging support."""
    
    def __init__(
        self,
        name: str,
        log_level: str = "INFO",
        log_file: Optional[str] = None,
        json_format: bool = False
    ):
        """
        Initialize AudioRAG logger.
        
        Args:
            name: Logger name (typically __name__)
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            log_file: Path to log file (None = console only)
            json_format: Use JSON formatting for file (struct logging)
        """
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, log_level.upper()))
        
        # Prevent duplicate handlers
        if self.logger.hasHandlers():
            return
        
        # Console handler (human-readable)
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, log_level.upper()))
        console_formatter = HumanReadableFormatter()
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)
        
        # File handler (if specified)
        if log_file:
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            
            file_handler = logging.handlers.RotatingFileHandler(
                log_path,
                maxBytes=10 * 1024 * 1024,  # 10MB
                backupCount=5
            )
            file_handler.setLevel(logging.DEBUG)  # File logs everything
            
            if json_format:
                file_formatter = JSONFormatter()
            else:
                file_formatter = HumanReadableFormatter()
            
            file_handler.setFormatter(file_formatter)
            self.logger.addHandler(file_handler)
    
    def debug(self, message: str, extra: Dict[str, Any] = None, **kwargs):
        """Log debug message."""
        record = self.logger.makeRecord(
            self.logger.name, logging.DEBUG, *self.logger._getCallerInfo(),
            message, args=(), exc_info=None, **kwargs
        )
        if extra:
            record.extra = extra
        self.logger.handle(record)
    
    def info(self, message: str, extra: Dict[str, Any] = None, **kwargs):
        """Log info message."""
        record = self.logger.makeRecord(
            self.logger.name, logging.INFO, *self.logger._getCallerInfo(),
            message, args=(), exc_info=None, **kwargs
        )
        if extra:
            record.extra = extra
        self.logger.handle(record)
    
    def warning(self, message: str, extra: Dict[str, Any] = None, **kwargs):
        """Log warning message."""
        record = self.logger.makeRecord(
            self.logger.name, logging.WARNING, *self.logger._getCallerInfo(),
            message, args=(), exc_info=None, **kwargs
        )
        if extra:
            record.extra = extra
        self.logger.handle(record)
    
    def error(self, message: str, extra: Dict[str, Any] = None, **kwargs):
        """Log error message."""
        record = self.logger.makeRecord(
            self.logger.name, logging.ERROR, *self.logger._getCallerInfo(),
            message, args=(), exc_info=None, **kwargs
        )
        if extra:
            record.extra = extra
        self.logger.handle(record)
    
    def critical(self, message: str, extra: Dict[str, Any] = None, **kwargs):
        """Log critical message."""
        record = self.logger.makeRecord(
            self.logger.name, logging.CRITICAL, *self.logger._getCallerInfo(),
            message, args=(), exc_info=None, **kwargs
        )
        if extra:
            record.extra = extra
        self.logger.handle(record)
    
    def exception(self, message: str, extra: Dict[str, Any] = None, **kwargs):
        """Log exception with traceback."""
        self.logger.exception(message, **kwargs)


# ============================================================================
# GLOBAL LOGGING SETUP
# ============================================================================

_loggers: Dict[str, AudioRAGLogger] = {}


def get_logger(
    name: str,
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    json_format: bool = False
) -> AudioRAGLogger:
    """
    Get or create a logger instance.
    
    Args:
        name: Logger name (typically __name__)
        log_level: Logging level
        log_file: Path to log file
        json_format: Use JSON formatting
        
    Returns:
        AudioRAGLogger instance
        
    Example:
        >>> logger = get_logger(__name__)
        >>> logger.info("Starting process", extra={"batch_size": 32})
    """
    if name not in _loggers:
        _loggers[name] = AudioRAGLogger(name, log_level, log_file, json_format)
    return _loggers[name]


def setup_logging(
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    json_format: bool = False
) -> None:
    """
    Setup logging for entire AudioRAG system.
    
    Args:
        log_level: Logging level for all loggers
        log_file: Path to log file
        json_format: Use JSON formatting for file
        
    Example:
        >>> setup_logging(log_level="DEBUG", log_file="audiorag.log")
    """
    for logger_name in _loggers:
        _loggers[logger_name] = AudioRAGLogger(
            logger_name, log_level, log_file, json_format
        )


# ============================================================================
# PERFORMANCE TRACKING
# ============================================================================

class PerformanceTracker:
    """Track performance metrics (latency, tokens, memory)."""
    
    def __init__(self, logger: AudioRAGLogger):
        self.logger = logger
        self.metrics: Dict[str, Any] = {}
    
    def track_latency(self, operation: str, duration_ms: float):
        """Track operation latency."""
        self.logger.info(
            f"{operation} completed",
            extra={"operation": operation, "latency_ms": duration_ms}
        )
        self.metrics[f"{operation}_latency_ms"] = duration_ms
    
    def track_tokens(self, operation: str, token_count: int):
        """Track token generation."""
        self.logger.info(
            f"{operation} generated tokens",
            extra={"operation": operation, "token_count": token_count}
        )
        self.metrics[f"{operation}_tokens"] = token_count
    
    def track_memory(self, operation: str, memory_mb: float):
        """Track memory usage."""
        self.logger.info(
            f"{operation} memory usage",
            extra={"operation": operation, "memory_mb": memory_mb}
        )
        self.metrics[f"{operation}_memory_mb"] = memory_mb
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of all tracked metrics."""
        return self.metrics.copy()


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    # Setup
    setup_logging(log_level="DEBUG", log_file="audiorag_test.log")
    logger = get_logger(__name__)
    
    # Test logging
    logger.debug("Debug message", extra={"test": "value"})
    logger.info("Info message", extra={"batch_size": 32})
    logger.warning("Warning message", extra={"retry_count": 3})
    logger.error("Error message", extra={"error_code": "TEST_001"})
    
    # Test performance tracking
    tracker = PerformanceTracker(logger)
    tracker.track_latency("embedding", 1234.5)
    tracker.track_tokens("generation", 42)
    tracker.track_memory("inference", 512.3)
    print("\nPerformance Summary:")
    print(json.dumps(tracker.get_summary(), indent=2))
