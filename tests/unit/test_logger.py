"""Test logging framework."""

import pytest
import json
from pathlib import Path
from audiorag.core.logger import (
    get_logger,
    setup_logging,
    PerformanceTracker,
    AudioRAGLogger,
)


def test_get_logger():
    """Test getting a logger instance."""
    logger = get_logger("test_module")
    assert logger is not None
    assert isinstance(logger, AudioRAGLogger)


def test_logger_info(capsys):
    """Test info logging."""
    logger = get_logger("test_info")
    logger.info("Test message", extra={"key": "value"})
    captured = capsys.readouterr()
    assert "Test message" in captured.out


def test_logger_error(capsys):
    """Test error logging."""
    logger = get_logger("test_error")
    logger.error("Error message", extra={"error_code": "TEST_001"})
    captured = capsys.readouterr()
    assert "Error message" in captured.out


def test_logger_with_file(tmp_path):
    """Test logging to file."""
    log_file = tmp_path / "test.log"
    logger = AudioRAGLogger("test_file", log_file=str(log_file))
    logger.info("Test message")
    
    assert log_file.exists()
    with open(log_file) as f:
        content = f.read()
    assert "Test message" in content


def test_performance_tracker():
    """Test performance tracking."""
    logger = get_logger("test_perf")
    tracker = PerformanceTracker(logger)
    
    tracker.track_latency("embedding", 1234.5)
    tracker.track_tokens("generation", 42)
    tracker.track_memory("inference", 512.3)
    
    summary = tracker.get_summary()
    assert "embedding_latency_ms" in summary
    assert summary["generation_tokens"] == 42
    assert summary["inference_memory_mb"] == 512.3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
