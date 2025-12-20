"""Test Pydantic configuration management."""

import pytest
import json
from pathlib import Path
from audiorag.core.config import (
    ChunkConfig,
    EmbeddingConfig,
    RAGConfig,
    get_lightweight_config,
    get_production_config,
)


def test_chunk_config_defaults():
    """Test ChunkConfig with defaults."""
    cfg = ChunkConfig()
    assert cfg.chunk_size == 300
    assert cfg.overlap == 50
    assert cfg.preserve_tables is True


def test_chunk_config_validation():
    """Test ChunkConfig validation."""
    with pytest.raises(ValueError):
        # overlap >= chunk_size should fail
        ChunkConfig(chunk_size=100, overlap=150)
    
    with pytest.raises(ValueError):
        # chunk_size < 50 should fail
        ChunkConfig(chunk_size=30)


def test_embedding_config():
    """Test EmbeddingConfig."""
    cfg = EmbeddingConfig(batch_size=64)
    assert cfg.batch_size == 64
    assert cfg.normalize_embeddings is True


def test_rag_config_default():
    """Test RAGConfig with all defaults."""
    cfg = RAGConfig()
    assert cfg.chunk_config.chunk_size == 300
    assert cfg.embedding_config.batch_size == 32
    assert cfg.log_level == "INFO"


def test_rag_config_to_json(tmp_path):
    """Test saving RAGConfig to JSON."""
    cfg = RAGConfig()
    json_file = tmp_path / "config.json"
    cfg.to_json(str(json_file))
    
    assert json_file.exists()
    with open(json_file) as f:
        data = json.load(f)
    assert "chunk_config" in data


def test_rag_config_from_json(tmp_path):
    """Test loading RAGConfig from JSON."""
    cfg = RAGConfig(log_level="DEBUG")
    json_file = tmp_path / "config.json"
    cfg.to_json(str(json_file))
    
    # Load it back
    cfg2 = RAGConfig.from_json(str(json_file))
    assert cfg2.log_level == "DEBUG"


def test_lightweight_config():
    """Test lightweight config for 8GB RAM systems."""
    cfg = get_lightweight_config()
    assert cfg.inference_config.load_in_8bit is True
    assert cfg.fine_tuning_config.batch_size == 4


def test_production_config():
    """Test production config."""
    cfg = get_production_config()
    assert cfg.enable_monitoring is True
    assert cfg.retrieval_config.rerank_results is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
