"""Test custom exception hierarchy."""

import pytest
from audiorag.core.exceptions import (
    PDFParseError,
    EmbeddingGenerationError,
    SearchError,
    ModelLoadError,
    InvalidConfigError,
    exception_to_json,
    get_error_code_description,
)


def test_pdf_parse_error():
    """Test PDFParseError with context."""
    exc = PDFParseError(
        "File appears to be corrupted",
        file_path="/path/to/file.pdf"
    )
    assert exc.error_code == "CHUNK_001"
    assert "file_path" in exc.context
    assert exc.timestamp is not None


def test_embedding_generation_error_with_retry():
    """Test EmbeddingGenerationError with retry count."""
    exc = EmbeddingGenerationError(
        "CUDA out of memory",
        input_length=2048,
        retry_count=2
    )
    assert exc.error_code == "EMB_002"
    assert exc.context["retry_count"] == 2


def test_exception_to_json():
    """Test converting exception to JSON."""
    exc = ModelLoadError("Model file not found", model_name="phi-2")
    json_str = exception_to_json(exc)
    assert "INF_001" in json_str
    assert "phi-2" in json_str


def test_error_code_description():
    """Test getting human-readable error descriptions."""
    desc = get_error_code_description("CHUNK_001")
    assert "PDF parsing" in desc
    assert "corrupted" in desc


def test_search_error():
    """Test SearchError exception."""
    exc = SearchError(
        "Vector search failed",
        query_text="What is audio compression?",
        results_count=5
    )
    assert exc.error_code == "RET_002"
    assert exc.context["results_count"] == 5


def test_invalid_config_error():
    """Test InvalidConfigError exception."""
    exc = InvalidConfigError(
        "Invalid chunk size",
        param_name="chunk_size",
        expected_type="int"
    )
    assert exc.error_code == "CFG_001"
    assert exc.context["param_name"] == "chunk_size"


def test_all_exception_codes_have_descriptions():
    """Ensure all exception codes have descriptions."""
    error_codes = [
        "CHUNK_001", "CHUNK_002", "CHUNK_003", "CHUNK_004",
        "EMB_001", "EMB_002", "EMB_003",
        "RET_001", "RET_002", "RET_003", "RET_004",
        "ADP_001", "ADP_002", "ADP_003", "ADP_004",
        "INF_001", "INF_002", "INF_003", "INF_004",
        "CFG_001", "CFG_002", "CFG_003", "CFG_004",
        "VAL_001", "VAL_002",
        "SYS_001", "SYS_002",
    ]
    for code in error_codes:
        desc = get_error_code_description(code)
        assert desc != "Unknown error", f"Missing description for {code}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
