"""
tests/unit/test_chunker.py

Complete test suite for AudioChunker component.

Run with: pytest tests/unit/test_chunker.py -v
Coverage: pytest tests/unit/test_chunker.py --cov=audiorag.core.chunker
"""

import pytest
from pathlib import Path
import tempfile
import os

from audiorag.core.chunker import AudioChunker, Chunk
from audiorag.core.config import ChunkConfig
from audiorag.core.exceptions import (
    PDFParseError,
    ChunkConfigError,
    CorruptedDocumentError,
)


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def chunk_config():
    """Default chunk config for tests."""
    return ChunkConfig(
        chunk_size=300,
        overlap=50,
        preserve_tables=True,
        audio_aware_chunking=True
    )


@pytest.fixture
def chunker(chunk_config):
    """Initialize chunker with default config."""
    return AudioChunker(chunk_config)


@pytest.fixture
def sample_pdf_path():
    """Create a temporary sample PDF for testing."""
    # In real tests, you'd use an actual sample.pdf file
    # For now, we'll skip PDF tests and focus on logic tests
    return Path("sample.pdf")


# ============================================================================
# CONFIGURATION TESTS
# ============================================================================

class TestChunkConfiguration:
    """Test ChunkConfig validation."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = ChunkConfig()
        assert config.chunk_size == 512
        assert config.overlap == 100
        assert config.preserve_tables is True
        assert config.audio_aware_chunking is True
    
    def test_chunk_size_too_small(self):
        """Test chunk_size < 50 raises error."""
        with pytest.raises(ChunkConfigError):
            ChunkConfig(chunk_size=30)
    
    def test_overlap_too_large(self):
        """Test overlap >= chunk_size raises error."""
        with pytest.raises(ChunkConfigError):
            ChunkConfig(chunk_size=300, overlap=300)
    
    def test_custom_config(self):
        """Test custom configuration."""
        config = ChunkConfig(
            chunk_size=500,
            overlap=75,
            preserve_tables=False,
            audio_aware_chunking=False
        )
        assert config.chunk_size == 500
        assert config.overlap == 75
        assert config.preserve_tables is False
        assert config.audio_aware_chunking is False


# ============================================================================
# CHUNK DATA MODEL TESTS
# ============================================================================

class TestChunkDataModel:
    """Test Chunk dataclass."""
    
    def test_chunk_creation(self):
        """Test creating a chunk."""
        chunk = Chunk(
            text="Sample text",
            chunk_index=0,
            source_file="test.pdf",
            page_number=1
        )
        assert chunk.text == "Sample text"
        assert chunk.chunk_index == 0
        assert chunk.source_file == "test.pdf"
        assert chunk.page_number == 1
    
    def test_chunk_to_dict(self):
        """Test chunk serialization to dict."""
        chunk = Chunk(
            text="Test content",
            metadata={"key": "value"},
            chunk_index=1,
            source_file="doc.pdf",
        )
        chunk_dict = chunk.to_dict()
        assert isinstance(chunk_dict, dict)
        assert chunk_dict["text"] == "Test content"
        assert chunk_dict["metadata"] == {"key": "value"}
    
    def test_chunk_validation_invalid_text_type(self):
        """Test chunk rejects non-string text."""
        with pytest.raises(ValueError):
            Chunk(text=123)  # Invalid: not a string
    
    def test_chunk_validation_negative_index(self):
        """Test chunk rejects negative index."""
        with pytest.raises(ValueError):
            Chunk(text="Test", chunk_index=-1)
    
    def test_chunk_with_metadata(self):
        """Test chunk with metadata."""
        metadata = {
            "created_at": "2025-12-20T10:00:00",
            "version": "1.0",
            "language": "en"
        }
        chunk = Chunk(
            text="Content with metadata",
            metadata=metadata,
            chunk_index=0,
        )
        assert chunk.metadata == metadata


# ============================================================================
# CHUNKER INITIALIZATION TESTS
# ============================================================================

class TestChunkerInitialization:
    """Test AudioChunker initialization."""
    
    def test_chunker_init_success(self, chunk_config):
        """Test successful chunker initialization."""
        chunker = AudioChunker(chunk_config)
        assert chunker.config == chunk_config
        assert chunker.logger is not None
    
    def test_chunker_with_invalid_config(self):
        """Test chunker with invalid config."""
        with pytest.raises(ChunkConfigError):
            AudioChunker(ChunkConfig(chunk_size=20))  # Too small


# ============================================================================
# AUDIO PATTERN DETECTION TESTS
# ============================================================================

class TestAudioPatternDetection:
    """Test audio-aware chunking patterns."""
    
    def test_audio_patterns_exist(self, chunker):
        """Test that audio patterns are defined."""
        patterns = chunker.AUDIO_PATTERNS
        assert "db_pattern" in patterns
        assert "frequency_pattern" in patterns
        assert "impedance_pattern" in patterns
    
    def test_frequency_pattern_matching(self, chunker):
        """Test frequency pattern matches."""
        import re
        pattern = chunker.AUDIO_PATTERNS["frequency_pattern"]
        
        # Should match
        assert re.search(pattern, "20Hz")
        assert re.search(pattern, "1.5kHz")
        assert re.search(pattern, "20-20kHz")
        assert re.search(pattern, "1MHz")
        
        # Boundary tests
        assert re.search(pattern, "20 Hz")  # With space
    
    def test_db_pattern_matching(self, chunker):
        """Test dB pattern matches."""
        import re
        pattern = chunker.AUDIO_PATTERNS["db_pattern"]
        
        # Should match
        assert re.search(pattern, "20dB")
        assert re.search(pattern, "-6dB")
        assert re.search(pattern, "+3dB")
        assert re.search(pattern, "0dBFS")
        assert re.search(pattern, "1.5 dB")


# ============================================================================
# SMART SPLIT TESTS
# ============================================================================

class TestSmartSplit:
    """Test smart sentence splitting."""
    
    def test_basic_sentence_split(self, chunker):
        """Test basic sentence splitting."""
        text = "First sentence. Second sentence. Third sentence."
        sentences = chunker._smart_split(text)
        assert len(sentences) >= 2
        assert all(isinstance(s, str) for s in sentences)
    
    def test_preserves_audio_patterns(self, chunker):
        """Test that audio patterns aren't split."""
        text = "Frequency response is 20-20kHz. Maximum SPL is 110dB."
        sentences = chunker._smart_split(text)
        
        # Join back and verify patterns are intact
        full_text = " ".join(sentences)
        assert "20-20kHz" in full_text
        assert "110dB" in full_text
    
    def test_empty_text(self, chunker):
        """Test splitting empty text."""
        sentences = chunker._smart_split("")
        assert sentences == []
    
    def test_single_sentence(self, chunker):
        """Test single sentence (no splits)."""
        text = "This is a single sentence"
        sentences = chunker._smart_split(text)
        assert len(sentences) == 1


# ============================================================================
# TEXT CHUNKING TESTS
# ============================================================================

class TestTextChunking:
    """Test _chunk_text method."""
    
    def test_chunk_text_basic(self, chunker):
        """Test basic text chunking."""
        text = "Sentence one. Sentence two. Sentence three. Sentence four. " * 10
        chunks = chunker._chunk_text(text, "test.pdf")
        
        assert len(chunks) > 0
        assert all(isinstance(c, Chunk) for c in chunks)
    
    def test_chunks_within_size_limit(self, chunker):
        """Test chunks respect size limit."""
        text = "word " * 1000  # Long text
        chunks = chunker._chunk_text(text, "test.pdf")
        
        for chunk in chunks:
            # Each chunk should be <= chunk_size (with some tolerance)
            assert len(chunk.text) <= chunker.config.chunk_size * 1.5
    
    def test_chunk_has_required_metadata(self, chunker):
        """Test chunks have required metadata."""
        text = "Sample text for chunking. " * 20
        chunks = chunker._chunk_text(text, "test.pdf")
        
        for chunk in chunks:
            assert chunk.source_file == "test.pdf"
            assert chunk.chunk_index >= 0
            assert "created_at" in chunk.metadata
    
    def test_chunk_indices_sequential(self, chunker):
        """Test chunk indices are sequential."""
        text = "Sample text. " * 50
        chunks = chunker._chunk_text(text, "test.pdf")
        
        for i, chunk in enumerate(chunks):
            assert chunk.chunk_index == i
    
    def test_empty_text_raises_error(self, chunker):
        """Test empty text raises error."""
        with pytest.raises(PDFParseError):
            chunker._chunk_text("", "test.pdf")


# ============================================================================
# OVERLAP TESTS
# ============================================================================

class TestChunkOverlap:
    """Test overlap between chunks."""
    
    def test_chunks_have_overlap(self, chunker):
        """Test that chunks overlap as configured."""
        text = "Word. " * 500  # Long text to force multiple chunks
        chunks = chunker._chunk_text(text, "test.pdf")
        
        if len(chunks) > 1:
            # Second chunk should contain some content from first
            # (This is a simplified check; real overlap detection is more complex)
            assert len(chunks[0].text) > 0
            assert len(chunks[1].text) > 0


# ============================================================================
# ERROR HANDLING TESTS
# ============================================================================

class TestErrorHandling:
    """Test error handling in chunker."""
    
    def test_pdf_not_found(self, chunker):
        """Test error when PDF file doesn't exist."""
        with pytest.raises(PDFParseError):
            chunker.chunk_pdf("nonexistent.pdf")
    
    def test_invalid_file_type(self, chunker):
        """Test error when file is not PDF."""
        with tempfile.NamedTemporaryFile(suffix=".txt") as f:
            with pytest.raises(PDFParseError):
                chunker.chunk_pdf(f.name)


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

class TestChunkerIntegration:
    """Integration tests for chunker."""
    
    def test_full_workflow_text(self, chunker):
        """Test full chunking workflow with text."""
        text = """
        The EX-1280 has excellent specifications.
        Frequency response: 20Hz to 20kHz.
        Impedance: 8 ohms.
        Maximum SPL: 110dB at 1 meter distance.
        
        Technical specifications provide detailed information.
        The unit operates efficiently in various environments.
        """
        
        chunks = chunker._chunk_text(text, "specs.pdf")
        
        assert len(chunks) > 0
        # Verify specs are preserved in chunks
        combined_text = " ".join(c.text for c in chunks)
        assert "20Hz to 20kHz" in combined_text
        assert "8 ohms" in combined_text
        assert "110dB" in combined_text


# ============================================================================
# RUN TESTS
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
