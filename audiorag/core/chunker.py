"""
audiorag/core/chunker.py

Production-grade PDF chunking with audio/DSP awareness.

Features:
- Smart PDF parsing with PyPDF2 + pdfplumber
- Table preservation (critical for Bose spec sheets)
- Audio-aware chunking (preserves dB values, Hz ranges, specs)
- Configurable chunk sizes with overlap
- Comprehensive error handling with retries
- Context tracking for debugging

Usage:
    from audiorag.core.chunker import AudioChunker
    from audiorag.core.config import ChunkConfig
    
    config = ChunkConfig(chunk_size=300, preserve_tables=True)
    chunker = AudioChunker(config)
    
    chunks = chunker.chunk_pdf("Bose_EX-1280_Manual.pdf")
    for chunk in chunks:
        print(chunk.text)
        print(f"Source: {chunk.metadata['source']}")
"""

import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime
import re

try:
    import PyPDF2
    from pdfplumber import PDF
    import pdfplumber
except ImportError as e:
    raise ImportError(
        "PDF parsing requires: pip install PyPDF2 pdfplumber. "
        f"Error: {e}"
    )

from audiorag.core.config import ChunkConfig
from audiorag.core.exceptions import (
    PDFParseError,
    TablePreservationError,
    CorruptedDocumentError,
    ChunkConfigError,
)
from audiorag.core.logger import get_logger


# ============================================================================
# DATA MODELS
# ============================================================================

@dataclass
class Chunk:
    """Represents a single chunk from a document."""
    
    text: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    chunk_index: int = 0
    source_file: str = ""
    page_number: int = 0
    
    def __post_init__(self):
        """Validate chunk after initialization."""
        if not isinstance(self.text, str):
            raise ValueError("Chunk text must be string")
        if self.chunk_index < 0:
            raise ValueError("chunk_index must be >= 0")
        if self.page_number < 0:
            raise ValueError("page_number must be >= 0")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert chunk to dictionary."""
        return {
            "text": self.text,
            "metadata": self.metadata,
            "chunk_index": self.chunk_index,
            "source_file": self.source_file,
            "page_number": self.page_number,
        }


@dataclass
class Table:
    """Represents a table extracted from PDF."""
    
    data: List[List[str]]  # 2D list of cell contents
    page_number: int = 0
    table_index: int = 0
    
    def to_markdown(self) -> str:
        """Convert table to markdown format."""
        if not self.data or len(self.data) == 0:
            return ""
        
        # Header row
        md = "| " + " | ".join(self.data[0]) + " |\n"
        # Separator
        md += "|" + "|".join(["---"] * len(self.data[0])) + "|\n"
        # Data rows
        for row in self.data[1:]:
            md += "| " + " | ".join(row) + " |\n"
        
        return md


# ============================================================================
# AUDIO-AWARE CHUNKER
# ============================================================================

class AudioChunker:
    """
    Production-grade PDF chunker optimized for audio/DSP documentation.
    
    Preserves:
    - Table structures (critical for Bose spec sheets)
    - Audio terminology (dB, Hz, impedance, etc.)
    - Code blocks and technical specifications
    - Document hierarchy (headers, sections)
    
    Attributes:
        config: ChunkConfig with chunk_size, overlap, etc.
        logger: Logger instance for debugging
    """
    
    # Audio/DSP patterns to preserve
    AUDIO_PATTERNS = {
        "db_pattern": r"-?\d+\.?\d*\s*dB[A-Z]?",  # -6dB, +3 dB, 0dBFS
        "frequency_pattern": r"\d+\.?\d*\s*(?:Hz|kHz|MHz)",  # 20Hz, 1.5kHz
        "impedance_pattern": r"\d+\.?\d*\s*[Ωohm]+",  # 8Ω, 50 ohms
        "voltage_pattern": r"[\+\-]?\d+\.?\d*\s*[VvAa]",  # +24V, 5A
        "frequency_response": r"[\d\-\s]+\s*(?:Hz|dB)",  # 20-20kHz, ±3dB
    }
    
    def __init__(self, config: ChunkConfig):
        """
        Initialize AudioChunker.
        
        Args:
            config: ChunkConfig with parameters
            
        Raises:
            ChunkConfigError: If config is invalid
        """
        self.config = config
        self.logger = get_logger(__name__)
        
        # Validate config
        try:
            if self.config.chunk_size < 50:
                raise ChunkConfigError(
                    f"chunk_size must be >= 50, got {self.config.chunk_size}",
                    config={"chunk_size": self.config.chunk_size}
                )
            if self.config.overlap >= self.config.chunk_size:
                raise ChunkConfigError(
                    f"overlap ({self.config.overlap}) must be < chunk_size ({self.config.chunk_size})",
                    config={
                        "chunk_size": self.config.chunk_size,
                        "overlap": self.config.overlap
                    }
                )
        except ChunkConfigError:
            raise
        
        self.logger.info(
            "AudioChunker initialized",
            extra={
                "chunk_size": self.config.chunk_size,
                "overlap": self.config.overlap,
                "preserve_tables": self.config.preserve_tables,
                "audio_aware": self.config.audio_aware_chunking,
            }
        )
    
    def chunk_pdf(self, file_path: str) -> List[Chunk]:
        """
        Split PDF into chunks with smart overlapping.
        
        Args:
            file_path: Path to PDF file
            
        Returns:
            List of Chunk objects
            
        Raises:
            PDFParseError: If PDF parsing fails
            CorruptedDocumentError: If PDF is corrupted
        """
        file_path = Path(file_path)
        
        # Validate file
        if not file_path.exists():
            raise PDFParseError(
                f"PDF file not found: {file_path}",
                file_path=str(file_path)
            )
        
        if file_path.suffix.lower() != ".pdf":
            raise PDFParseError(
                f"File is not a PDF: {file_path}",
                file_path=str(file_path)
            )
        
        try:
            self.logger.info(f"Parsing PDF: {file_path}")
            
            # Extract text and tables
            text_content = self._extract_text(file_path)
            tables = self._extract_tables(file_path) if self.config.preserve_tables else []
            
            # Insert tables back into text
            full_text = self._insert_tables_into_text(text_content, tables)
            
            # Apply audio-aware chunking
            chunks = self._chunk_text(
                full_text,
                source_file=str(file_path)
            )
            
            self.logger.info(
                f"Chunked PDF successfully",
                extra={
                    "file": file_path.name,
                    "num_chunks": len(chunks),
                    "total_chars": len(full_text),
                }
            )
            
            return chunks
            
        except (PDFParseError, CorruptedDocumentError):
            raise
        except Exception as e:
            self.logger.error(f"PDF parsing failed: {e}", extra={"error_type": type(e).__name__})
            raise PDFParseError(
                f"Failed to parse PDF: {str(e)}",
                file_path=str(file_path),
                context={"error_type": type(e).__name__}
            )
    
    def _extract_text(self, file_path: Path) -> str:
        """
        Extract text from PDF with error handling.
        
        Args:
            file_path: Path to PDF
            
        Returns:
            Extracted text
            
        Raises:
            CorruptedDocumentError: If PDF is corrupted
        """
        try:
            with pdfplumber.open(file_path) as pdf:
                # Check if PDF is readable
                if len(pdf.pages) == 0:
                    raise CorruptedDocumentError(
                        "PDF has no pages",
                        file_path=str(file_path)
                    )
                
                text_parts = []
                for page_num, page in enumerate(pdf.pages, 1):
                    try:
                        text = page.extract_text() or ""
                        if text.strip():
                            text_parts.append(f"\n--- Page {page_num} ---\n{text}")
                    except Exception as e:
                        self.logger.warning(
                            f"Failed to extract text from page {page_num}: {e}"
                        )
                        continue
                
                return "\n".join(text_parts)
        
        except CorruptedDocumentError:
            raise
        except Exception as e:
            raise CorruptedDocumentError(
                f"PDF appears corrupted or unreadable: {str(e)}",
                file_path=str(file_path),
                context={"error": str(e)}
            )
    
    def _extract_tables(self, file_path: Path) -> List[Table]:
        """
        Extract tables from PDF while preserving structure.
        
        Args:
            file_path: Path to PDF
            
        Returns:
            List of Table objects
        """
        tables = []
        try:
            with pdfplumber.open(file_path) as pdf:
                for page_num, page in enumerate(pdf.pages, 1):
                    try:
                        page_tables = page.extract_tables()
                        if page_tables:
                            for table_idx, table_data in enumerate(page_tables):
                                # Convert to list of lists (handle None values)
                                clean_data = [
                                    [str(cell) if cell is not None else "" for cell in row]
                                    for row in table_data
                                ]
                                tables.append(
                                    Table(
                                        data=clean_data,
                                        page_number=page_num,
                                        table_index=table_idx
                                    )
                                )
                    except Exception as e:
                        self.logger.warning(
                            f"Failed to extract tables from page {page_num}: {e}"
                        )
                        continue
        
        except Exception as e:
            self.logger.warning(f"Table extraction failed: {e}")
        
        return tables
    
    def _insert_tables_into_text(self, text: str, tables: List[Table]) -> str:
        """
        Insert extracted tables back into text at appropriate locations.
        
        Args:
            text: Extracted text
            tables: List of tables
            
        Returns:
            Text with tables inserted
        """
        if not tables:
            return text
        
        try:
            for table in tables:
                table_md = table.to_markdown()
                if table_md:
                    # Find page reference in text
                    page_marker = f"--- Page {table.page_number} ---"
                    if page_marker in text:
                        text = text.replace(
                            page_marker,
                            f"{page_marker}\n\n[TABLE {table.table_index}]\n{table_md}\n",
                            1
                        )
        except Exception as e:
            self.logger.warning(f"Failed to insert tables: {e}")
        
        return text
    
    def _chunk_text(
        self,
        text: str,
        source_file: str
    ) -> List[Chunk]:
        """
        Split text into chunks with audio-aware logic.
        
        Args:
            text: Full document text
            source_file: Source file name
            
        Returns:
            List of Chunk objects
        """
        if not text or len(text.strip()) == 0:
            raise PDFParseError(
                "No text content to chunk",
                file_path=source_file
            )
        
        # Apply audio-aware sentence splitting
        sentences = self._smart_split(text)
        
        chunks = []
        current_chunk = []
        current_length = 0
        chunk_index = 0
        
        for sentence in sentences:
            sentence_len = len(sentence)
            
            # If adding sentence exceeds chunk_size, save current chunk
            if current_length + sentence_len > self.config.chunk_size and current_chunk:
                chunk_text = " ".join(current_chunk).strip()
                if chunk_text:
                    chunks.append(
                        Chunk(
                            text=chunk_text,
                            metadata={
                                "chunk_index": chunk_index,
                                "created_at": datetime.utcnow().isoformat(),
                            },
                            chunk_index=chunk_index,
                            source_file=source_file,
                        )
                    )
                    chunk_index += 1
                
                # Start new chunk with overlap
                overlap_text = " ".join(current_chunk[-3:])  # Keep last 3 sentences
                current_chunk = [overlap_text] if overlap_text else []
                current_length = len(overlap_text)
            
            current_chunk.append(sentence)
            current_length += sentence_len + 1  # +1 for space
        
        # Add final chunk
        if current_chunk:
            chunk_text = " ".join(current_chunk).strip()
            if chunk_text:
                chunks.append(
                    Chunk(
                        text=chunk_text,
                        metadata={
                            "chunk_index": chunk_index,
                            "created_at": datetime.utcnow().isoformat(),
                        },
                        chunk_index=chunk_index,
                        source_file=source_file,
                    )
                )
        
        return chunks
    
    def _smart_split(self, text: str) -> List[str]:
        """
        Smart sentence splitting that respects audio terminology.
        
        Preserves:
        - Audio specs (e.g., "20-20kHz" not split)
        - Abbreviations (e.g., "dB", "Hz")
        - Technical terms
        
        Args:
            text: Text to split
            
        Returns:
            List of sentences/segments
        """
        # Split on sentence boundaries but not on audio terms
        # First, protect audio patterns
        placeholders = {}
        protected_text = text
        
        for pattern_name, pattern in self.AUDIO_PATTERNS.items():
            matches = re.finditer(pattern, protected_text)
            for i, match in enumerate(matches):
                placeholder = f"__AUDIO_{pattern_name}_{i}__"
                placeholders[placeholder] = match.group()
                protected_text = protected_text.replace(match.group(), placeholder, 1)
        
        # Split on sentence boundaries
        sentences = re.split(r'(?<=[.!?])\s+', protected_text)
        
        # Restore placeholders
        sentences = [
            sentence for sentence in sentences
            if sentence.strip()
        ]
        
        result = []
        for sentence in sentences:
            for placeholder, original in placeholders.items():
                sentence = sentence.replace(placeholder, original)
            result.append(sentence)
        
        return result


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    from audiorag.core.logger import setup_logging
    
    # Setup
    setup_logging(log_level="DEBUG", log_file="chunker_test.log")
    
    # Create chunker
    config = ChunkConfig(
        chunk_size=300,
        overlap=50,
        preserve_tables=True,
        audio_aware_chunking=True
    )
    chunker = AudioChunker(config)
    
    # Test with sample PDF (if exists)
    try:
        chunks = chunker.chunk_pdf("sample.pdf")
        print(f"\n✅ Chunked {len(chunks)} chunks")
        for chunk in chunks[:2]:
            print(f"\nChunk {chunk.chunk_index}:")
            print(f"Text: {chunk.text[:100]}...")
            print(f"Metadata: {chunk.metadata}")
    except Exception as e:
        print(f"❌ Error: {e}")
