# audiorag/__init__.py
"""AudioRAG: Production-grade RAG library for audio/DSP documentation."""

__version__ = "0.1.0"
__author__ = "Your Name"

# Import main components
from audiorag.core.exceptions import AudioRAGException
from audiorag.core.config import RAGConfig
from audiorag.core.logger import get_logger

__all__ = [
    "AudioRAGException",
    "RAGConfig",
    "get_logger",
]
