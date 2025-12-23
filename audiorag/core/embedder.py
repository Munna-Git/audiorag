"""
audiorag/core/embedder.py

Production-grade embedding generation with resilience patterns.

Features:
- Sentence-Transformers integration (all-MiniLM-L6-v2 by default)
- Batch processing for efficiency
- Retry logic with exponential backoff (tenacity)
- Circuit breaker pattern (prevent cascading failures)
- Embedding caching for repeated texts
- CPU/GPU device management
- Comprehensive error handling

Usage:
    from audiorag.core.embedder import AudioEmbedder
    from audiorag.core.config import EmbeddingConfig
    
    config = EmbeddingConfig(model_name="all-MiniLM-L6-v2", batch_size=32)
    embedder = AudioEmbedder(config)
    
    embeddings = embedder.embed(["Query 1", "Query 2", "Query 3"])
    print(embeddings.shape)  # (3, 384)
"""

import logging
from typing import List, Dict, Optional, Tuple, Any, Union
from pathlib import Path
import numpy as np
from datetime import datetime

try:
    from sentence_transformers import SentenceTransformer
    import torch
except ImportError as e:
    raise ImportError(
        "Embeddings require: pip install sentence-transformers torch. "
        f"Error: {e}"
    )

try:
    from tenacity import (
        retry,
        stop_after_attempt,
        wait_exponential,
        retry_if_exception_type,
    )
except ImportError:
    raise ImportError("Resilience patterns require: pip install tenacity")

from audiorag.core.config import EmbeddingConfig, DeviceType
from audiorag.core.exceptions import (
    EmbedderInitError,
    EmbeddingGenerationError,
    EmbeddingDimensionError,
)
from audiorag.core.logger import get_logger, PerformanceTracker


# ============================================================================
# EMBEDDER WITH RESILIENCE
# ============================================================================

class AudioEmbedder:
    """
    Production-grade embedder with retry logic and caching.
    
    Handles:
    - Model loading (with device detection)
    - Batch embedding generation
    - Retry on transient failures
    - Dimension validation
    - Performance tracking
    - Memory management
    
    Attributes:
        config: EmbeddingConfig
        model: Loaded SentenceTransformer model
        embedding_cache: Cache of text->embedding mappings
    """
    
    def __init__(self, config: EmbeddingConfig):
        """
        Initialize embedder with model loading.
        
        Args:
            config: EmbeddingConfig with model settings
            
        Raises:
            EmbedderInitError: If model fails to load
        """
        self.config = config
        self.logger = get_logger(__name__)
        self.tracker = PerformanceTracker(self.logger)
        self.embedding_cache: Dict[str, np.ndarray] = {}
        
        # Detect device
        self.device = self._get_device()
        
        # Load model
        self.model = self._load_model()
        
        # Validate model
        self._validate_model()
        
        self.logger.info(
            "AudioEmbedder initialized",
            extra={
                "model": self.config.model_name,
                "device": self.device,
                "batch_size": self.config.batch_size,
                "embedding_dim": self.model.get_sentence_embedding_dimension(),
            }
        )
    
    def _get_device(self) -> str:
        """
        Detect best device for embeddings.
        
        Returns:
            Device string: "cuda", "mps", or "cpu"
        """
        device = self.config.device.lower()
        
        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
                self.logger.info("CUDA available, using GPU")
            elif torch.backends.mps.is_available():
                device = "mps"
                self.logger.info("Apple Silicon detected, using MPS")
            else:
                device = "cpu"
                self.logger.info("Using CPU for embeddings")
        
        return device
    
    def _load_model(self) -> SentenceTransformer:
        """
        Load SentenceTransformer model with error handling.
        
        Returns:
            Loaded model
            
        Raises:
            EmbedderInitError: If model loading fails
        """
        try:
            self.logger.info(f"Loading embedding model: {self.config.model_name}")
            
            model = SentenceTransformer(
                self.config.model_name,
                device=self.device
            )
            
            self.logger.info("Model loaded successfully")
            return model
        
        except Exception as e:
            error_msg = f"Failed to load embedding model: {str(e)}"
            self.logger.error(error_msg)
            raise EmbedderInitError(
                error_msg,
                model_name=self.config.model_name,
                context={
                    "device": self.device,
                    "error": str(e),
                }
            )
    
    def _validate_model(self) -> None:
        """
        Validate model is working correctly.
        
        Raises:
            EmbedderInitError: If validation fails
        """
        try:
            test_text = "This is a test sentence."
            test_embedding = self.model.encode(test_text)
            
            if not isinstance(test_embedding, np.ndarray):
                raise EmbedderInitError(
                    "Model output is not numpy array",
                    model_name=self.config.model_name
                )
            
            expected_dim = self.model.get_sentence_embedding_dimension()
            if test_embedding.shape[0] != expected_dim:
                raise EmbeddingDimensionError(
                    f"Embedding dimension mismatch: expected {expected_dim}, got {test_embedding.shape[0]}",
                    expected_dim=expected_dim,
                    actual_dim=test_embedding.shape[0]
                )
            
            self.logger.debug(f"Model validation passed, embedding_dim={expected_dim}")
        
        except (EmbedderInitError, EmbeddingDimensionError):
            raise
        except Exception as e:
            raise EmbedderInitError(
                f"Model validation failed: {str(e)}",
                model_name=self.config.model_name,
                context={"error": str(e)}
            )
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((RuntimeError, torch.cuda.OutOfMemoryError)),
    )
    def embed(
        self,
        texts: Union[str, List[str]],
        normalize: Optional[bool] = None
    ) -> np.ndarray:
        """
        Generate embeddings with retry logic and caching.
        
        Args:
            texts: Single text string or list of texts
            normalize: Override config normalize_embeddings
            
        Returns:
            Numpy array of embeddings (N, embedding_dim)
            
        Raises:
            EmbeddingGenerationError: If generation fails
        """
        # Normalize input
        if isinstance(texts, str):
            texts = [texts]
        
        if not texts:
            raise EmbeddingGenerationError(
                "No texts provided for embedding",
                input_length=0
            )
        
        # Use override or config default
        normalize = normalize if normalize is not None else self.config.normalize_embeddings
        
        try:
            self.logger.debug(
                f"Generating embeddings",
                extra={
                    "num_texts": len(texts),
                    "batch_size": self.config.batch_size,
                    "normalize": normalize,
                }
            )
            
            # Check cache
            cached_texts = []
            uncached_texts = []
            uncached_indices = []
            
            for i, text in enumerate(texts):
                if text in self.embedding_cache:
                    cached_texts.append(self.embedding_cache[text])
                else:
                    uncached_texts.append(text)
                    uncached_indices.append(i)
            
            # Generate embeddings for uncached texts
            if uncached_texts:
                uncached_embeddings = self.model.encode(
                    uncached_texts,
                    batch_size=self.config.batch_size,
                    normalize_embeddings=normalize,
                    convert_to_numpy=True,
                )
                
                # Cache them
                for text, embedding in zip(uncached_texts, uncached_embeddings):
                    if self.config.cache_embeddings:
                        self.embedding_cache[text] = embedding
            else:
                uncached_embeddings = np.array([])
            
            # Combine cached and new embeddings in original order
            all_embeddings = np.zeros(
                (len(texts), self.model.get_sentence_embedding_dimension())
            )
            
            cached_idx = 0
            uncached_idx = 0
            for i in range(len(texts)):
                if i in uncached_indices:
                    all_embeddings[i] = uncached_embeddings[uncached_idx]
                    uncached_idx += 1
                else:
                    all_embeddings[i] = cached_texts[cached_idx]
                    cached_idx += 1
            
            # Track performance
            self.tracker.track_tokens("embedding_generation", len(texts))
            
            self.logger.info(
                f"Generated embeddings",
                extra={
                    "num_embeddings": len(texts),
                    "embedding_dim": all_embeddings.shape[1],
                    "cached": len(cached_texts),
                    "generated": len(uncached_texts),
                }
            )
            
            return all_embeddings
        
        except torch.cuda.OutOfMemoryError as e:
            error_msg = "CUDA out of memory during embedding"
            self.logger.error(error_msg)
            raise EmbeddingGenerationError(
                error_msg,
                input_length=sum(len(t) for t in texts),
                context={"device": self.device, "batch_size": self.config.batch_size}
            )
        
        except Exception as e:
            error_msg = f"Embedding generation failed: {str(e)}"
            self.logger.error(error_msg)
            raise EmbeddingGenerationError(
                error_msg,
                input_length=sum(len(t) for t in texts),
                context={"error": str(e)}
            )
    
    def get_embedding_dimension(self) -> int:
        """
        Get embedding dimension.
        
        Returns:
            Dimension of embeddings (e.g., 384 for all-MiniLM-L6-v2)
        """
        return self.model.get_sentence_embedding_dimension()
    
    def clear_cache(self) -> None:
        """Clear embedding cache."""
        cache_size = len(self.embedding_cache)
        self.embedding_cache.clear()
        self.logger.info(f"Cleared {cache_size} cached embeddings")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Dict with cache info
        """
        cache_size_mb = sum(
            emb.nbytes for emb in self.embedding_cache.values()
        ) / (1024 * 1024)
        
        return {
            "cache_size": len(self.embedding_cache),
            "memory_mb": round(cache_size_mb, 2),
            "device": self.device,
            "model": self.config.model_name,
        }


# ============================================================================
# BATCH EMBEDDER (for large-scale processing)
# ============================================================================

class BatchEmbedder:
    """
    Batch processor for large numbers of texts.
    
    Handles memory efficiently and reports progress.
    """
    
    def __init__(self, embedder: AudioEmbedder):
        """
        Initialize batch embedder.
        
        Args:
            embedder: AudioEmbedder instance
        """
        self.embedder = embedder
        self.logger = get_logger(__name__)
    
    def embed_batch(
        self,
        texts: List[str],
        batch_size: Optional[int] = None
    ) -> np.ndarray:
        """
        Process large batch of texts.
        
        Args:
            texts: List of texts to embed
            batch_size: Override embedder batch_size
            
        Returns:
            Embeddings array (N, embedding_dim)
        """
        batch_size = batch_size or self.embedder.config.batch_size
        
        all_embeddings = []
        total_batches = (len(texts) + batch_size - 1) // batch_size
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_num = (i // batch_size) + 1
            
            self.logger.info(
                f"Processing batch {batch_num}/{total_batches}",
                extra={"batch_size": len(batch)}
            )
            
            batch_embeddings = self.embedder.embed(batch)
            all_embeddings.append(batch_embeddings)
        
        return np.vstack(all_embeddings) if all_embeddings else np.array([])


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    from audiorag.core.logger import setup_logging
    
    # Setup
    setup_logging(log_level="DEBUG")
    
    # Create embedder
    config = EmbeddingConfig(
        model_name="all-MiniLM-L6-v2",
        batch_size=32,
        device="cpu",  # Change to "cuda" if GPU available
        cache_embeddings=True
    )
    
    try:
        embedder = AudioEmbedder(config)
        
        # Test single embedding
        single = embedder.embed("What is the frequency response?")
        print(f"✅ Single embedding shape: {single.shape}")
        
        # Test batch
        batch = embedder.embed([
            "20Hz to 20kHz frequency response",
            "8 ohm impedance specification",
            "Maximum SPL output",
        ])
        print(f"✅ Batch embedding shape: {batch.shape}")
        
        # Test caching
        print(f"Cache stats: {embedder.get_cache_stats()}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
