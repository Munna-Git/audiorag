"""
audiorag/core/config.py

Pydantic-based configuration management for AudioRAG.
All configs are type-safe and validated automatically.

Usage:
    from audiorag.core.config import ChunkConfig, EmbeddingConfig, RAGConfig
    
    chunk_cfg = ChunkConfig(chunk_size=500)  # Validated
    rag_cfg = RAGConfig.from_yaml("config.yaml")  # Load from file
"""

from pydantic import BaseModel, Field, validator, root_validator
from typing import List, Optional, Dict, Any
from enum import Enum
import yaml
from pathlib import Path


# ============================================================================
# ENUMS
# ============================================================================

class DeviceType(str, Enum):
    """Supported devices for inference."""
    CPU = "cpu"
    CUDA = "cuda"
    MPS = "mps"  # Apple Silicon
    AUTO = "auto"  # Auto-detect


class QuantizationType(str, Enum):
    """Supported quantization types."""
    NONE = "none"
    INT8 = "int8"
    INT4 = "int4"
    GGUF = "gguf"  # GGML format for LLAMA.CPP


# ============================================================================
# CHUNKING CONFIGURATION
# ============================================================================

class ChunkConfig(BaseModel):
    """Configuration for PDF chunking and document splitting."""
    
    chunk_size: int = Field(
        300,
        ge=50,
        le=2000,
        description="Number of characters per chunk"
    )
    overlap: int = Field(
        50,
        ge=0,
        le=500,
        description="Character overlap between chunks"
    )
    preserve_tables: bool = Field(
        True,
        description="Preserve table structures in output"
    )
    preserve_code_blocks: bool = Field(
        True,
        description="Preserve code blocks with formatting"
    )
    preserve_headers: bool = Field(
        True,
        description="Include document headers in chunks"
    )
    audio_aware_chunking: bool = Field(
        True,
        description="Apply audio/DSP-specific chunking rules (preserve specs)"
    )
    max_chunk_retries: int = Field(
        3,
        ge=1,
        le=10,
        description="Max retries on chunking failure"
    )
    
    @validator('overlap')
    def overlap_less_than_size(cls, v, values):
        """Ensure overlap < chunk_size."""
        if 'chunk_size' in values and v >= values['chunk_size']:
            raise ValueError("overlap must be < chunk_size")
        return v
    
    class Config:
        use_enum_values = True


# ============================================================================
# EMBEDDING CONFIGURATION
# ============================================================================

class EmbeddingConfig(BaseModel):
    """Configuration for embedding generation."""
    
    model_name: str = Field(
        "all-MiniLM-L6-v2",
        description="HuggingFace model identifier for embeddings"
    )
    device: DeviceType = Field(
        DeviceType.CPU,
        description="Device for embedding computation"
    )
    batch_size: int = Field(
        32,
        ge=1,
        le=512,
        description="Batch size for embedding generation"
    )
    normalize_embeddings: bool = Field(
        True,
        description="Normalize embeddings to unit length"
    )
    max_sequence_length: int = Field(
        512,
        ge=128,
        le=2048,
        description="Max tokens for embedding model"
    )
    max_retries: int = Field(
        3,
        ge=1,
        le=10,
        description="Max retry attempts for embedding generation"
    )
    retry_delay_seconds: float = Field(
        1.0,
        gt=0,
        le=30,
        description="Initial delay between retries (exponential backoff)"
    )
    cache_embeddings: bool = Field(
        True,
        description="Cache generated embeddings"
    )
    
    class Config:
        use_enum_values = True


# ============================================================================
# VECTOR DATABASE CONFIGURATION
# ============================================================================

class VectorDBConfig(BaseModel):
    """Configuration for vector database."""
    
    db_type: str = Field(
        "chroma",
        description="Vector DB type: 'chroma', 'faiss', 'pinecone'"
    )
    db_path: Optional[str] = Field(
        "./audiorag_vectordb",
        description="Local path for vector DB (for Chroma/FAISS)"
    )
    collection_name: str = Field(
        "audiorag_documents",
        description="Collection/index name in vector DB"
    )
    similarity_metric: str = Field(
        "cosine",
        description="Similarity metric: 'cosine', 'euclidean', 'dot'"
    )
    max_results: int = Field(
        5,
        ge=1,
        le=100,
        description="Max documents to retrieve per query"
    )
    similarity_threshold: float = Field(
        0.5,
        ge=0,
        le=1,
        description="Minimum similarity score for results"
    )
    
    class Config:
        use_enum_values = True


# ============================================================================
# RETRIEVAL CONFIGURATION
# ============================================================================

class RetrievalConfig(BaseModel):
    """Configuration for retrieval pipeline."""
    
    top_k: int = Field(
        5,
        ge=1,
        le=50,
        description="Number of results to retrieve"
    )
    similarity_threshold: float = Field(
        0.3,
        ge=0,
        le=1,
        description="Minimum similarity score"
    )
    return_metadata: bool = Field(
        True,
        description="Include document metadata in results"
    )
    rerank_results: bool = Field(
        False,
        description="Re-rank retrieved results using semantic similarity"
    )
    max_retrieval_retries: int = Field(
        2,
        ge=1,
        le=5,
        description="Max retry attempts for retrieval"
    )
    retrieval_timeout_seconds: float = Field(
        5.0,
        gt=0,
        le=30,
        description="Timeout for retrieval operation"
    )


# ============================================================================
# LORA ADAPTER CONFIGURATION
# ============================================================================

class LoRAConfig(BaseModel):
    """Configuration for LoRA fine-tuning."""
    
    r: int = Field(
        16,
        ge=1,
        le=256,
        description="LoRA rank"
    )
    lora_alpha: int = Field(
        32,
        ge=1,
        le=512,
        description="LoRA alpha (scaling factor)"
    )
    target_modules: List[str] = Field(
        ["q_proj", "v_proj"],
        description="Target modules for LoRA"
    )
    lora_dropout: float = Field(
        0.05,
        ge=0,
        le=1,
        description="Dropout for LoRA layers"
    )
    bias: str = Field(
        "none",
        description="Bias type: 'none', 'all', 'lora_only'"
    )
    task_type: str = Field(
        "CAUSAL_LM",
        description="Task type for fine-tuning"
    )
    
    @validator('lora_alpha')
    def alpha_gte_r(cls, v, values):
        """Ensure alpha >= r for good scaling."""
        if 'r' in values and v < values['r']:
            raise ValueError("lora_alpha should be >= r for good scaling")
        return v


# ============================================================================
# FINE-TUNING CONFIGURATION
# ============================================================================

class FineTuningConfig(BaseModel):
    """Configuration for fine-tuning pipeline."""
    
    base_model: str = Field(
        "microsoft/phi-2",
        description="Base model identifier"
    )
    learning_rate: float = Field(
        2e-4,
        gt=0,
        le=0.1,
        description="Learning rate for training"
    )
    num_epochs: int = Field(
        3,
        ge=1,
        le=10,
        description="Number of training epochs"
    )
    batch_size: int = Field(
        8,
        ge=1,
        le=128,
        description="Training batch size"
    )
    gradient_accumulation_steps: int = Field(
        1,
        ge=1,
        le=32,
        description="Gradient accumulation steps"
    )
    warmup_steps: int = Field(
        100,
        ge=0,
        le=5000,
        description="Number of warmup steps"
    )
    max_grad_norm: float = Field(
        1.0,
        gt=0,
        le=10,
        description="Max gradient norm for clipping"
    )
    save_steps: int = Field(
        100,
        ge=10,
        le=10000,
        description="Save checkpoint every N steps"
    )
    eval_steps: int = Field(
        100,
        ge=10,
        le=10000,
        description="Evaluate every N steps"
    )
    output_dir: str = Field(
        "./audiorag_adapters",
        description="Directory to save adapters"
    )
    lora_config: LoRAConfig = Field(
        default_factory=LoRAConfig,
        description="LoRA configuration"
    )


# ============================================================================
# INFERENCE CONFIGURATION
# ============================================================================

class InferenceConfig(BaseModel):
    """Configuration for model inference."""
    
    model_name: str = Field(
        "microsoft/phi-2",
        description="Model identifier"
    )
    adapter_path: Optional[str] = Field(
        None,
        description="Path to LoRA adapter (optional)"
    )
    quantization: QuantizationType = Field(
        QuantizationType.NONE,
        description="Quantization type"
    )
    device: DeviceType = Field(
        DeviceType.CPU,
        description="Device for inference"
    )
    max_tokens: int = Field(
        256,
        ge=1,
        le=4096,
        description="Max tokens to generate"
    )
    temperature: float = Field(
        0.7,
        ge=0,
        le=2,
        description="Sampling temperature"
    )
    top_p: float = Field(
        0.9,
        ge=0,
        le=1,
        description="Nucleus sampling parameter"
    )
    top_k: int = Field(
        50,
        ge=1,
        le=1000,
        description="Top-k sampling"
    )
    do_sample: bool = Field(
        True,
        description="Use sampling vs greedy decoding"
    )
    generation_timeout_seconds: float = Field(
        5.0,
        gt=0,
        le=60,
        description="Timeout for generation"
    )
    load_in_8bit: bool = Field(
        False,
        description="Load model in 8-bit mode (reduces memory)"
    )
    
    class Config:
        use_enum_values = True


# ============================================================================
# RETRY & RESILIENCE CONFIGURATION
# ============================================================================

class RetryConfig(BaseModel):
    """Configuration for retry strategy."""
    
    max_attempts: int = Field(
        3,
        ge=1,
        le=10,
        description="Maximum retry attempts"
    )
    initial_delay_seconds: float = Field(
        1.0,
        gt=0,
        le=30,
        description="Initial delay between retries"
    )
    max_delay_seconds: float = Field(
        30.0,
        gt=0,
        le=300,
        description="Maximum delay between retries"
    )
    backoff_factor: float = Field(
        2.0,
        gt=1,
        le=10,
        description="Exponential backoff multiplier"
    )
    retry_on_exceptions: List[str] = Field(
        ["TimeoutError", "ConnectionError"],
        description="Exception types to retry on"
    )


class CircuitBreakerConfig(BaseModel):
    """Configuration for circuit breaker pattern."""
    
    failure_threshold: int = Field(
        5,
        ge=1,
        le=50,
        description="Failures before opening circuit"
    )
    recovery_timeout_seconds: int = Field(
        60,
        ge=10,
        le=3600,
        description="Seconds before attempting recovery"
    )
    monitored_exceptions: List[str] = Field(
        ["ConnectionError", "TimeoutError"],
        description="Exception types that trigger circuit breaker"
    )


# ============================================================================
# MAIN RAG CONFIGURATION
# ============================================================================

class RAGConfig(BaseModel):
    """Master configuration for AudioRAG pipeline."""
    
    # Component configs
    chunk_config: ChunkConfig = Field(
        default_factory=ChunkConfig,
        description="Chunking configuration"
    )
    embedding_config: EmbeddingConfig = Field(
        default_factory=EmbeddingConfig,
        description="Embedding configuration"
    )
    vector_db_config: VectorDBConfig = Field(
        default_factory=VectorDBConfig,
        description="Vector DB configuration"
    )
    retrieval_config: RetrievalConfig = Field(
        default_factory=RetrievalConfig,
        description="Retrieval configuration"
    )
    fine_tuning_config: FineTuningConfig = Field(
        default_factory=FineTuningConfig,
        description="Fine-tuning configuration"
    )
    inference_config: InferenceConfig = Field(
        default_factory=InferenceConfig,
        description="Inference configuration"
    )
    
    # Resilience configs
    retry_config: RetryConfig = Field(
        default_factory=RetryConfig,
        description="Retry configuration"
    )
    circuit_breaker_config: CircuitBreakerConfig = Field(
        default_factory=CircuitBreakerConfig,
        description="Circuit breaker configuration"
    )
    
    # System configs
    log_level: str = Field(
        "INFO",
        description="Logging level: DEBUG, INFO, WARNING, ERROR, CRITICAL"
    )
    log_file: Optional[str] = Field(
        None,
        description="Log file path (optional)"
    )
    enable_monitoring: bool = Field(
        True,
        description="Enable system health monitoring"
    )
    memory_warning_threshold: float = Field(
        80.0,
        ge=50,
        le=95,
        description="Memory usage % threshold for warning"
    )
    
    @classmethod
    def from_yaml(cls, file_path: str) -> "RAGConfig":
        """Load configuration from YAML file."""
        with open(file_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)
    
    def to_yaml(self, file_path: str) -> None:
        """Save configuration to YAML file."""
        with open(file_path, 'w') as f:
            yaml.dump(self.dict(), f, default_flow_style=False)
    
    @classmethod
    def from_json(cls, file_path: str) -> "RAGConfig":
        """Load configuration from JSON file."""
        import json
        with open(file_path, 'r') as f:
            config_dict = json.load(f)
        return cls(**config_dict)
    
    def to_json(self, file_path: str) -> None:
        """Save configuration to JSON file."""
        import json
        with open(file_path, 'w') as f:
            json.dump(self.dict(), f, indent=2)
    
    class Config:
        use_enum_values = True
        arbitrary_types_allowed = True


# ============================================================================
# EXAMPLE CONFIGURATIONS
# ============================================================================

def get_default_config() -> RAGConfig:
    """Get default RAGConfig with sensible defaults."""
    return RAGConfig()


def get_lightweight_config() -> RAGConfig:
    """Get config optimized for 8GB RAM systems."""
    return RAGConfig(
        chunk_config=ChunkConfig(chunk_size=250, overlap=25),
        embedding_config=EmbeddingConfig(batch_size=16),
        inference_config=InferenceConfig(
            max_tokens=128,
            load_in_8bit=True
        ),
        fine_tuning_config=FineTuningConfig(batch_size=4)
    )


def get_production_config() -> RAGConfig:
    """Get config optimized for production deployment."""
    return RAGConfig(
        chunk_config=ChunkConfig(
            chunk_size=500,
            preserve_tables=True,
            audio_aware_chunking=True
        ),
        embedding_config=EmbeddingConfig(
            batch_size=64,
            cache_embeddings=True
        ),
        retrieval_config=RetrievalConfig(
            top_k=10,
            rerank_results=True
        ),
        inference_config=InferenceConfig(
            max_tokens=512,
            generation_timeout_seconds=10
        ),
        enable_monitoring=True,
        log_level="INFO"
    )
