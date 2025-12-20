"""
audiorag/core/exceptions.py

Production-grade custom exception hierarchy for AudioRAG.
Every exception includes: error_code, message, context, timestamp.

This allows Bose engineers to:
- Catch specific errors
- Log with context for debugging
- Build retry/fallback logic
- Monitor error patterns
"""

from datetime import datetime
from typing import Dict, Optional, Any
import json


class AudioRAGException(Exception):
    """
    Base exception for all AudioRAG errors.
    
    Attributes:
        message: Human-readable error description
        error_code: Standardized error code (e.g., "CHUNK_001")
        context: Dict with debug information (model, inputs, state)
        timestamp: When the error occurred
    
    Example:
        try:
            result = chunker.chunk_pdf(doc)
        except AudioRAGException as e:
            logger.error(f"[{e.error_code}] {e.message}", extra={"context": e.context})
    """
    
    def __init__(
        self,
        message: str,
        error_code: str,
        context: Optional[Dict[str, Any]] = None
    ):
        self.message = message
        self.error_code = error_code
        self.context = context or {}
        self.timestamp = datetime.utcnow().isoformat()
        
        # Store in context for logging
        self.context["error_code"] = error_code
        self.context["timestamp"] = self.timestamp
        
        super().__init__(self._format_message())
    
    def _format_message(self) -> str:
        """Format error message for logging."""
        return f"[{self.error_code}] {self.message}"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict for JSON logging."""
        return {
            "error_type": self.__class__.__name__,
            "error_code": self.error_code,
            "message": self.message,
            "timestamp": self.timestamp,
            "context": self.context
        }


# ============================================================================
# CHUNKING ERRORS
# ============================================================================

class ChunkingError(AudioRAGException):
    """Base exception for document chunking failures."""
    pass


class PDFParseError(ChunkingError):
    """Raised when PDF parsing fails (corrupted, unreadable, unsupported format)."""
    
    def __init__(self, message: str, file_path: str = None, context: Dict = None):
        ctx = context or {}
        if file_path:
            ctx["file_path"] = file_path
        super().__init__(message, "CHUNK_001", ctx)


class TablePreservationError(ChunkingError):
    """Raised when table extraction/preservation fails."""
    
    def __init__(self, message: str, table_index: int = None, context: Dict = None):
        ctx = context or {}
        if table_index is not None:
            ctx["table_index"] = table_index
        super().__init__(message, "CHUNK_002", ctx)


class CorruptedDocumentError(ChunkingError):
    """Raised when document appears corrupted or unreadable."""
    
    def __init__(self, message: str, file_path: str = None, context: Dict = None):
        ctx = context or {}
        if file_path:
            ctx["file_path"] = file_path
        super().__init__(message, "CHUNK_003", ctx)


class ChunkConfigError(ChunkingError):
    """Raised when chunk configuration is invalid."""
    
    def __init__(self, message: str, config: Dict = None, context: Dict = None):
        ctx = context or {}
        if config:
            ctx["config"] = config
        super().__init__(message, "CHUNK_004", ctx)


# ============================================================================
# EMBEDDING ERRORS
# ============================================================================

class EmbeddingError(AudioRAGException):
    """Base exception for embedding generation failures."""
    pass


class EmbedderInitError(EmbeddingError):
    """Raised when embedder fails to initialize (model missing, wrong device)."""
    
    def __init__(self, message: str, model_name: str = None, context: Dict = None):
        ctx = context or {}
        if model_name:
            ctx["model_name"] = model_name
        super().__init__(message, "EMB_001", ctx)


class EmbeddingGenerationError(EmbeddingError):
    """Raised when embedding generation fails (OOM, timeout, corrupted input)."""
    
    def __init__(
        self,
        message: str,
        input_length: int = None,
        retry_count: int = None,
        context: Dict = None
    ):
        ctx = context or {}
        if input_length is not None:
            ctx["input_length"] = input_length
        if retry_count is not None:
            ctx["retry_count"] = retry_count
        super().__init__(message, "EMB_002", ctx)


class EmbeddingDimensionError(EmbeddingError):
    """Raised when embedding dimension mismatches expected."""
    
    def __init__(
        self,
        message: str,
        expected_dim: int = None,
        actual_dim: int = None,
        context: Dict = None
    ):
        ctx = context or {}
        if expected_dim is not None:
            ctx["expected_dimension"] = expected_dim
        if actual_dim is not None:
            ctx["actual_dimension"] = actual_dim
        super().__init__(message, "EMB_003", ctx)


# ============================================================================
# RETRIEVAL ERRORS
# ============================================================================

class RetrievalError(AudioRAGException):
    """Base exception for retrieval failures."""
    pass


class VectorDBConnectionError(RetrievalError):
    """Raised when vector DB is unreachable or connection fails."""
    
    def __init__(self, message: str, db_type: str = None, context: Dict = None):
        ctx = context or {}
        if db_type:
            ctx["vector_db_type"] = db_type
        super().__init__(message, "RET_001", ctx)


class SearchError(RetrievalError):
    """Raised when search query execution fails."""
    
    def __init__(
        self,
        message: str,
        query_text: str = None,
        results_count: int = None,
        context: Dict = None
    ):
        ctx = context or {}
        if query_text:
            ctx["query_length"] = len(query_text)
        if results_count is not None:
            ctx["results_count"] = results_count
        super().__init__(message, "RET_002", ctx)


class NoResultsError(RetrievalError):
    """Raised when search returns no results (may not be error, but flag it)."""
    
    def __init__(self, message: str, threshold: float = None, context: Dict = None):
        ctx = context or {}
        if threshold is not None:
            ctx["similarity_threshold"] = threshold
        super().__init__(message, "RET_003", ctx)


class VectorDBIndexError(RetrievalError):
    """Raised when vector DB index is corrupted or missing."""
    
    def __init__(self, message: str, index_name: str = None, context: Dict = None):
        ctx = context or {}
        if index_name:
            ctx["index_name"] = index_name
        super().__init__(message, "RET_004", ctx)


# ============================================================================
# ADAPTER & FINE-TUNING ERRORS
# ============================================================================

class AdapterError(AudioRAGException):
    """Base exception for LoRA adapter failures."""
    pass


class AdapterLoadError(AdapterError):
    """Raised when adapter loading fails (file missing, incompatible)."""
    
    def __init__(self, message: str, adapter_path: str = None, context: Dict = None):
        ctx = context or {}
        if adapter_path:
            ctx["adapter_path"] = adapter_path
        super().__init__(message, "ADP_001", ctx)


class AdapterTrainingError(AdapterError):
    """Raised when LoRA training fails."""
    
    def __init__(
        self,
        message: str,
        epoch: int = None,
        step: int = None,
        context: Dict = None
    ):
        ctx = context or {}
        if epoch is not None:
            ctx["epoch"] = epoch
        if step is not None:
            ctx["step"] = step
        super().__init__(message, "ADP_002", ctx)


class AdapterMergeError(AdapterError):
    """Raised when adapter cannot be merged with base model."""
    
    def __init__(self, message: str, adapter_name: str = None, context: Dict = None):
        ctx = context or {}
        if adapter_name:
            ctx["adapter_name"] = adapter_name
        super().__init__(message, "ADP_003", ctx)


class LoRAConfigError(AdapterError):
    """Raised when LoRA configuration is invalid."""
    
    def __init__(self, message: str, config: Dict = None, context: Dict = None):
        ctx = context or {}
        if config:
            ctx["config"] = config
        super().__init__(message, "ADP_004", ctx)


# ============================================================================
# INFERENCE ERRORS
# ============================================================================

class InferenceError(AudioRAGException):
    """Base exception for model inference failures."""
    pass


class ModelLoadError(InferenceError):
    """Raised when model fails to load (missing, incompatible, corrupted)."""
    
    def __init__(self, message: str, model_name: str = None, context: Dict = None):
        ctx = context or {}
        if model_name:
            ctx["model_name"] = model_name
        super().__init__(message, "INF_001", ctx)


class GenerationTimeoutError(InferenceError):
    """Raised when text generation exceeds time limit."""
    
    def __init__(
        self,
        message: str,
        timeout_seconds: float = None,
        tokens_generated: int = None,
        context: Dict = None
    ):
        ctx = context or {}
        if timeout_seconds is not None:
            ctx["timeout_seconds"] = timeout_seconds
        if tokens_generated is not None:
            ctx["tokens_generated"] = tokens_generated
        super().__init__(message, "INF_002", ctx)


class OutOfMemoryError(InferenceError):
    """Raised when system runs out of memory during inference."""
    
    def __init__(
        self,
        message: str,
        memory_required_gb: float = None,
        memory_available_gb: float = None,
        context: Dict = None
    ):
        ctx = context or {}
        if memory_required_gb is not None:
            ctx["memory_required_gb"] = memory_required_gb
        if memory_available_gb is not None:
            ctx["memory_available_gb"] = memory_available_gb
        super().__init__(message, "INF_003", ctx)


class QuantizationError(InferenceError):
    """Raised when model quantization fails."""
    
    def __init__(
        self,
        message: str,
        quantization_type: str = None,
        context: Dict = None
    ):
        ctx = context or {}
        if quantization_type:
            ctx["quantization_type"] = quantization_type
        super().__init__(message, "INF_004", ctx)


# ============================================================================
# CONFIGURATION ERRORS
# ============================================================================

class ConfigurationError(AudioRAGException):
    """Base exception for configuration issues."""
    pass


class InvalidConfigError(ConfigurationError):
    """Raised when configuration is invalid (wrong type, out of range)."""
    
    def __init__(
        self,
        message: str,
        param_name: str = None,
        expected_type: str = None,
        context: Dict = None
    ):
        ctx = context or {}
        if param_name:
            ctx["param_name"] = param_name
        if expected_type:
            ctx["expected_type"] = expected_type
        super().__init__(message, "CFG_001", ctx)


class MissingParameterError(ConfigurationError):
    """Raised when required parameter is missing."""
    
    def __init__(self, message: str, param_name: str = None, context: Dict = None):
        ctx = context or {}
        if param_name:
            ctx["param_name"] = param_name
        super().__init__(message, "CFG_002", ctx)


class DependencyError(ConfigurationError):
    """Raised when required dependency is missing or incompatible."""
    
    def __init__(
        self,
        message: str,
        dependency_name: str = None,
        required_version: str = None,
        context: Dict = None
    ):
        ctx = context or {}
        if dependency_name:
            ctx["dependency_name"] = dependency_name
        if required_version:
            ctx["required_version"] = required_version
        super().__init__(message, "CFG_003", ctx)


class EnvironmentError(ConfigurationError):
    """Raised when environment setup is incorrect (GPU, CUDA, paths)."""
    
    def __init__(self, message: str, issue_type: str = None, context: Dict = None):
        ctx = context or {}
        if issue_type:
            ctx["issue_type"] = issue_type
        super().__init__(message, "CFG_004", ctx)


# ============================================================================
# VALIDATION ERRORS
# ============================================================================

class ValidationError(AudioRAGException):
    """Base exception for input validation failures."""
    pass


class InputValidationError(ValidationError):
    """Raised when input data fails validation."""
    
    def __init__(
        self,
        message: str,
        input_field: str = None,
        expected_format: str = None,
        context: Dict = None
    ):
        ctx = context or {}
        if input_field:
            ctx["input_field"] = input_field
        if expected_format:
            ctx["expected_format"] = expected_format
        super().__init__(message, "VAL_001", ctx)


class OutputValidationError(ValidationError):
    """Raised when output fails quality checks."""
    
    def __init__(
        self,
        message: str,
        issue_type: str = None,
        context: Dict = None
    ):
        ctx = context or {}
        if issue_type:
            ctx["issue_type"] = issue_type
        super().__init__(message, "VAL_002", ctx)


# ============================================================================
# CIRCUIT BREAKER & RESOURCE ERRORS
# ============================================================================

class CircuitBreakerOpenError(AudioRAGException):
    """Raised when circuit breaker is open (too many failures)."""
    
    def __init__(
        self,
        message: str,
        service_name: str = None,
        recovery_time_sec: int = None,
        context: Dict = None
    ):
        ctx = context or {}
        if service_name:
            ctx["service_name"] = service_name
        if recovery_time_sec is not None:
            ctx["recovery_time_seconds"] = recovery_time_sec
        super().__init__(message, "SYS_001", ctx)


class ResourceExhaustedError(AudioRAGException):
    """Raised when system resource (memory, disk, connections) is exhausted."""
    
    def __init__(
        self,
        message: str,
        resource_type: str = None,
        context: Dict = None
    ):
        ctx = context or {}
        if resource_type:
            ctx["resource_type"] = resource_type
        super().__init__(message, "SYS_002", ctx)


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def exception_to_json(exc: AudioRAGException) -> str:
    """Convert exception to JSON string for structured logging."""
    return json.dumps(exc.to_dict(), indent=2)


def get_error_code_description(error_code: str) -> str:
    """Return human-readable description of error code."""
    descriptions = {
        # Chunking
        "CHUNK_001": "PDF parsing failed - file may be corrupted or unsupported format",
        "CHUNK_002": "Table extraction failed - unable to preserve table structure",
        "CHUNK_003": "Document is corrupted or unreadable",
        "CHUNK_004": "Invalid chunk configuration - check chunk_size and overlap",
        # Embedding
        "EMB_001": "Embedder initialization failed - check model name and device",
        "EMB_002": "Embedding generation failed - check input and retry",
        "EMB_003": "Embedding dimension mismatch - incompatible model",
        # Retrieval
        "RET_001": "Vector DB connection failed - check DB status",
        "RET_002": "Search query failed - check query and retry",
        "RET_003": "No results found - query may be too specific",
        "RET_004": "Vector DB index corrupted - rebuild index",
        # Adapter
        "ADP_001": "Adapter load failed - check adapter path and compatibility",
        "ADP_002": "Adapter training failed - check dataset and config",
        "ADP_003": "Adapter merge failed - incompatible with base model",
        "ADP_004": "Invalid LoRA config - check rank, alpha, target modules",
        # Inference
        "INF_001": "Model load failed - check model path and compatibility",
        "INF_002": "Generation timeout - increase timeout or reduce max_tokens",
        "INF_003": "Out of memory - reduce batch_size or model size",
        "INF_004": "Quantization failed - check format and model compatibility",
        # Config
        "CFG_001": "Invalid configuration value - check parameter type and range",
        "CFG_002": "Missing required parameter - check config completeness",
        "CFG_003": "Dependency missing or incompatible - install or upgrade",
        "CFG_004": "Environment setup incorrect - check GPU, CUDA, paths",
        # Validation
        "VAL_001": "Input validation failed - check data format",
        "VAL_002": "Output validation failed - retry or check model",
        # System
        "SYS_001": "Circuit breaker open - service temporarily unavailable",
        "SYS_002": "Resource exhausted - free up memory/disk or restart",
    }
    return descriptions.get(error_code, "Unknown error")
