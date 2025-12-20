# System Architecture

User Query
    ↓
[Query Validator] → Custom exceptions for invalid input
    ↓
[Audio-Aware Chunker] → Preserve tables, specs, audio context
    ↓
[Embedder] → Generate embeddings with retry logic
    ↓
[Vector DB Retriever] → Search with circuit breaker pattern
    ↓
[LoRA Adapter Loader] → Load fine-tuned weights
    ↓
[LLM Inference] → Generate answer with monitoring
    ↓
[Response Validator] → Ensure output quality
    ↓
Answer to User





# Project structure

```
audiorag/
├── __init__.py                          # Package exports
├── version.py                           # Version management
│
├── core/                                # Core components
│   ├── __init__.py
│   ├── exceptions.py                   # 🔴 Custom exceptions (ERROR HANDLING)
│   ├── logger.py                       # 📋 Structured logging
│   ├── config.py                       # ⚙️ Pydantic configs
│   ├── chunker.py                      # Split PDFs smartly
│   ├── embedder.py                     # Generate embeddings
│   ├── retriever.py                    # Vector search
│   └── health_monitor.py               # 🏥 System health checks
│
├── adapters/                            # LoRA fine-tuning
│   ├── __init__.py
│   ├── lora_config.py                  # LoRA configurations
│   ├── trainer.py                      # Training pipeline
│   ├── inference.py                    # On-device inference
│   └── quantization.py                 # Model quantization (memory efficient)
│
├── pipeline/                            # End-to-end RAG
│   ├── __init__.py
│   ├── rag_pipeline.py                 # Main RAG flow
│   └── retry_strategy.py               # 🔄 Retry + circuit breaker
│
├── utils/                               # Utilities
│   ├── __init__.py
│   ├── validators.py                   # Input validation
│   ├── parsers.py                      # File parsing helpers
│   └── metrics.py                      # Performance tracking
│
├── tests/                               # Comprehensive testing
│   ├── unit/
│   │   ├── test_chunker.py
│   │   ├── test_embedder.py
│   │   ├── test_retriever.py
│   │   ├── test_trainer.py
│   │   └── test_exceptions.py
│   ├── integration/
│   │   ├── test_rag_pipeline.py
│   │   └── test_bose_manual_rag.py
│   ├── performance/
│   │   ├── test_latency.py
│   │   └── test_memory.py
│   └── fixtures/
│       ├── sample_bose_docs.pdf
│       └── test_queries.json
│
├── examples/                            # Real-world examples
│   ├── basic_rag_example.py
│   ├── bose_ex1280_rag.py
│   ├── dsp_expert_adapter.py
│   └── on_device_inference.py
│
├── docs/                                # Documentation
│   ├── README.md
│   ├── API.md
│   ├── CONTRIBUTING.md
│   ├── ERROR_HANDLING.md
│   └── DEPLOYMENT.md
│
├── setup.py                             # PyPI packaging
├── requirements.txt                     # Dependencies
├── .github/
│   └── workflows/
│       └── tests.yml                    # CI/CD pipeline
│
└── ARCHITECTURE.md                      # This file
```


# Custom Exception Hierarchy

```AudioRAGException (Base)
├── ChunkingError
│   ├── PDFParseError
│   ├── TablePreservationError
│   └── CorruptedDocumentError
├── EmbeddingError
│   ├── EmbedderInitError
│   ├── EmbeddingGenerationError
│   └── EmbeddingDimensionError
├── RetrievalError
│   ├── VectorDBConnectionError
│   ├── SearchError
│   └── NoResultsError
├── AdapterError
│   ├── AdapterLoadError
│   ├── AdapterTrainingError
│   └── AdapterMergeError
├── InferenceError
│   ├── ModelLoadError
│   ├── GenerationTimeoutError
│   └── MemoryError
└── ConfigurationError
    ├── InvalidConfigError
    ├── MissingParameterError
    └── DependencyError
```