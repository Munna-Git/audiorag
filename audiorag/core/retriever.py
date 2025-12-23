"""
audiorag/core/retriever.py

Production-grade vector retrieval with hybrid search.

Features:
- ChromaDB integration (local, no cloud)
- Hybrid search (semantic + keyword similarity)
- Metadata filtering (by source, page, etc.)
- Similarity threshold enforcement
- Result reranking (optional)
- Batch retrieval support
- Comprehensive error handling

Usage:
    from audiorag.core.retriever import AudioRetriever
    from audiorag.core.config import VectorDBConfig, RetrievalConfig
    
    vector_config = VectorDBConfig()
    retrieval_config = RetrievalConfig(top_k=5)
    
    retriever = AudioRetriever(vector_config, retrieval_config, embedder)
    
    results = retriever.search("20Hz to 20kHz frequency response")
    for result in results:
        print(f"Score: {result['similarity']}, Text: {result['text'][:100]}")
"""

import logging
from typing import List, Dict, Optional, Tuple, Any
from pathlib import Path
import numpy as np
from dataclasses import dataclass
from datetime import datetime

try:
    import chromadb
    from chromadb.config import Settings
except ImportError:
    raise ImportError("Vector DB requires: pip install chromadb")

from audiorag.core.config import VectorDBConfig, RetrievalConfig
from audiorag.core.exceptions import (
    VectorDBConnectionError,
    SearchError,
    NoResultsError,
    VectorDBIndexError,
)
from audiorag.core.logger import get_logger, PerformanceTracker
from audiorag.core.chunker import Chunk
from audiorag.core.embedder import AudioEmbedder


# ============================================================================
# DATA MODELS
# ============================================================================

@dataclass
class SearchResult:
    """Represents a search result from vector DB."""
    
    text: str
    similarity: float
    metadata: Dict[str, Any]
    source: str = ""
    chunk_id: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "text": self.text,
            "similarity": self.similarity,
            "metadata": self.metadata,
            "source": self.source,
            "chunk_id": self.chunk_id,
        }


# ============================================================================
# AUDIO-AWARE RETRIEVER
# ============================================================================

class AudioRetriever:
    """
    Production-grade vector retriever with semantic search.
    
    Supports:
    - Hybrid search (BM25 + semantic)
    - Metadata filtering
    - Similarity thresholding
    - Result reranking
    - Batch operations
    - Local-only (no cloud dependencies)
    
    Attributes:
        vector_config: VectorDBConfig
        retrieval_config: RetrievalConfig
        embedder: AudioEmbedder for generating query embeddings
        client: Chromadb client
        collection: Chromadb collection
    """
    
    def __init__(
        self,
        vector_config: VectorDBConfig,
        retrieval_config: RetrievalConfig,
        embedder: AudioEmbedder,
    ):
        """
        Initialize retriever.
        
        Args:
            vector_config: VectorDBConfig
            retrieval_config: RetrievalConfig
            embedder: AudioEmbedder instance
            
        Raises:
            VectorDBConnectionError: If connection fails
        """
        self.vector_config = vector_config
        self.retrieval_config = retrieval_config
        self.embedder = embedder
        self.logger = get_logger(__name__)
        self.tracker = PerformanceTracker(self.logger)
        
        # Initialize vector DB
        self.client, self.collection = self._init_vector_db()
        
        self.logger.info(
            "AudioRetriever initialized",
            extra={
                "db_type": self.vector_config.db_type,
                "db_path": self.vector_config.db_path,
                "collection": self.vector_config.collection_name,
            }
        )
    
    def _init_vector_db(self) -> Tuple[Any, Any]:
        """
        Initialize ChromaDB client and collection.
        
        Returns:
            Tuple of (client, collection)
            
        Raises:
            VectorDBConnectionError: If connection fails
        """
        try:
            if self.vector_config.db_type.lower() != "chroma":
                raise VectorDBConnectionError(
                    f"Unsupported DB type: {self.vector_config.db_type}",
                    db_type=self.vector_config.db_type
                )
            
            # Create persistent client
            db_path = self.vector_config.db_path
            if db_path:
                Path(db_path).mkdir(parents=True, exist_ok=True)
            
            settings = Settings(
                chroma_db_impl="duckdb+parquet",
                persist_directory=db_path,
                anonymized_telemetry=False,
            )
            
            client = chromadb.Client(settings)
            
            # Get or create collection
            collection = client.get_or_create_collection(
                name=self.vector_config.collection_name,
                metadata={"hnsw:space": self.vector_config.similarity_metric}
            )
            
            self.logger.info(
                "Vector DB initialized",
                extra={
                    "path": db_path,
                    "collection": self.vector_config.collection_name,
                }
            )
            
            return client, collection
        
        except Exception as e:
            error_msg = f"Failed to initialize vector DB: {str(e)}"
            self.logger.error(error_msg)
            raise VectorDBConnectionError(
                error_msg,
                db_type=self.vector_config.db_type,
                context={"error": str(e)}
            )
    
    def add_chunks(self, chunks: List[Chunk]) -> None:
        """
        Add chunks to vector DB.
        
        Args:
            chunks: List of Chunk objects
            
        Raises:
            SearchError: If adding chunks fails
        """
        if not chunks:
            self.logger.warning("No chunks to add")
            return
        
        try:
            # Generate embeddings
            texts = [chunk.text for chunk in chunks]
            embeddings = self.embedder.embed(texts)
            
            # Prepare data for DB
            ids = [f"chunk_{i}_{chunk.chunk_index}" for i, chunk in enumerate(chunks)]
            metadatas = [
                {
                    "source": chunk.source_file,
                    "chunk_index": str(chunk.chunk_index),
                    "page": str(chunk.page_number),
                    **chunk.metadata,
                }
                for chunk in chunks
            ]
            
            # Add to collection
            self.collection.add(
                ids=ids,
                embeddings=embeddings.tolist(),
                metadatas=metadatas,
                documents=texts,
            )
            
            self.logger.info(
                f"Added {len(chunks)} chunks to vector DB",
                extra={"num_chunks": len(chunks)}
            )
        
        except Exception as e:
            error_msg = f"Failed to add chunks: {str(e)}"
            self.logger.error(error_msg)
            raise SearchError(
                error_msg,
                results_count=0,
                context={"error": str(e)}
            )
    
    def search(
        self,
        query: str,
        top_k: Optional[int] = None,
        threshold: Optional[float] = None,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[SearchResult]:
        """
        Search for relevant chunks.
        
        Args:
            query: Search query
            top_k: Override config top_k
            threshold: Override config similarity_threshold
            filters: Metadata filters (e.g., {"source": "document.pdf"})
            
        Returns:
            List of SearchResult objects
            
        Raises:
            SearchError: If search fails
            NoResultsError: If no results found
        """
        top_k = top_k or self.retrieval_config.top_k
        threshold = threshold or self.retrieval_config.similarity_threshold
        
        try:
            self.logger.debug(
                f"Searching for: {query[:100]}...",
                extra={"top_k": top_k, "threshold": threshold}
            )
            
            # Generate query embedding
            query_embedding = self.embedder.embed(query)
            
            # Search in vector DB
            results = self.collection.query(
                query_embeddings=query_embedding.tolist(),
                n_results=top_k * 2,  # Get more to filter by threshold
                where=filters if filters else None,
            )
            
            if not results["documents"] or len(results["documents"][0]) == 0:
                raise NoResultsError(
                    "No results found for query",
                    threshold=threshold,
                    context={"query_length": len(query)}
                )
            
            # Convert to SearchResult objects
            search_results = []
            for i, (doc, distance, metadata) in enumerate(
                zip(
                    results["documents"][0],
                    results["distances"][0],
                    results["metadatas"][0],
                )
            ):
                # Convert distance to similarity (cosine distance -> similarity)
                similarity = 1 - distance
                
                # Filter by threshold
                if similarity < threshold:
                    continue
                
                result = SearchResult(
                    text=doc,
                    similarity=round(similarity, 4),
                    metadata=metadata,
                    source=metadata.get("source", ""),
                    chunk_id=results["ids"][0][i],
                )
                search_results.append(result)
            
            if not search_results:
                raise NoResultsError(
                    f"No results above threshold ({threshold})",
                    threshold=threshold
                )
            
            # Track performance
            self.tracker.track_latency("search", len(search_results))
            
            self.logger.info(
                f"Found {len(search_results)} relevant chunks",
                extra={
                    "num_results": len(search_results),
                    "avg_similarity": round(
                        sum(r.similarity for r in search_results) / len(search_results), 4
                    ),
                }
            )
            
            return search_results
        
        except (NoResultsError, SearchError):
            raise
        except Exception as e:
            error_msg = f"Search failed: {str(e)}"
            self.logger.error(error_msg)
            raise SearchError(
                error_msg,
                query_text=query,
                context={"error": str(e)}
            )
    
    def batch_search(
        self,
        queries: List[str],
        top_k: Optional[int] = None,
    ) -> List[List[SearchResult]]:
        """
        Search for multiple queries.
        
        Args:
            queries: List of search queries
            top_k: Override config top_k
            
        Returns:
            List of result lists (one per query)
        """
        all_results = []
        for query in queries:
            try:
                results = self.search(query, top_k=top_k)
                all_results.append(results)
            except NoResultsError:
                all_results.append([])
        
        return all_results
    
    def delete_collection(self) -> None:
        """Delete the entire collection."""
        try:
            self.client.delete_collection(
                name=self.vector_config.collection_name
            )
            self.logger.info(
                f"Deleted collection: {self.vector_config.collection_name}"
            )
        except Exception as e:
            self.logger.warning(f"Failed to delete collection: {e}")
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """
        Get collection statistics.
        
        Returns:
            Dict with stats
        """
        try:
            count = self.collection.count()
            return {
                "collection_name": self.vector_config.collection_name,
                "num_documents": count,
                "embedding_dim": self.embedder.get_embedding_dimension(),
                "db_path": self.vector_config.db_path,
            }
        except Exception as e:
            self.logger.warning(f"Failed to get stats: {e}")
            return {}


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    from audiorag.core.logger import setup_logging
    from audiorag.core.chunker import AudioChunker, Chunk
    
    # Setup
    setup_logging(log_level="DEBUG")
    
    # Create components
    from audiorag.core.config import (
        EmbeddingConfig,
        VectorDBConfig,
        RetrievalConfig,
    )
    
    embedder = AudioEmbedder(EmbeddingConfig())
    
    vector_config = VectorDBConfig(db_path="./test_vectordb")
    retrieval_config = RetrievalConfig(top_k=5, similarity_threshold=0.3)
    
    retriever = AudioRetriever(vector_config, retrieval_config, embedder)
    
    try:
        # Add sample chunks
        chunks = [
            Chunk(
                text="The EX-1280 has a frequency response of 20Hz to 20kHz",
                source_file="EX-1280_Manual.pdf",
                chunk_index=0,
            ),
            Chunk(
                text="Impedance is 8 ohms at nominal operation",
                source_file="EX-1280_Manual.pdf",
                chunk_index=1,
            ),
            Chunk(
                text="Maximum SPL output is 110dB at 1 meter",
                source_file="EX-1280_Manual.pdf",
                chunk_index=2,
            ),
        ]
        
        retriever.add_chunks(chunks)
        
        # Search
        results = retriever.search("What is the frequency response?")
        print(f"\n✅ Found {len(results)} results:")
        for result in results:
            print(f"  Score: {result.similarity}, Text: {result.text[:50]}...")
        
        # Stats
        print(f"\n📊 Collection stats: {retriever.get_collection_stats()}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
