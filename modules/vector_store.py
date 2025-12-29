"""
Vector Store Module

Manages FAISS vector database for storing and
retrieving code chunk embeddings.
"""
import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass
import faiss

from modules.chunking import CodeChunk


@dataclass
class SearchResult:
    """Represents a search result with chunk and score."""
    chunk: CodeChunk
    score: float
    rank: int


class VectorStore:
    """
    FAISS-based vector store for code embeddings.
    
    Stores embeddings alongside chunk metadata for
    fast similarity-based retrieval.
    """
    
    def __init__(self, dimension: int = 384):
        """
        Initialize the vector store.
        
        Args:
            dimension: Dimension of embedding vectors
        """
        self.dimension = dimension
        self.index: Optional[faiss.IndexFlatIP] = None
        self.chunks: List[CodeChunk] = []
        self._is_initialized = False
    
    def _initialize_index(self):
        """Create a new FAISS index."""
        # Using Inner Product (cosine similarity) index
        self.index = faiss.IndexFlatIP(self.dimension)
        self._is_initialized = True
    
    def add_embeddings(
        self, 
        embeddings: np.ndarray, 
        chunks: List[CodeChunk]
    ) -> None:
        """
        Add embeddings and their associated chunks to the store.
        
        Args:
            embeddings: Numpy array of shape (n, dimension)
            chunks: List of CodeChunk objects (same length as embeddings)
            
        Raises:
            ValueError: If embeddings and chunks have different lengths
        """
        if len(embeddings) != len(chunks):
            raise ValueError(
                f"Embeddings count ({len(embeddings)}) doesn't match "
                f"chunks count ({len(chunks)})"
            )
        
        if not self._is_initialized:
            self.dimension = embeddings.shape[1]
            self._initialize_index()
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        
        # Add to FAISS index
        self.index.add(embeddings.astype(np.float32))
        
        # Store chunks for retrieval
        self.chunks.extend(chunks)
    
    def search(
        self, 
        query_embedding: np.ndarray, 
        top_k: int = 5
    ) -> List[SearchResult]:
        """
        Search for similar chunks using a query embedding.
        
        Args:
            query_embedding: The query vector
            top_k: Number of results to return
            
        Returns:
            List of SearchResult objects with chunks and scores
            
        Raises:
            ValueError: If the store is empty
        """
        if not self._is_initialized or self.index.ntotal == 0:
            raise ValueError("Vector store is empty. Load a repository first.")
        
        # Normalize query for cosine similarity
        query = query_embedding.reshape(1, -1).astype(np.float32)
        faiss.normalize_L2(query)
        
        # Limit k to available vectors
        k = min(top_k, self.index.ntotal)
        
        # Search
        scores, indices = self.index.search(query, k)
        
        # Build results
        results = []
        for rank, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx >= 0 and idx < len(self.chunks):
                results.append(SearchResult(
                    chunk=self.chunks[idx],
                    score=float(score),
                    rank=rank + 1
                ))
        
        return results
    
    def clear(self) -> None:
        """Clear all data from the vector store."""
        self.index = None
        self.chunks = []
        self._is_initialized = False
    
    @property
    def size(self) -> int:
        """Get the number of vectors in the store."""
        if not self._is_initialized:
            return 0
        return self.index.ntotal
    
    def get_stats(self) -> dict:
        """Get statistics about the vector store."""
        unique_files = set(c.relative_path for c in self.chunks)
        
        return {
            "total_vectors": self.size,
            "total_chunks": len(self.chunks),
            "unique_files": len(unique_files),
            "dimension": self.dimension
        }
