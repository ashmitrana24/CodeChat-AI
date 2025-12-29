"""
Embedding Generation Module

Converts code chunks into vector representations
using sentence-transformers models.
"""
import numpy as np
from typing import List, Union
from sentence_transformers import SentenceTransformer

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import EMBEDDING_MODEL
from modules.chunking import CodeChunk


class EmbeddingGenerator:
    """
    Generates vector embeddings for code chunks.
    
    Uses sentence-transformers models optimized for
    semantic similarity matching.
    """
    
    def __init__(self, model_name: str = None):
        """
        Initialize the embedding generator.
        
        Args:
            model_name: HuggingFace model identifier
        """
        self.model_name = model_name or EMBEDDING_MODEL
        self._model = None
    
    @property
    def model(self) -> SentenceTransformer:
        """Lazy load the embedding model."""
        if self._model is None:
            print(f"Loading embedding model: {self.model_name}")
            self._model = SentenceTransformer(self.model_name)
        return self._model
    
    @property
    def embedding_dimension(self) -> int:
        """Get the dimension of embeddings produced by the model."""
        return self.model.get_sentence_embedding_dimension()
    
    def generate_embedding(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single piece of text.
        
        Args:
            text: Text to embed
            
        Returns:
            Numpy array of the embedding vector
        """
        return self.model.encode(text, convert_to_numpy=True)
    
    def generate_embeddings(
        self, 
        texts: List[str], 
        batch_size: int = 32,
        show_progress: bool = True
    ) -> np.ndarray:
        """
        Generate embeddings for multiple texts.
        
        Args:
            texts: List of texts to embed
            batch_size: Number of texts to process at once
            show_progress: Whether to show progress bar
            
        Returns:
            Numpy array of shape (n_texts, embedding_dim)
        """
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True
        )
        return embeddings
    
    def embed_chunks(
        self, 
        chunks: List[CodeChunk],
        batch_size: int = 32
    ) -> np.ndarray:
        """
        Generate embeddings for code chunks.
        
        Args:
            chunks: List of CodeChunk objects
            batch_size: Number of chunks to process at once
            
        Returns:
            Numpy array of embeddings
        """
        # Create enriched text with metadata context
        texts = []
        for chunk in chunks:
            # Add file context to help with retrieval
            enriched_text = f"File: {chunk.relative_path}\nLanguage: {chunk.language}\n\n{chunk.content}"
            texts.append(enriched_text)
        
        return self.generate_embeddings(texts, batch_size=batch_size)
    
    def embed_query(self, query: str) -> np.ndarray:
        """
        Generate embedding for a search query.
        
        Args:
            query: The user's question
            
        Returns:
            Numpy array of the query embedding
        """
        return self.generate_embedding(query)
