"""
Question Processing Module

Handles user questions by converting them to embeddings
and performing similarity search against stored vectors.
"""
from typing import List

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TOP_K_RESULTS
from modules.embeddings import EmbeddingGenerator
from modules.vector_store import VectorStore, SearchResult


class QuestionProcessor:
    """
    Processes user questions and retrieves relevant code chunks.
    
    Converts natural language questions to embeddings and
    performs similarity search against the vector store.
    """
    
    def __init__(
        self, 
        embedding_generator: EmbeddingGenerator,
        vector_store: VectorStore,
        top_k: int = None
    ):
        """
        Initialize the question processor.
        
        Args:
            embedding_generator: For converting questions to vectors
            vector_store: For similarity search
            top_k: Number of chunks to retrieve
        """
        self.embedding_generator = embedding_generator
        self.vector_store = vector_store
        self.top_k = top_k or TOP_K_RESULTS
    
    def process(self, question: str) -> List[SearchResult]:
        """
        Process a question and retrieve relevant code chunks.
        
        Args:
            question: The user's natural language question
            
        Returns:
            List of SearchResult objects with relevant chunks
        """
        # Generate embedding for the question
        query_embedding = self.embedding_generator.embed_query(question)
        
        # Search for similar chunks
        results = self.vector_store.search(query_embedding, self.top_k)
        
        return results
    
    def format_context(self, results: List[SearchResult]) -> str:
        """
        Format search results as context for the LLM.
        
        Args:
            results: List of search results
            
        Returns:
            Formatted string with code context
        """
        if not results:
            return "No relevant code found."
        
        context_parts = []
        
        for result in results:
            chunk = result.chunk
            context_parts.append(
                f"--- File: {chunk.relative_path} (Relevance: {result.score:.3f}) ---\n"
                f"```{chunk.language}\n{chunk.content}\n```"
            )
        
        return "\n\n".join(context_parts)
    
    def get_file_references(self, results: List[SearchResult]) -> List[str]:
        """
        Extract unique file references from results.
        
        Args:
            results: List of search results
            
        Returns:
            List of unique relative file paths
        """
        seen = set()
        files = []
        
        for result in results:
            path = result.chunk.relative_path
            if path not in seen:
                seen.add(path)
                files.append(path)
        
        return files
