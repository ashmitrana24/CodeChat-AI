"""
Code Chunking Module

Splits large source files into smaller, meaningful chunks
while maintaining logical coherence and metadata.
"""
from typing import List
from dataclasses import dataclass

from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    Language
)

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import CHUNK_SIZE, CHUNK_OVERLAP
from modules.ingestion import SourceFile


# Map file extensions to LangChain Language enum
EXTENSION_TO_LANGUAGE = {
    ".py": Language.PYTHON,
    ".js": Language.JS,
    ".ts": Language.TS,
    ".java": Language.JAVA,
    ".cpp": Language.CPP,
    ".md": Language.MARKDOWN,
}


@dataclass
class CodeChunk:
    """Represents a chunk of code with metadata."""
    content: str
    file_path: str
    relative_path: str
    chunk_index: int
    total_chunks: int
    language: str


class CodeChunker:
    """
    Splits source files into semantically meaningful chunks.
    
    Uses language-aware splitting to preserve code structure
    like function and class boundaries where possible.
    """
    
    def __init__(self, chunk_size: int = None, chunk_overlap: int = None):
        """
        Initialize the chunker with size parameters.
        
        Args:
            chunk_size: Maximum size of each chunk in characters
            chunk_overlap: Number of overlapping characters between chunks
        """
        self.chunk_size = chunk_size or CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or CHUNK_OVERLAP
        self._splitters = {}
    
    def _get_splitter(self, extension: str) -> RecursiveCharacterTextSplitter:
        """
        Get or create a language-specific text splitter.
        
        Args:
            extension: File extension (e.g., '.py')
            
        Returns:
            Configured text splitter for the language
        """
        if extension not in self._splitters:
            language = EXTENSION_TO_LANGUAGE.get(extension)
            
            if language:
                self._splitters[extension] = RecursiveCharacterTextSplitter.from_language(
                    language=language,
                    chunk_size=self.chunk_size,
                    chunk_overlap=self.chunk_overlap
                )
            else:
                # Fallback for unsupported languages
                self._splitters[extension] = RecursiveCharacterTextSplitter(
                    chunk_size=self.chunk_size,
                    chunk_overlap=self.chunk_overlap,
                    separators=["\n\n", "\n", " ", ""]
                )
        
        return self._splitters[extension]
    
    def chunk_file(self, source_file: SourceFile) -> List[CodeChunk]:
        """
        Split a single source file into chunks.
        
        Args:
            source_file: The source file to chunk
            
        Returns:
            List of CodeChunk objects
        """
        splitter = self._get_splitter(source_file.extension)
        
        # Split the content
        text_chunks = splitter.split_text(source_file.content)
        
        # Create CodeChunk objects with metadata
        chunks = []
        for i, content in enumerate(text_chunks):
            chunks.append(CodeChunk(
                content=content,
                file_path=source_file.file_path,
                relative_path=source_file.relative_path,
                chunk_index=i,
                total_chunks=len(text_chunks),
                language=source_file.extension.lstrip('.')
            ))
        
        return chunks
    
    def chunk_files(self, source_files: List[SourceFile]) -> List[CodeChunk]:
        """
        Split multiple source files into chunks.
        
        Args:
            source_files: List of source files to chunk
            
        Returns:
            List of all CodeChunk objects from all files
        """
        all_chunks = []
        
        for source_file in source_files:
            chunks = self.chunk_file(source_file)
            all_chunks.extend(chunks)
        
        return all_chunks
    
    def get_stats(self, chunks: List[CodeChunk]) -> dict:
        """
        Get statistics about chunked content.
        
        Args:
            chunks: List of code chunks
            
        Returns:
            Dictionary with chunk statistics
        """
        if not chunks:
            return {"total_chunks": 0, "avg_chunk_size": 0, "by_language": {}}
        
        total_chars = sum(len(c.content) for c in chunks)
        by_language = {}
        
        for chunk in chunks:
            lang = chunk.language
            by_language[lang] = by_language.get(lang, 0) + 1
        
        return {
            "total_chunks": len(chunks),
            "avg_chunk_size": total_chars // len(chunks),
            "by_language": by_language
        }
