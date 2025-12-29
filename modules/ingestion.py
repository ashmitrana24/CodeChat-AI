"""
Codebase Ingestion Module

Handles reading source files from a project directory,
filtering by supported extensions and ignoring unnecessary directories.
"""
import os
from pathlib import Path
from typing import List, Tuple
from dataclasses import dataclass

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import SUPPORTED_EXTENSIONS, IGNORED_DIRECTORIES


@dataclass
class SourceFile:
    """Represents a source file with its content and metadata."""
    file_path: str
    content: str
    extension: str
    relative_path: str


class CodebaseIngester:
    """
    Ingests source files from a project directory.
    
    Filters files by supported extensions and excludes
    unnecessary directories like node_modules, .git, etc.
    """
    
    def __init__(
        self,
        supported_extensions: set = None,
        ignored_directories: set = None
    ):
        """
        Initialize the ingester with configuration.
        
        Args:
            supported_extensions: Set of file extensions to include
            ignored_directories: Set of directory names to skip
        """
        self.supported_extensions = supported_extensions or SUPPORTED_EXTENSIONS
        self.ignored_directories = ignored_directories or IGNORED_DIRECTORIES
    
    def _should_ignore_directory(self, dir_name: str) -> bool:
        """Check if a directory should be ignored."""
        return dir_name in self.ignored_directories
    
    def _is_supported_file(self, file_path: str) -> bool:
        """Check if a file has a supported extension."""
        return Path(file_path).suffix.lower() in self.supported_extensions
    
    def _read_file_content(self, file_path: str) -> str:
        """
        Safely read file content with encoding fallback.
        
        Args:
            file_path: Path to the file
            
        Returns:
            File content as string, or empty string on error
        """
        encodings = ['utf-8', 'utf-8-sig', 'latin-1', 'cp1252']
        
        for encoding in encodings:
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    return f.read()
            except (UnicodeDecodeError, UnicodeError):
                continue
            except Exception as e:
                print(f"Error reading {file_path}: {e}")
                return ""
        
        return ""
    
    def ingest(self, project_path: str) -> List[SourceFile]:
        """
        Ingest all supported source files from a project directory.
        
        Args:
            project_path: Path to the project root directory
            
        Returns:
            List of SourceFile objects with content and metadata
            
        Raises:
            ValueError: If the path doesn't exist or isn't a directory
        """
        project_path = Path(project_path).resolve()
        
        if not project_path.exists():
            raise ValueError(f"Path does not exist: {project_path}")
        
        if not project_path.is_dir():
            raise ValueError(f"Path is not a directory: {project_path}")
        
        source_files: List[SourceFile] = []
        
        for root, dirs, files in os.walk(project_path):
            # Filter out ignored directories (modifies dirs in-place)
            dirs[:] = [d for d in dirs if not self._should_ignore_directory(d)]
            
            for file_name in files:
                file_path = os.path.join(root, file_name)
                
                if not self._is_supported_file(file_path):
                    continue
                
                content = self._read_file_content(file_path)
                
                if not content.strip():
                    continue  # Skip empty files
                
                relative_path = os.path.relpath(file_path, project_path)
                
                source_files.append(SourceFile(
                    file_path=file_path,
                    content=content,
                    extension=Path(file_path).suffix.lower(),
                    relative_path=relative_path
                ))
        
        return source_files
    
    def get_stats(self, source_files: List[SourceFile]) -> dict:
        """
        Get statistics about ingested files.
        
        Args:
            source_files: List of ingested source files
            
        Returns:
            Dictionary with file counts by extension
        """
        stats = {"total_files": len(source_files), "by_extension": {}}
        
        for sf in source_files:
            ext = sf.extension
            stats["by_extension"][ext] = stats["by_extension"].get(ext, 0) + 1
        
        return stats
