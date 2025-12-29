"""
Configuration settings for the RAG Codebase Analysis System.
"""
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# API Keys
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

# Supported file extensions for code analysis
SUPPORTED_EXTENSIONS = {
    ".py",    # Python
    ".js",    # JavaScript
    ".ts",    # TypeScript
    ".java",  # Java
    ".cpp",   # C++
    ".md",    # Markdown documentation
}

# Directories to ignore during ingestion
IGNORED_DIRECTORIES = {
    "node_modules",
    ".git",
    "build",
    "dist",
    "__pycache__",
    ".venv",
    "venv",
    ".env",
    ".idea",
    ".vscode",
}

# Embedding Model Configuration
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

# Chunking Configuration
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))

# Retrieval Configuration
TOP_K_RESULTS = int(os.getenv("TOP_K_RESULTS", "5"))

# LLM Configuration
# Available models: gemini-pro-latest, gemini-flash-latest, gemini-2.5-flash, gemini-2.5-pro
# Use model name WITHOUT "models/" prefix when creating GenerativeModel
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-flash-latest")
