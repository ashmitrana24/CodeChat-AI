"""
RAG Answer Generation Module

Uses retrieved code context to generate accurate,
grounded answers using Google Gemini SDK.
"""

from typing import List
from dataclasses import dataclass
import google.generativeai as genai

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import GEMINI_API_KEY, GEMINI_MODEL
from modules.vector_store import SearchResult
from modules.question_processor import QuestionProcessor


@dataclass
class RAGResponse:
    """Represents a RAG-generated response."""
    answer: str
    source_files: List[str]
    chunks_used: int


SYSTEM_PROMPT = """
You are an expert code assistant analyzing a codebase. 
Answer questions based ONLY on the provided context.

Rules:
1. Only answer from the code context
2. If information is missing, say so clearly
3. Reference specific files when applicable
4. Be concise and precise
5. Use code snippets if helpful
"""


class RAGGenerator:
    """
    Generates answers using Retrieval-Augmented Generation (RAG)
    with Google Gemini SDK.
    """

    def __init__(
        self,
        question_processor: QuestionProcessor,
        api_key: str = None,
        model_name: str = None
    ):
        self.question_processor = question_processor
        self.api_key = api_key or GEMINI_API_KEY
        self.model_name = model_name or GEMINI_MODEL

        if not self.api_key:
            raise ValueError("GEMINI_API_KEY is missing in config or .env file")

        # Configure the Gemini API
        genai.configure(api_key=self.api_key)
        
        # Create the model instance
        self.model = genai.GenerativeModel(self.model_name)

    def _build_prompt(self, question: str, context: str) -> str:
        """Build a prompt combining system instructions and context."""
        return f"""
{SYSTEM_PROMPT}

### Code Context
{context}

### Question
{question}

### Answer
"""

    def generate(self, question: str) -> RAGResponse:
        """Generate an answer for a question using RAG."""
        results = self.question_processor.process(question)
        context = self.question_processor.format_context(results)
        source_files = self.question_processor.get_file_references(results)
        prompt = self._build_prompt(question, context)

        if not results:
            return RAGResponse(
                answer="No relevant context found to answer the question.",
                source_files=[],
                chunks_used=0
            )

        try:
            response = self.model.generate_content(prompt)
            answer = response.text.strip()
        except Exception as e:
            answer = f"Gemini error: {str(e)}"

        return RAGResponse(
            answer=answer,
            source_files=source_files,
            chunks_used=len(results)
        )

    def generate_with_context(
        self,
        question: str,
        results: List[SearchResult]
    ) -> RAGResponse:
        """Generate an answer using pre-retrieved search results."""
        context = self.question_processor.format_context(results)
        source_files = self.question_processor.get_file_references(results)
        prompt = self._build_prompt(question, context)

        if not results:
            return RAGResponse(
                answer="No relevant context found to answer the question.",
                source_files=[],
                chunks_used=0
            )

        try:
            response = self.model.generate_content(prompt)
            answer = response.text.strip()
        except Exception as e:
            answer = f"Gemini error: {str(e)}"

        return RAGResponse(
            answer=answer,
            source_files=source_files,
            chunks_used=len(results)
        )
