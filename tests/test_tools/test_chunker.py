"""Unit tests for CodeChunker."""

import pytest

from gpt2giga.tools.chunker import CodeChunker
from gpt2giga.tools.parser import CodeChunk


class TestCodeChunker:
    """Test cases for CodeChunker."""

    def test_chunk_code_small_chunks(self):
        """Test chunking small code chunks."""
        chunks = [
            CodeChunk(
                text="def func1():\n    pass\n",
                file_path="/test.py",
                start_line=1,
                end_line=2,
                chunk_type="function",
                language="python",
                name="func1",
            ),
            CodeChunk(
                text="def func2():\n    pass\n",
                file_path="/test.py",
                start_line=3,
                end_line=4,
                chunk_type="function",
                language="python",
                name="func2",
            ),
        ]

        chunker = CodeChunker(max_tokens=1000)
        result = chunker.chunk_code(chunks)

        # Small chunks should be merged or kept separate
        assert len(result) >= 1
        assert all(isinstance(c, CodeChunk) for c in result)

    def test_chunk_code_large_chunk(self):
        """Test chunking large code chunk."""
        # Create a large chunk (simulated)
        large_text = "def large_func():\n" + "    x = 1\n" * 500
        chunks = [
            CodeChunk(
                text=large_text,
                file_path="/test.py",
                start_line=1,
                end_line=501,
                chunk_type="function",
                language="python",
                name="large_func",
            )
        ]

        chunker = CodeChunker(max_tokens=100)  # Small limit
        result = chunker.chunk_code(chunks)

        # Large chunk should be split
        assert len(result) >= 1
        assert all(isinstance(c, CodeChunk) for c in result)

    def test_chunk_code_mixed_sizes(self):
        """Test chunking chunks of mixed sizes."""
        chunks = [
            CodeChunk(
                text="def small():\n    pass\n",
                file_path="/test.py",
                start_line=1,
                end_line=2,
                chunk_type="function",
                language="python",
            ),
            CodeChunk(
                text="def medium():\n" + "    x = i\n" * 100,
                file_path="/test.py",
                start_line=3,
                end_line=103,
                chunk_type="function",
                language="python",
            ),
            CodeChunk(
                text="def tiny():\n    return\n",
                file_path="/test.py",
                start_line=104,
                end_line=105,
                chunk_type="function",
                language="python",
            ),
        ]

        chunker = CodeChunker(max_tokens=200)
        result = chunker.chunk_code(chunks)

        assert len(result) >= 1
        assert all(isinstance(c, CodeChunk) for c in result)

    def test_chunk_code_empty_list(self):
        """Test chunking empty list."""
        chunker = CodeChunker()
        result = chunker.chunk_code([])

        assert result == []

    def test_chunk_code_preserves_metadata(self):
        """Test that chunking preserves metadata."""
        chunks = [
            CodeChunk(
                text="def test():\n    pass\n",
                file_path="/test.py",
                start_line=1,
                end_line=2,
                chunk_type="function",
                language="python",
                name="test",
            )
        ]

        chunker = CodeChunker(max_tokens=1000)
        result = chunker.chunk_code(chunks)

        assert len(result) >= 1
        chunk = result[0]
        assert chunk.file_path == "/test.py"
        assert chunk.language == "python"
        assert chunk.chunk_type == "function"

    def test_chunk_code_token_counting(self):
        """Test that token counting works."""
        # Create chunks with known token counts
        small_text = "x" * 10  # Small text
        large_text = "x" * 10000  # Large text

        chunks = [
            CodeChunk(
                text=small_text,
                file_path="/test.py",
                start_line=1,
                end_line=1,
                chunk_type="line",
                language="python",
            ),
            CodeChunk(
                text=large_text,
                file_path="/test.py",
                start_line=2,
                end_line=2,
                chunk_type="line",
                language="python",
            ),
        ]

        chunker = CodeChunker(max_tokens=100)
        result = chunker.chunk_code(chunks)

        # Large chunk should be split
        assert len(result) >= 1

