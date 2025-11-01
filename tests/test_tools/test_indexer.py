"""Unit tests for CodebaseIndexer."""

import asyncio
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gpt2giga.tools.indexer import CodebaseIndexer
from gpt2giga.tools.embedder import CodebaseEmbedder
from gpt2giga.tools.scanner import CodebaseScanner
from gpt2giga.tools.parser import CodeParser
from gpt2giga.tools.chunker import CodeChunker
from gpt2giga.tools.vector_db import SimpleVectorDB


class TestCodebaseIndexer:
    """Test cases for CodebaseIndexer."""

    @pytest.mark.asyncio
    async def test_index_codebase_basic(self, tmp_path):
        """Test basic codebase indexing workflow."""
        # Create test files
        (tmp_path / "test.py").write_text("def hello():\n    return 'world'\n")
        (tmp_path / "test2.py").write_text("def bye():\n    return 'goodbye'\n")

        # Mock components
        vector_db = SimpleVectorDB()
        embedder = CodebaseEmbedder(gpt2giga_url="http://localhost:8090")

        # Mock embedding generation
        with patch.object(
            embedder, "generate_embeddings", new_callable=AsyncMock
        ) as mock_embed:
            mock_embed.return_value = [
                {
                    "chunk": MagicMock(
                        text="def hello():\n    return 'world'\n",
                        file_path=str(tmp_path / "test.py"),
                        start_line=1,
                        end_line=2,
                        chunk_type="function",
                        language="python",
                        name="hello",
                    ),
                    "embedding": [0.1, 0.2, 0.3],
                },
                {
                    "chunk": MagicMock(
                        text="def bye():\n    return 'goodbye'\n",
                        file_path=str(tmp_path / "test2.py"),
                        start_line=1,
                        end_line=2,
                        chunk_type="function",
                        language="python",
                        name="bye",
                    ),
                    "embedding": [0.4, 0.5, 0.6],
                },
            ]

            scanner = CodebaseScanner(include_patterns=[".py"])
            parser = CodeParser()
            chunker = CodeChunker(max_tokens=1000)

            indexer = CodebaseIndexer(
                codebase_path=str(tmp_path),
                collection_name="test_collection",
                gpt2giga_url="http://localhost:8090",
                vector_db=vector_db,
                embedder=embedder,
                scanner=scanner,
                parser=parser,
                chunker=chunker,
                incremental=False,
            )

            stats = await indexer.index_codebase()

            assert stats["files_scanned"] >= 2
            assert stats["files_parsed"] >= 2
            assert stats["chunks_created"] >= 2
            assert stats["embeddings_generated"] == 2
            assert stats["documents_stored"] == 2
            assert len(stats["errors"]) == 0

    @pytest.mark.asyncio
    async def test_index_codebase_with_errors(self, tmp_path):
        """Test indexing with file errors."""
        # Create one valid file and one problematic file
        (tmp_path / "valid.py").write_text("def test(): pass\n")

        # Create a file that might cause parsing errors
        (tmp_path / "broken.py").write_text("invalid syntax here\n")

        vector_db = SimpleVectorDB()
        embedder = CodebaseEmbedder(gpt2giga_url="http://localhost:8090")

        with patch.object(
            embedder, "generate_embeddings", new_callable=AsyncMock
        ) as mock_embed:
            mock_embed.return_value = [
                {
                    "chunk": MagicMock(
                        text="def test(): pass\n",
                        file_path=str(tmp_path / "valid.py"),
                        start_line=1,
                        end_line=1,
                        chunk_type="function",
                        language="python",
                    ),
                    "embedding": [0.1, 0.2],
                }
            ]

            scanner = CodebaseScanner(include_patterns=[".py"])
            parser = CodeParser()
            chunker = CodeChunker()

            indexer = CodebaseIndexer(
                codebase_path=str(tmp_path),
                collection_name="test_collection",
                gpt2giga_url="http://localhost:8090",
                vector_db=vector_db,
                embedder=embedder,
                scanner=scanner,
                parser=parser,
                chunker=chunker,
                incremental=False,
            )

            stats = await indexer.index_codebase()

            # Should still process valid files
            assert stats["files_scanned"] >= 1
            # Errors might be recorded
            assert isinstance(stats["errors"], list)

    @pytest.mark.asyncio
    async def test_index_codebase_empty(self, tmp_path):
        """Test indexing empty codebase."""
        vector_db = SimpleVectorDB()
        embedder = CodebaseEmbedder(gpt2giga_url="http://localhost:8090")

        scanner = CodebaseScanner()
        parser = CodeParser()
        chunker = CodeChunker()

        indexer = CodebaseIndexer(
            codebase_path=str(tmp_path),
            collection_name="test_collection",
            gpt2giga_url="http://localhost:8090",
            vector_db=vector_db,
            embedder=embedder,
            scanner=scanner,
            parser=parser,
            chunker=chunker,
            incremental=False,
        )

        stats = await indexer.index_codebase()

        assert stats["files_scanned"] == 0
        assert stats["chunks_created"] == 0
        assert stats["documents_stored"] == 0

    @pytest.mark.asyncio
    async def test_index_codebase_stats(self, tmp_path):
        """Test that stats are correctly recorded."""
        (tmp_path / "file1.py").write_text("def a(): pass\n")
        (tmp_path / "file2.py").write_text("def b(): pass\n")

        vector_db = SimpleVectorDB()
        embedder = CodebaseEmbedder(gpt2giga_url="http://localhost:8090")

        with patch.object(
            embedder, "generate_embeddings", new_callable=AsyncMock
        ) as mock_embed:
            mock_embed.return_value = [
                {
                    "chunk": MagicMock(
                        text="def a(): pass\n",
                        file_path=str(tmp_path / "file1.py"),
                        start_line=1,
                        end_line=1,
                        chunk_type="function",
                        language="python",
                    ),
                    "embedding": [0.1],
                },
                {
                    "chunk": MagicMock(
                        text="def b(): pass\n",
                        file_path=str(tmp_path / "file2.py"),
                        start_line=1,
                        end_line=1,
                        chunk_type="function",
                        language="python",
                    ),
                    "embedding": [0.2],
                },
            ]

            scanner = CodebaseScanner(include_patterns=[".py"])
            parser = CodeParser()
            chunker = CodeChunker()

            indexer = CodebaseIndexer(
                codebase_path=str(tmp_path),
                collection_name="test_collection",
                gpt2giga_url="http://localhost:8090",
                vector_db=vector_db,
                embedder=embedder,
                scanner=scanner,
                parser=parser,
                chunker=chunker,
                incremental=False,
            )

            stats = await indexer.index_codebase()

            # Verify all stats fields exist
            assert "files_scanned" in stats
            assert "files_parsed" in stats
            assert "chunks_created" in stats
            assert "embeddings_generated" in stats
            assert "documents_stored" in stats
            assert "files_skipped" in stats
            assert "errors" in stats

            assert stats["files_scanned"] >= 2
            assert stats["documents_stored"] == 2

