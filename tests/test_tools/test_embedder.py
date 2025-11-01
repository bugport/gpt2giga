"""Unit tests for CodebaseEmbedder."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gpt2giga.tools.embedder import CodebaseEmbedder
from gpt2giga.tools.parser import CodeChunk


class TestCodebaseEmbedder:
    """Test cases for CodebaseEmbedder."""

    @pytest.mark.asyncio
    async def test_generate_embeddings_success(self):
        """Test successful embedding generation."""
        # Mock chunks
        chunks = [
            CodeChunk(
                text="def hello():\n    pass\n",
                file_path="/test.py",
                start_line=1,
                end_line=2,
                chunk_type="function",
                language="python",
            ),
            CodeChunk(
                text="def world():\n    pass\n",
                file_path="/test.py",
                start_line=3,
                end_line=4,
                chunk_type="function",
                language="python",
            ),
        ]

        # Mock response
        mock_response_data = {
            "data": [
                {"embedding": [0.1, 0.2, 0.3], "index": 0},
                {"embedding": [0.4, 0.5, 0.6], "index": 1},
            ],
            "model": "EmbeddingsGigaR",
        }

        embedder = CodebaseEmbedder(
            gpt2giga_url="http://localhost:8090", batch_size=10
        )

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_response = AsyncMock()
            mock_response.json.return_value = mock_response_data
            mock_response.raise_for_status = MagicMock()

            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client_class.return_value.__aenter__.return_value = mock_client
            mock_client_class.return_value.__aexit__.return_value = None

            results = await embedder.generate_embeddings(chunks)

            assert len(results) == 2
            assert results[0]["chunk"] == chunks[0]
            assert results[0]["embedding"] == [0.1, 0.2, 0.3]
            assert results[1]["chunk"] == chunks[1]
            assert results[1]["embedding"] == [0.4, 0.5, 0.6]

    @pytest.mark.asyncio
    async def test_generate_embeddings_batching(self):
        """Test that embeddings are batched correctly."""
        # Create many chunks
        chunks = [
            CodeChunk(
                text=f"def func{i}():\n    pass\n",
                file_path="/test.py",
                start_line=i * 2,
                end_line=i * 2 + 1,
                chunk_type="function",
                language="python",
            )
            for i in range(150)  # More than batch_size
        ]

        embedder = CodebaseEmbedder(
            gpt2giga_url="http://localhost:8090", batch_size=50
        )

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_response = AsyncMock()
            mock_response.json.return_value = {
                "data": [{"embedding": [0.1] * 100, "index": i} for i in range(50)],
                "model": "EmbeddingsGigaR",
            }
            mock_response.raise_for_status = MagicMock()

            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client_class.return_value.__aenter__.return_value = mock_client
            mock_client_class.return_value.__aexit__.return_value = None

            results = await embedder.generate_embeddings(chunks)

            # Should process in batches
            assert mock_client.post.call_count == 3  # 150 / 50 = 3 batches
            assert len(results) == 150

    @pytest.mark.asyncio
    async def test_generate_embeddings_rate_limit_retry(self):
        """Test retry on rate limit (429)."""
        chunks = [
            CodeChunk(
                text="def test():\n    pass\n",
                file_path="/test.py",
                start_line=1,
                end_line=2,
                chunk_type="function",
                language="python",
            )
        ]

        embedder = CodebaseEmbedder(
            gpt2giga_url="http://localhost:8090", batch_size=10
        )

        with patch("httpx.AsyncClient") as mock_client_class:
            from httpx import HTTPStatusError

            # First call: 429 error, second call: success
            mock_response_429 = AsyncMock()
            mock_response_429.raise_for_status.side_effect = HTTPStatusError(
                "429", request=MagicMock(), response=MagicMock(status_code=429)
            )

            mock_response_ok = AsyncMock()
            mock_response_ok.json.return_value = {
                "data": [{"embedding": [0.1, 0.2, 0.3], "index": 0}],
                "model": "EmbeddingsGigaR",
            }
            mock_response_ok.raise_for_status = MagicMock()

            mock_client = AsyncMock()
            mock_client.post = AsyncMock(
                side_effect=[mock_response_429, mock_response_ok]
            )
            mock_client_class.return_value.__aenter__.return_value = mock_client
            mock_client_class.return_value.__aexit__.return_value = None

            with patch("asyncio.sleep", new_callable=AsyncMock):
                results = await embedder.generate_embeddings(chunks)

                assert len(results) == 1
                assert results[0]["embedding"] == [0.1, 0.2, 0.3]

    @pytest.mark.asyncio
    async def test_generate_embeddings_empty_chunks(self):
        """Test embedding generation with empty chunks."""
        embedder = CodebaseEmbedder(gpt2giga_url="http://localhost:8090")
        results = await embedder.generate_embeddings([])

        assert results == []

    @pytest.mark.asyncio
    async def test_generate_embeddings_base64_handling(self):
        """Test handling base64 encoded embeddings."""
        chunks = [
            CodeChunk(
                text="def test():\n    pass\n",
                file_path="/test.py",
                start_line=1,
                end_line=2,
                chunk_type="function",
                language="python",
            )
        ]

        embedder = CodebaseEmbedder(gpt2giga_url="http://localhost:8090")

        with patch("httpx.AsyncClient") as mock_client_class:
            import base64

            # Create base64 encoded embedding
            embedding_bytes = b"\x00\x01\x02" * 100
            b64_embedding = base64.b64encode(embedding_bytes).decode("utf-8")

            mock_response = AsyncMock()
            mock_response.json.return_value = {
                "data": [{"embedding": b64_embedding, "index": 0}],
                "model": "EmbeddingsGigaR",
            }
            mock_response.raise_for_status = MagicMock()

            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client_class.return_value.__aenter__.return_value = mock_client
            mock_client_class.return_value.__aexit__.return_value = None

            results = await embedder.generate_embeddings(chunks)

            # Should handle base64 or decode it
            assert len(results) == 1
            # Result should have an embedding (decoded or original)
            assert "embedding" in results[0]

