"""Embedding generator that calls gpt2giga's /embeddings endpoint."""

import json
import time
from typing import List, Dict, Any, Optional
import httpx
import asyncio


class CodebaseEmbedder:
    """Generates embeddings for code chunks using gpt2giga API."""

    def __init__(
        self,
        gpt2giga_url: str,
        model: str = "EmbeddingsGigaR",
        batch_size: int = 100,
        timeout: float = 60.0,
    ):
        self.gpt2giga_url = gpt2giga_url.rstrip("/")
        self.model = model
        self.batch_size = batch_size
        self.timeout = timeout

    async def generate_embeddings(
        self, chunks: List, client: Optional[httpx.AsyncClient] = None
    ) -> List[Dict[str, Any]]:
        """Generate embeddings for code chunks."""
        if not chunks:
            return []

        # Prepare text inputs
        texts = [chunk.text for chunk in chunks]

        # Create HTTP client if not provided
        close_client = False
        if client is None:
            client = httpx.AsyncClient(timeout=self.timeout)
            close_client = True

        try:
            # Batch processing
            results = []
            for i in range(0, len(texts), self.batch_size):
                batch_texts = texts[i : i + self.batch_size]
                batch_chunks = chunks[i : i + self.batch_size]

                # Call embeddings endpoint
                response = await self._call_embeddings(
                    batch_texts, client, retries=3
                )

                if response:
                    embeddings = response.get("data", [])
                    for idx, embedding_data in enumerate(embeddings):
                        embedding = embedding_data.get("embedding", [])
                        # Handle base64 encoded embeddings
                        if isinstance(embedding, str):
                            import base64

                            try:
                                # Decode base64 if needed
                                embedding = base64.b64decode(embedding)
                                # Convert bytes to list if needed
                                if isinstance(embedding, bytes):
                                    import struct

                                    # Try to decode as float array
                                    embedding = list(
                                        struct.unpack(
                                            f"{len(embedding)//4}f", embedding
                                        )
                                    )
                            except Exception:
                                # If decoding fails, try to parse as JSON
                                try:
                                    embedding = json.loads(embedding)
                                except Exception:
                                    pass

                        results.append(
                            {
                                "chunk": batch_chunks[idx],
                                "embedding": embedding if isinstance(embedding, list) else [],
                            }
                        )
                else:
                    # Failed to get embeddings for this batch
                    for chunk in batch_chunks:
                        results.append({"chunk": chunk, "embedding": []})

            return results
        finally:
            if close_client:
                await client.aclose()

    async def _call_embeddings(
        self, texts: List[str], client: httpx.AsyncClient, retries: int = 3
    ) -> Optional[Dict[str, Any]]:
        """Call embeddings endpoint with retry logic."""
        url = f"{self.gpt2giga_url}/embeddings"
        if not url.endswith("/embeddings"):
            # Try /v1/embeddings if needed
            if "/v1" not in url:
                url = f"{self.gpt2giga_url}/v1/embeddings"

        payload = {"model": self.model, "input": texts}

        for attempt in range(retries):
            try:
                response = await client.post(url, json=payload)
                response.raise_for_status()
                return response.json()
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 429:
                    # Rate limited: exponential backoff
                    wait_time = 2 ** attempt
                    await asyncio.sleep(wait_time)
                    if attempt < retries - 1:
                        continue
                raise
            except Exception as e:
                if attempt < retries - 1:
                    wait_time = 2 ** attempt
                    await asyncio.sleep(wait_time)
                    continue
                raise

        return None

