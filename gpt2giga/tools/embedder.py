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
        # Normalize URL: ensure it has a scheme
        gpt2giga_url = gpt2giga_url.strip().rstrip("/")
        if not gpt2giga_url.startswith(("http://", "https://")):
            # Default to http:// if no scheme provided
            gpt2giga_url = f"http://{gpt2giga_url}"
        
        self.gpt2giga_url = gpt2giga_url
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
            # Increase connect timeout for initial connection attempts
            timeout_config = httpx.Timeout(
                connect=10.0,  # Allow more time for initial connection
                read=self.timeout,
                write=30.0,
                pool=None
            )
            client = httpx.AsyncClient(
                timeout=timeout_config,
                limits=httpx.Limits(max_connections=10, max_keepalive_connections=5),
            )
            close_client = True

        try:
            # Batch processing with adaptive batch size reduction for 413 errors
            results = []
            current_batch_size = self.batch_size
            i = 0
            
            while i < len(texts):
                # Calculate current batch
                batch_texts = texts[i : i + current_batch_size]
                batch_chunks = chunks[i : i + current_batch_size]
                
                if not batch_texts:
                    break

                # Call embeddings endpoint
                try:
                    response = await self._call_embeddings(
                        batch_texts, client, retries=3
                    )
                except httpx.HTTPStatusError as e:
                    if e.response.status_code == 413:
                        # Request too large - reduce batch size and retry
                        import logging
                        logger = logging.getLogger(__name__)
                        
                        if current_batch_size > 1:
                            # Reduce batch size by half (minimum 1)
                            new_batch_size = max(1, current_batch_size // 2)
                            logger.warning(
                                f"Request too large (413) with batch size {current_batch_size}. "
                                f"Reducing to {new_batch_size} and retrying..."
                            )
                            current_batch_size = new_batch_size
                            # Don't advance i, retry with smaller batch
                            continue
                        else:
                            # Already at minimum batch size, skip this item
                            logger.error(
                                f"Request too large even with single item (size: {len(batch_texts[0])} chars). "
                                f"Skipping this chunk."
                            )
                            results.append({"chunk": batch_chunks[0], "embedding": []})
                            i += 1
                            continue
                    else:
                        # Other HTTP errors, re-raise
                        raise

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
                    # Successfully processed batch, advance to next
                    i += len(batch_texts)
                else:
                    # Failed to get embeddings for this batch
                    for chunk in batch_chunks:
                        results.append({"chunk": chunk, "embedding": []})
                    i += len(batch_texts)

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
                elif e.response.status_code == 413:
                    # Request too large: reduce batch size
                    import logging
                    logger = logging.getLogger(__name__)
                    logger.warning(
                        f"Request too large (413) for batch of {len(texts)} items. "
                        f"Suggest reducing batch_size (current: {self.batch_size})"
                    )
                    # Re-raise to let caller handle batch size reduction
                    raise
                raise
            except httpx.ConnectError as e:
                # Connection failed - log details and re-raise
                import logging
                logger = logging.getLogger(__name__)
                logger.error(
                    f"Connection error to {url} (attempt {attempt + 1}/{retries}): {e}. "
                    f"Check if gpt2giga service is running at {self.gpt2giga_url}"
                )
                if attempt < retries - 1:
                    wait_time = 2 ** attempt
                    await asyncio.sleep(wait_time)
                    continue
                raise
            except httpx.TimeoutError as e:
                import logging
                logger = logging.getLogger(__name__)
                logger.error(
                    f"Timeout connecting to {url} (attempt {attempt + 1}/{retries}): {e}"
                )
                if attempt < retries - 1:
                    wait_time = 2 ** attempt
                    await asyncio.sleep(wait_time)
                    continue
                raise
            except Exception as e:
                import logging
                logger = logging.getLogger(__name__)
                logger.error(
                    f"Unexpected error calling {url} (attempt {attempt + 1}/{retries}): {e}",
                    exc_info=True
                )
                if attempt < retries - 1:
                    wait_time = 2 ** attempt
                    await asyncio.sleep(wait_time)
                    continue
                raise

        return None

