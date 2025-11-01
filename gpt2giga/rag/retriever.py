"""RAG retriever for code context retrieval."""

import os
import json
import httpx
import asyncio
from typing import List, Dict, Any, Optional
import logging

from gpt2giga.tools.vector_db import VectorDB


class RAGRetriever:
    """Retrieves relevant code context using RAG."""
    
    def __init__(
        self,
        vector_db: VectorDB,
        collection_name: str,
        gpt2giga_url: str = "http://localhost:8090",
        embedding_model: str = "EmbeddingsGigaR",
        top_k: int = 5,
        min_similarity: float = 0.7,
        logger: Optional[logging.Logger] = None,
    ):
        self.vector_db = vector_db
        self.collection_name = collection_name
        self.gpt2giga_url = gpt2giga_url.rstrip("/")
        self.embedding_model = embedding_model
        self.top_k = top_k
        self.min_similarity = min_similarity
        self.logger = logger or logging.getLogger(__name__)
    
    async def retrieve_context(self, query: str) -> List[Dict[str, Any]]:
        """Retrieve relevant code context for a query."""
        # Generate embedding for query
        query_embedding = await self._generate_query_embedding(query)
        if not query_embedding:
            self.logger.warning("Failed to generate query embedding")
            return []
        
        # Search vector DB
        results = await self.vector_db.search(
            collection_name=self.collection_name,
            query_embedding=query_embedding,
            top_k=self.top_k,
            min_similarity=self.min_similarity,
        )
        
        self.logger.info(f"Retrieved {len(results)} relevant code chunks")
        return results
    
    async def _generate_query_embedding(self, query: str) -> Optional[List[float]]:
        """Generate embedding for query using gpt2giga /embeddings endpoint."""
        url = f"{self.gpt2giga_url}/embeddings"
        if not url.endswith("/embeddings"):
            # Try /v1/embeddings if needed
            if "/v1" not in url:
                url = f"{self.gpt2giga_url}/v1/embeddings"
        
        payload = {
            "model": self.embedding_model,
            "input": query,
        }
        
        try:
            # Use a separate HTTP client with connection pooling disabled to avoid loops
            async with httpx.AsyncClient(
                timeout=httpx.Timeout(30.0, connect=10.0),
                limits=httpx.Limits(max_connections=1, max_keepalive_connections=0),
                follow_redirects=False,  # Prevent redirect loops
            ) as client:
                try:
                    response = await client.post(url, json=payload, follow_redirects=False)
                    response.raise_for_status()
                    data = response.json()
                    
                    embedding_data = data.get("data", [])
                    if embedding_data:
                        embedding = embedding_data[0].get("embedding", [])
                        # Handle base64 if needed
                        if isinstance(embedding, str):
                            import base64
                            try:
                                # Decode base64 if needed
                                embedding_bytes = base64.b64decode(embedding)
                                # Convert bytes to list if needed
                                if isinstance(embedding_bytes, bytes):
                                    import struct
                                    # Try to decode as float array
                                    embedding = list(
                                        struct.unpack(
                                            f"{len(embedding_bytes)//4}f", embedding_bytes
                                        )
                                    )
                            except Exception:
                                # If decoding fails, try to parse as JSON
                                try:
                                    embedding = json.loads(embedding)
                                except Exception:
                                    pass
                        
                        return embedding if isinstance(embedding, list) else []
                except httpx.HTTPStatusError as e:
                    self.logger.error(f"HTTP error generating query embedding: {e.response.status_code} - {e.response.text[:200]}")
                    return None
                except httpx.RequestError as e:
                    self.logger.error(f"Request error generating query embedding: {e}")
                    return None
        except Exception as e:
            self.logger.error(f"Error generating query embedding: {e}", exc_info=True)
            return None
    
    def format_context_for_messages(self, results: List[Dict[str, Any]]) -> str:
        """Format retrieved code chunks for injection into messages."""
        if not results:
            return ""
        
        context_parts = []
        for i, result in enumerate(results, 1):
            metadata = result.get("metadata", {})
            text = result.get("text", "")
            file_path = metadata.get("file_path", "unknown")
            start_line = metadata.get("start_line", 0)
            end_line = metadata.get("end_line", 0)
            chunk_type = metadata.get("chunk_type", "code")
            similarity = result.get("similarity", 0.0)
            
            context_parts.append(
                f"--- Code Chunk {i} ({similarity:.2f} similarity) ---\n"
                f"File: {file_path} (lines {start_line}-{end_line}, type: {chunk_type})\n"
                f"{text}\n"
            )
        
        return "\n".join(context_parts)

