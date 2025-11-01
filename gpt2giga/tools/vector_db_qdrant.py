"""Qdrant vector database adapter."""

import os
from typing import List, Dict, Any, Optional
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from gpt2giga.tools.vector_db import VectorDB


class QdrantVectorDB(VectorDB):
    """Qdrant vector database adapter."""

    def __init__(
        self, 
        url: Optional[str] = None, 
        api_key: Optional[str] = None,
        verify_ssl: Optional[bool] = None,
    ):
        # Determine SSL verification setting
        # Default: True for HTTPS, False for HTTP, or from env var
        if verify_ssl is None:
            verify_ssl_str = os.getenv("GPT2GIGA_QDRANT_VERIFY_SSL", "").lower()
            if verify_ssl_str in ("false", "0", "no", "off"):
                verify_ssl = False
            elif verify_ssl_str in ("true", "1", "yes", "on"):
                verify_ssl = True
            else:
                # Auto-detect from URL: False for HTTP, True for HTTPS (unless env says otherwise)
                verify_ssl = None  # Let QdrantClient use defaults
        
        try:
            if url:
                # Normalize URL: ensure it has a scheme
                url_normalized = url.strip().rstrip("/")
                if not url_normalized.startswith(("http://", "https://")):
                    # Default to http:// if no scheme provided
                    url_normalized = f"http://{url_normalized}"
                
                # Parse URL to extract components for better error handling
                try:
                    from urllib.parse import urlparse
                    parsed = urlparse(url_normalized)
                    host = parsed.hostname or "localhost"
                    port = parsed.port or (443 if parsed.scheme == "https" else 80)
                    use_https = parsed.scheme == "https"
                    prefix = parsed.path.rstrip("/") if parsed.path and parsed.path != "/" else None
                    
                    # Build client kwargs
                    client_kwargs = {
                        "host": host,
                        "port": port,
                        "https": use_https,
                    }
                    if prefix:
                        client_kwargs["prefix"] = prefix
                    if api_key:
                        client_kwargs["api_key"] = api_key
                    
                    # Add SSL verification setting if explicitly set
                    # Note: QdrantClient may use httpx internally, which supports verify parameter
                    # Some versions might not support this, so we'll try it and catch errors
                    if verify_ssl is not None:
                        try:
                            # Try to pass verify parameter (for newer versions that support it)
                            client_kwargs["verify"] = verify_ssl
                        except Exception:
                            # If verify is not supported, we'll handle it in the exception below
                            pass
                    
                    self.client = QdrantClient(**client_kwargs)
                except Exception as parse_error:
                    # Fallback to URL-based initialization if parsing fails
                    client_kwargs = {"url": url_normalized}
                    if api_key:
                        client_kwargs["api_key"] = api_key
                    if verify_ssl is not None:
                        try:
                            client_kwargs["verify"] = verify_ssl
                        except Exception:
                            pass
                    self.client = QdrantClient(**client_kwargs)
            else:
                # Default to localhost:6333
                self.client = QdrantClient(host="localhost", port=6333)
        except FileNotFoundError as e:
            raise RuntimeError(
                f"Failed to initialize Qdrant client: File not found error. "
                f"This might indicate QdrantClient is trying to access a local file. "
                f"URL: {url or 'localhost:6333'}, API key: {'provided' if api_key else 'none'}. "
                f"Original error: {e}"
            ) from e
        except (OSError, ValueError, AttributeError) as e:
            # Common SSL context errors
            error_str = str(e).lower()
            if any(keyword in error_str for keyword in ['ssl', 'context', 'certificate', 'verify', 'cert']):
                raise RuntimeError(
                    f"Failed to initialize Qdrant client due to SSL/TLS error: {e}. "
                    f"URL: {url or 'localhost:6333'}, HTTPS: {url and url.startswith('https://') if url else False}. "
                    f"This might be due to:\n"
                    f"  - SSL certificate verification failing (try setting GPT2GIGA_QDRANT_VERIFY_SSL=false)\n"
                    f"  - Missing SSL certificates on the system\n"
                    f"  - Using HTTPS when server only supports HTTP (try http:// instead)\n"
                    f"If using HTTP (not HTTPS), this error shouldn't occur. "
                    f"Check your URL: ensure it uses 'http://' for non-SSL connections."
                ) from e
            raise RuntimeError(
                f"Failed to initialize Qdrant client: {e}. "
                f"URL: {url or 'localhost:6333'}, API key: {'provided' if api_key else 'none'}. "
                f"Make sure Qdrant server is running and accessible at the URL."
            ) from e
        except Exception as e:
            raise RuntimeError(
                f"Failed to initialize Qdrant client: {e}. "
                f"URL: {url or 'localhost:6333'}, API key: {'provided' if api_key else 'none'}. "
                f"Make sure Qdrant server is running and accessible at the URL."
            ) from e

    async def _ensure_collection(self, collection_name: str, vector_size: int = 1024):
        """Ensure collection exists with proper configuration."""
        try:
            collections = self.client.get_collections().collections
            existing = any(c.name == collection_name for c in collections)
            if not existing:
                self.client.create_collection(
                    collection_name=collection_name,
                    vectors_config=VectorParams(
                        size=vector_size, distance=Distance.COSINE
                    ),
                )
        except Exception:
            pass

    async def upsert(
        self,
        collection_name: str,
        documents: List[Dict[str, Any]],
    ) -> int:
        """Upsert documents into collection."""
        if not documents:
            return 0

        # Get vector size from first document
        first_embedding = documents[0].get("embedding", [])
        vector_size = len(first_embedding) if first_embedding else 1024

        # Ensure collection exists
        await self._ensure_collection(collection_name, vector_size)

        # Prepare points
        points = []
        for doc in documents:
            doc_id = doc.get("id")
            if isinstance(doc_id, str):
                # Convert string ID to integer if possible
                try:
                    point_id = int(doc_id)
                except ValueError:
                    # Use hash for string IDs
                    point_id = hash(doc_id) % (2 ** 63)
            else:
                point_id = doc_id

            embedding = doc.get("embedding", [])
            payload = doc.get("metadata", {})
            payload["text"] = doc.get("text", "")

            points.append(
                PointStruct(
                    id=point_id,
                    vector=embedding,
                    payload=payload,
                )
            )

        # Upsert points
        self.client.upsert(collection_name=collection_name, points=points)
        return len(points)

    async def delete_by_metadata(
        self, collection_name: str, metadata_filter: Dict[str, Any]
    ) -> int:
        """Delete documents matching metadata filter."""
        from qdrant_client.models import Filter, FieldCondition, MatchValue

        # Build filter conditions
        conditions = []
        for key, value in metadata_filter.items():
            conditions.append(
                FieldCondition(key=key, match=MatchValue(value=value))
            )

        filter_obj = Filter(must=conditions)

        # Delete by filter
        result = self.client.delete(
            collection_name=collection_name, points_selector=filter_obj
        )
        return result.deleted_count if hasattr(result, "deleted_count") else 0

    async def search(
        self,
        collection_name: str,
        query_embedding: List[float],
        top_k: int = 5,
        min_similarity: float = 0.7,
    ) -> List[Dict[str, Any]]:
        """Search for similar documents using Qdrant search API."""
        from qdrant_client.models import SearchRequest
        
        try:
            # Search using Qdrant API
            search_results = self.client.search(
                collection_name=collection_name,
                query_vector=query_embedding,
                limit=top_k,
                score_threshold=min_similarity,
            )
            
            results = []
            for result in search_results:
                payload = result.payload or {}
                results.append({
                    "text": payload.get("text", ""),
                    "metadata": {k: v for k, v in payload.items() if k != "text"},
                    "similarity": float(result.score) if hasattr(result, "score") else 0.0,
                })
            
            return results
        except Exception:
            # Return empty list on error
            return []

    async def close(self):
        """Close database connection."""
        # Qdrant client doesn't require explicit close
        pass

