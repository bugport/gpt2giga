"""Qdrant vector database adapter."""

from typing import List, Dict, Any, Optional
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from gpt2giga.tools.vector_db import VectorDB


class QdrantVectorDB(VectorDB):
    """Qdrant vector database adapter."""

    def __init__(self, url: Optional[str] = None, api_key: Optional[str] = None):
        if url:
            self.client = QdrantClient(url=url, api_key=api_key)
        else:
            self.client = QdrantClient(host="localhost", port=6333)

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

    async def close(self):
        """Close database connection."""
        # Qdrant client doesn't require explicit close
        pass

