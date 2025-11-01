"""Vector database interface for storing code embeddings."""

from typing import List, Dict, Any, Optional
from abc import ABC, abstractmethod


class VectorDB(ABC):
    """Abstract base class for vector database adapters."""

    @abstractmethod
    async def upsert(
        self,
        collection_name: str,
        documents: List[Dict[str, Any]],
    ) -> int:
        """Upsert documents into collection. Returns number of documents upserted."""
        pass

    @abstractmethod
    async def delete_by_metadata(
        self, collection_name: str, metadata_filter: Dict[str, Any]
    ) -> int:
        """Delete documents matching metadata filter. Returns number deleted."""
        pass

    @abstractmethod
    async def close(self):
        """Close database connection."""
        pass


class SimpleVectorDB(VectorDB):
    """Simple in-memory vector DB (for testing/development)."""

    def __init__(self):
        self.collections: Dict[str, List[Dict[str, Any]]] = {}

    async def upsert(
        self,
        collection_name: str,
        documents: List[Dict[str, Any]],
    ) -> int:
        """Upsert documents into collection."""
        if collection_name not in self.collections:
            self.collections[collection_name] = []

        existing_ids = {
            doc.get("id") for doc in self.collections[collection_name]
        }

        count = 0
        for doc in documents:
            doc_id = doc.get("id")
            if doc_id in existing_ids:
                # Update existing
                idx = next(
                    i
                    for i, d in enumerate(self.collections[collection_name])
                    if d.get("id") == doc_id
                )
                self.collections[collection_name][idx] = doc
            else:
                # Add new
                self.collections[collection_name].append(doc)
            count += 1

        return count

    async def delete_by_metadata(
        self, collection_name: str, metadata_filter: Dict[str, Any]
    ) -> int:
        """Delete documents matching metadata filter."""
        if collection_name not in self.collections:
            return 0

        original_count = len(self.collections[collection_name])
        self.collections[collection_name] = [
            doc
            for doc in self.collections[collection_name]
            if not self._matches_filter(doc.get("metadata", {}), metadata_filter)
        ]

        return original_count - len(self.collections[collection_name])

    def _matches_filter(self, metadata: Dict[str, Any], filter: Dict[str, Any]) -> bool:
        """Check if metadata matches filter."""
        for key, value in filter.items():
            if metadata.get(key) != value:
                return False
        return True

    async def close(self):
        """Close database connection."""
        pass


def create_vector_db(
    db_type: str = "simple",
    db_url: Optional[str] = None,
    api_key: Optional[str] = None,
) -> VectorDB:
    """Factory function to create vector DB instance."""
    if db_type == "simple" or db_type == "memory":
        return SimpleVectorDB()
    elif db_type == "qdrant":
        try:
            from gpt2giga.tools.vector_db_qdrant import QdrantVectorDB

            return QdrantVectorDB(url=db_url, api_key=api_key)
        except ImportError:
            raise ImportError(
                "Qdrant support requires 'qdrant-client' package. "
                "Install with: pip install qdrant-client"
            )
    else:
        raise ValueError(f"Unsupported vector DB type: {db_type}")

