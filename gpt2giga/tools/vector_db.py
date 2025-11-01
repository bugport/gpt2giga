"""Vector database interface for storing code embeddings."""

import os
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
    async def search(
        self,
        collection_name: str,
        query_embedding: List[float],
        top_k: int = 5,
        min_similarity: float = 0.7,
    ) -> List[Dict[str, Any]]:
        """Search for similar documents using cosine similarity.
        
        Returns list of documents with metadata, ordered by similarity (highest first).
        Each document dict contains:
        - text: str - document text
        - metadata: dict - document metadata
        - similarity: float - cosine similarity score (0.0 to 1.0)
        """
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

    async def search(
        self,
        collection_name: str,
        query_embedding: List[float],
        top_k: int = 5,
        min_similarity: float = 0.7,
    ) -> List[Dict[str, Any]]:
        """Search using cosine similarity (simple implementation)."""
        if collection_name not in self.collections:
            return []
        
        try:
            import numpy as np
        except ImportError:
            # Fallback to pure Python cosine similarity if numpy not available
            return self._search_pure_python(collection_name, query_embedding, top_k, min_similarity)
        
        results = []
        query_vec = np.array(query_embedding, dtype=float)
        query_norm = np.linalg.norm(query_vec)
        
        if query_norm == 0:
            return []
        
        for doc in self.collections[collection_name]:
            doc_embedding = doc.get("embedding", [])
            if not doc_embedding or len(doc_embedding) != len(query_embedding):
                continue
                
            try:
                doc_vec = np.array(doc_embedding, dtype=float)
                doc_norm = np.linalg.norm(doc_vec)
                
                if doc_norm == 0:
                    continue
                
                # Cosine similarity
                similarity = float(np.dot(query_vec, doc_vec) / (query_norm * doc_norm))
                
                if similarity >= min_similarity:
                    results.append({
                        "text": doc.get("text", ""),
                        "metadata": doc.get("metadata", {}),
                        "similarity": similarity,
                    })
            except (ValueError, TypeError):
                # Skip invalid embeddings
                continue
        
        # Sort by similarity (descending) and return top_k
        results.sort(key=lambda x: x["similarity"], reverse=True)
        return results[:top_k]
    
    def _search_pure_python(
        self,
        collection_name: str,
        query_embedding: List[float],
        top_k: int = 5,
        min_similarity: float = 0.7,
    ) -> List[Dict[str, Any]]:
        """Pure Python cosine similarity (fallback if numpy not available)."""
        if collection_name not in self.collections:
            return []
        
        def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
            """Calculate cosine similarity between two vectors."""
            if len(vec1) != len(vec2):
                return 0.0
            
            dot_product = sum(a * b for a, b in zip(vec1, vec2))
            norm1 = sum(a * a for a in vec1) ** 0.5
            norm2 = sum(b * b for b in vec2) ** 0.5
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            return dot_product / (norm1 * norm2)
        
        results = []
        for doc in self.collections[collection_name]:
            doc_embedding = doc.get("embedding", [])
            if not doc_embedding:
                continue
            
            try:
                similarity = cosine_similarity(query_embedding, doc_embedding)
                
                if similarity >= min_similarity:
                    results.append({
                        "text": doc.get("text", ""),
                        "metadata": doc.get("metadata", {}),
                        "similarity": similarity,
                    })
            except (ValueError, TypeError):
                continue
        
        results.sort(key=lambda x: x["similarity"], reverse=True)
        return results[:top_k]

    async def close(self):
        """Close database connection."""
        pass


def create_vector_db(
    db_type: str = "simple",
    db_url: Optional[str] = None,
    api_key: Optional[str] = None,
) -> VectorDB:
    """Factory function to create vector DB instance."""
    # Normalize db_type to lowercase for case-insensitive matching
    db_type_normalized = db_type.lower().strip() if db_type else "simple"
    
    if db_type_normalized == "simple" or db_type_normalized == "memory":
        return SimpleVectorDB()
    elif db_type_normalized == "qdrant":
        try:
            from gpt2giga.tools.vector_db_qdrant import QdrantVectorDB
        except ImportError as e:
            raise ImportError(
                f"Qdrant support requires 'qdrant-client' package. "
                f"Install with: pip install qdrant-client\n"
                f"Original error: {e}"
            ) from e
        except Exception as e:
            raise RuntimeError(
                f"Failed to import QdrantVectorDB module: {e}. "
                f"Make sure the module file exists at gpt2giga/tools/vector_db_qdrant.py"
            ) from e
        
        try:
            # Get verify_ssl from env if not explicitly passed
            verify_ssl_str = os.getenv("GPT2GIGA_QDRANT_VERIFY_SSL", "").lower() if db_url else None
            verify_ssl = None
            if verify_ssl_str in ("false", "0", "no", "off"):
                verify_ssl = False
            elif verify_ssl_str in ("true", "1", "yes", "on"):
                verify_ssl = True
            
            return QdrantVectorDB(url=db_url, api_key=api_key, verify_ssl=verify_ssl)
        except Exception as e:
            raise RuntimeError(
                f"Failed to create Qdrant vector DB: {e}. "
                f"URL: {db_url or 'localhost:6333'}. "
                f"Check if Qdrant server is running and accessible."
            ) from e
    else:
        raise ValueError(
            f"Unsupported vector DB type: '{db_type}' (normalized: '{db_type_normalized}'). "
            f"Supported types: 'simple', 'memory', 'qdrant'"
        )

