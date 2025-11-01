"""Unit tests for VectorDB implementations."""

import pytest

from gpt2giga.tools.vector_db import SimpleVectorDB, VectorDB, create_vector_db


class TestSimpleVectorDB:
    """Test cases for SimpleVectorDB."""

    @pytest.mark.asyncio
    async def test_upsert_documents(self):
        """Test upserting documents."""
        db = SimpleVectorDB()
        documents = [
            {
                "id": "doc1",
                "text": "Hello world",
                "embedding": [0.1, 0.2, 0.3],
                "metadata": {"file": "test.py"},
            },
            {
                "id": "doc2",
                "text": "Test content",
                "embedding": [0.4, 0.5, 0.6],
                "metadata": {"file": "test2.py"},
            },
        ]

        count = await db.upsert("test_collection", documents)

        assert count == 2
        assert len(db.collections["test_collection"]) == 2

    @pytest.mark.asyncio
    async def test_upsert_update_existing(self):
        """Test that upsert updates existing documents."""
        db = SimpleVectorDB()
        doc1 = {
            "id": "doc1",
            "text": "Original",
            "embedding": [0.1, 0.2],
            "metadata": {"file": "test.py"},
        }

        # Insert first time
        await db.upsert("test_collection", [doc1])

        # Update same document
        doc1_updated = doc1.copy()
        doc1_updated["text"] = "Updated"
        count = await db.upsert("test_collection", [doc1_updated])

        assert count == 1
        assert len(db.collections["test_collection"]) == 1
        assert db.collections["test_collection"][0]["text"] == "Updated"

    @pytest.mark.asyncio
    async def test_delete_by_metadata(self):
        """Test deleting documents by metadata filter."""
        db = SimpleVectorDB()
        documents = [
            {
                "id": "doc1",
                "text": "Hello",
                "embedding": [0.1, 0.2],
                "metadata": {"file_path": "/test.py", "language": "python"},
            },
            {
                "id": "doc2",
                "text": "World",
                "embedding": [0.3, 0.4],
                "metadata": {"file_path": "/test2.py", "language": "python"},
            },
            {
                "id": "doc3",
                "text": "Test",
                "embedding": [0.5, 0.6],
                "metadata": {"file_path": "/test.js", "language": "javascript"},
            },
        ]

        await db.upsert("test_collection", documents)

        # Delete documents with specific file_path
        deleted = await db.delete_by_metadata(
            "test_collection", {"file_path": "/test.py"}
        )

        assert deleted == 1
        assert len(db.collections["test_collection"]) == 2

    @pytest.mark.asyncio
    async def test_delete_by_metadata_nonexistent(self):
        """Test deleting with metadata filter that matches nothing."""
        db = SimpleVectorDB()
        await db.upsert(
            "test_collection",
            [
                {
                    "id": "doc1",
                    "text": "Hello",
                    "embedding": [0.1, 0.2],
                    "metadata": {"file": "test.py"},
                }
            ],
        )

        deleted = await db.delete_by_metadata(
            "test_collection", {"file": "nonexistent.py"}
        )

        assert deleted == 0
        assert len(db.collections["test_collection"]) == 1

    @pytest.mark.asyncio
    async def test_delete_by_metadata_empty_collection(self):
        """Test deleting from empty collection."""
        db = SimpleVectorDB()
        deleted = await db.delete_by_metadata("test_collection", {"file": "test.py"})

        assert deleted == 0

    @pytest.mark.asyncio
    async def test_close(self):
        """Test closing database (no-op for SimpleVectorDB)."""
        db = SimpleVectorDB()
        # Should not raise
        await db.close()


class TestCreateVectorDB:
    """Test cases for create_vector_db factory."""

    def test_create_simple_vector_db(self):
        """Test creating SimpleVectorDB."""
        db = create_vector_db(db_type="simple")

        assert isinstance(db, SimpleVectorDB)

    def test_create_memory_vector_db(self):
        """Test creating memory vector DB (alias for simple)."""
        db = create_vector_db(db_type="memory")

        assert isinstance(db, SimpleVectorDB)

    def test_create_qdrant_vector_db_without_dependency(self):
        """Test that creating Qdrant DB without dependency raises error."""
        with pytest.raises(ImportError, match="qdrant-client"):
            create_vector_db(db_type="qdrant")

    def test_create_unsupported_vector_db(self):
        """Test that creating unsupported DB type raises error."""
        with pytest.raises(ValueError, match="Unsupported"):
            create_vector_db(db_type="unsupported")

