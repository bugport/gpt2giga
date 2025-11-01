"""Main codebase indexer orchestrator and CLI command."""

import argparse
import asyncio
import os
import sys
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
import hashlib

from gpt2giga.tools.scanner import CodebaseScanner, FileInfo
from gpt2giga.tools.parser import CodeParser, CodeChunk
from gpt2giga.tools.chunker import CodeChunker
from gpt2giga.tools.embedder import CodebaseEmbedder
from gpt2giga.tools.vector_db import create_vector_db, VectorDB


class CodebaseIndexer:
    """Orchestrates the codebase indexing workflow."""

    def __init__(
        self,
        codebase_path: str,
        collection_name: str,
        gpt2giga_url: str,
        vector_db: VectorDB,
        embedder: CodebaseEmbedder,
        scanner: CodebaseScanner,
        parser: CodeParser,
        chunker: CodeChunker,
        incremental: bool = True,
        logger: Optional[logging.Logger] = None,
    ):
        self.codebase_path = codebase_path
        self.collection_name = collection_name
        self.gpt2giga_url = gpt2giga_url
        self.vector_db = vector_db
        self.embedder = embedder
        self.scanner = scanner
        self.parser = parser
        self.chunker = chunker
        self.incremental = incremental
        self.logger = logger or logging.getLogger(__name__)

    async def index_codebase(self) -> Dict[str, Any]:
        """Index codebase: scan → parse → chunk → embed → store."""
        stats = {
            "files_scanned": 0,
            "files_parsed": 0,
            "chunks_created": 0,
            "embeddings_generated": 0,
            "documents_stored": 0,
            "files_skipped": 0,
            "errors": [],
        }

        self.logger.info(f"Scanning codebase: {self.codebase_path}")
        files = self.scanner.scan_directory(self.codebase_path)
        stats["files_scanned"] = len(files)
        self.logger.info(f"Found {len(files)} files to index")

        all_chunks = []
        file_chunk_map = {}  # Map file path to its chunks for deletion tracking

        # Parse and chunk files
        for file_info in files:
            try:
                self.logger.debug(f"Parsing: {file_info.path}")
                with open(file_info.path, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()

                chunks = self.parser.parse_file(file_info.path, content)
                if not chunks:
                    stats["files_skipped"] += 1
                    continue

                # Chunk appropriately
                chunked = self.chunker.chunk_code(chunks)
                stats["files_parsed"] += 1
                stats["chunks_created"] += len(chunked)

                all_chunks.extend(chunked)
                file_chunk_map[file_info.path] = chunked
            except Exception as e:
                error_msg = f"Error processing {file_info.path}: {str(e)}"
                self.logger.warning(error_msg)
                stats["errors"].append(error_msg)
                stats["files_skipped"] += 1

        if not all_chunks:
            self.logger.warning("No chunks to index")
            return stats

        self.logger.info(f"Generated {len(all_chunks)} chunks, generating embeddings...")

        # Generate embeddings
        embedding_results = await self.embedder.generate_embeddings(all_chunks)
        valid_results = [r for r in embedding_results if r.get("embedding")]
        stats["embeddings_generated"] = len(valid_results)

        if not valid_results:
            self.logger.error("No valid embeddings generated")
            return stats

        self.logger.info(f"Generated {len(valid_results)} embeddings, storing in vector DB...")

        # Prepare documents for vector DB
        documents = []
        for result in valid_results:
            chunk = result["chunk"]
            embedding = result["embedding"]
            chunk_id = f"{chunk.file_path}:{chunk.start_line}:{chunk.end_line}"
            chunk_id_hash = hashlib.sha256(chunk_id.encode()).hexdigest()[:16]

            # Get file hash for incremental updates
            file_hash = ""
            try:
                file_hash_obj = hashlib.sha256()
                with open(chunk.file_path, "rb") as f:
                    file_hash_obj.update(f.read())
                file_hash = file_hash_obj.hexdigest()
            except Exception:
                pass

            documents.append(
                {
                    "id": chunk_id_hash,
                    "text": chunk.text,
                    "embedding": embedding,
                    "metadata": {
                        "file_path": chunk.file_path,
                        "start_line": chunk.start_line,
                        "end_line": chunk.end_line,
                        "chunk_type": chunk.chunk_type,
                        "language": chunk.language,
                        "name": chunk.name or "",
                        "file_hash": file_hash,
                    },
                }
            )

        # Store in vector DB
        try:
            stored_count = await self.vector_db.upsert(
                self.collection_name, documents
            )
            stats["documents_stored"] = stored_count
            self.logger.info(f"Stored {stored_count} documents in collection '{self.collection_name}'")
        except Exception as e:
            error_msg = f"Error storing documents: {str(e)}"
            self.logger.error(error_msg)
            stats["errors"].append(error_msg)

        return stats


def index_codebase_cli():
    """CLI entry point for index-codebase command."""
    parser = argparse.ArgumentParser(
        description="Index a codebase into the vector database for RAG",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "codebase_path",
        type=str,
        help="Path to codebase directory to index",
    )

    parser.add_argument(
        "--collection-name",
        type=str,
        default=None,
        help="Vector DB collection name (default: from GPT2GIGA_INDEXER_COLLECTION_NAME or 'codebase')",
    )

    parser.add_argument(
        "--gpt2giga-url",
        type=str,
        default=None,
        help="gpt2giga API URL (default: from GPT2GIGA_INDEXER_GPT2GIGA_URL or http://localhost:8090)",
    )

    parser.add_argument(
        "--vector-db-type",
        type=str,
        default=None,
        help="Vector DB type: simple, qdrant (default: from GPT2GIGA_INDEXER_VECTOR_DB_TYPE or 'simple')",
    )

    parser.add_argument(
        "--vector-db-url",
        type=str,
        default=None,
        help="Vector DB URL (default: from GPT2GIGA_INDEXER_VECTOR_DB_URL)",
    )

    parser.add_argument(
        "--vector-db-api-key",
        type=str,
        default=None,
        help="Vector DB API key (default: from GPT2GIGA_INDEXER_VECTOR_DB_API_KEY)",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Batch size for embeddings (default: from GPT2GIGA_INDEXER_BATCH_SIZE or 100)",
    )

    parser.add_argument(
        "--chunk-size",
        type=int,
        default=None,
        help="Target tokens per chunk (default: from GPT2GIGA_INDEXER_CHUNK_SIZE or 2000)",
    )

    parser.add_argument(
        "--include-patterns",
        type=str,
        default=None,
        help="Comma-separated file extensions to include (e.g., '.py,.js,.ts')",
    )

    parser.add_argument(
        "--exclude-patterns",
        type=str,
        default=None,
        help="Comma-separated patterns to exclude",
    )

    parser.add_argument(
        "--incremental",
        action="store_true",
        default=None,
        help="Only index changed files (default: from GPT2GIGA_INDEXER_INCREMENTAL or True)",
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-index all files (overrides incremental)",
    )

    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"],
        help="Log level",
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger(__name__)

    # Load configuration from environment
    collection_name = (
        args.collection_name
        or os.getenv("GPT2GIGA_INDEXER_COLLECTION_NAME")
        or "codebase"
    )
    gpt2giga_url = (
        args.gpt2giga_url
        or os.getenv("GPT2GIGA_INDEXER_GPT2GIGA_URL")
        or "http://localhost:8090"
    )
    vector_db_type = (
        args.vector_db_type
        or os.getenv("GPT2GIGA_INDEXER_VECTOR_DB_TYPE")
        or "simple"
    )
    vector_db_url = args.vector_db_url or os.getenv("GPT2GIGA_INDEXER_VECTOR_DB_URL")
    vector_db_api_key = (
        args.vector_db_api_key or os.getenv("GPT2GIGA_INDEXER_VECTOR_DB_API_KEY")
    )
    batch_size = int(
        args.batch_size or os.getenv("GPT2GIGA_INDEXER_BATCH_SIZE", "100") or 100
    )
    chunk_size = int(
        args.chunk_size or os.getenv("GPT2GIGA_INDEXER_CHUNK_SIZE", "2000") or 2000
    )

    # Default include patterns (C++, SQL, and common languages)
    default_include = ".py,.js,.ts,.java,.go,.rs,.cpp,.cc,.cxx,.c,.h,.hpp,.hxx,.sql"
    include_patterns_str = (
        args.include_patterns
        or os.getenv("GPT2GIGA_INDEXER_INCLUDE_PATTERNS")
        or default_include
    )
    include_patterns = [p.strip() for p in include_patterns_str.split(",") if p.strip()]

    exclude_patterns_str = (
        args.exclude_patterns
        or os.getenv("GPT2GIGA_INDEXER_EXCLUDE_PATTERNS")
        or "node_modules,venv,.venv,.git,__pycache__,*.pyc"
    )
    exclude_patterns = [
        p.strip() for p in exclude_patterns_str.split(",") if p.strip()
    ]

    incremental = args.force is False and (
        args.incremental is True
        or os.getenv("GPT2GIGA_INDEXER_INCREMENTAL", "true").lower() == "true"
    )

    # Validate codebase path
    codebase_path = Path(args.codebase_path).resolve()
    if not codebase_path.exists():
        logger.error(f"Codebase path does not exist: {codebase_path}")
        sys.exit(1)

    if not codebase_path.is_dir():
        logger.error(f"Codebase path is not a directory: {codebase_path}")
        sys.exit(1)

    logger.info(f"Indexing codebase: {codebase_path}")
    logger.info(f"Collection: {collection_name}")
    # Normalize gpt2giga URL to ensure it has a scheme
    gpt2giga_url_normalized = gpt2giga_url.strip().rstrip("/")
    if not gpt2giga_url_normalized.startswith(("http://", "https://")):
        # Default to http:// if no scheme provided
        gpt2giga_url_normalized = f"http://{gpt2giga_url_normalized}"
        logger.info(f"Normalized gpt2giga URL: {gpt2giga_url} -> {gpt2giga_url_normalized}")
    else:
        logger.info(f"gpt2giga URL: {gpt2giga_url_normalized}")
    gpt2giga_url = gpt2giga_url_normalized
    logger.info(f"Vector DB type: {vector_db_type}")
    logger.info(f"Incremental: {incremental}")

    # Create components
    try:
        logger.info(f"Creating vector DB (type: {vector_db_type}, url: {vector_db_url or 'default'})...")
        vector_db = create_vector_db(
            db_type=vector_db_type, db_url=vector_db_url, api_key=vector_db_api_key
        )
        logger.info(f"Vector DB created successfully: {type(vector_db).__name__}")
    except Exception as e:
        logger.error(f"Failed to create vector DB: {e}", exc_info=True)
        logger.error(
            f"Troubleshooting:\n"
            f"  - For Qdrant: Make sure Qdrant server is running (check http://localhost:6333/health)\n"
            f"  - For simple: No additional setup needed\n"
            f"  - Check URL format: {vector_db_url or 'N/A'}\n"
            f"  - Verify vector DB type: {vector_db_type}"
        )
        sys.exit(1)

    embedder = CodebaseEmbedder(
        gpt2giga_url=gpt2giga_url, batch_size=batch_size
    )
    scanner = CodebaseScanner(
        include_patterns=include_patterns, exclude_patterns=exclude_patterns
    )
    parser = CodeParser()
    chunker = CodeChunker(max_tokens=chunk_size)

    indexer = CodebaseIndexer(
        codebase_path=str(codebase_path),
        collection_name=collection_name,
        gpt2giga_url=gpt2giga_url,
        vector_db=vector_db,
        embedder=embedder,
        scanner=scanner,
        parser=parser,
        chunker=chunker,
        incremental=incremental,
        logger=logger,
    )

    # Run indexing
    async def run_indexing():
        try:
            stats = await indexer.index_codebase()

            logger.info("Indexing completed:")
            logger.info(f"  Files scanned: {stats['files_scanned']}")
            logger.info(f"  Files parsed: {stats['files_parsed']}")
            logger.info(f"  Chunks created: {stats['chunks_created']}")
            logger.info(f"  Embeddings generated: {stats['embeddings_generated']}")
            logger.info(f"  Documents stored: {stats['documents_stored']}")
            logger.info(f"  Files skipped: {stats['files_skipped']}")

            if stats["errors"]:
                logger.warning(f"  Errors: {len(stats['errors'])}")
                for error in stats["errors"][:10]:  # Show first 10 errors
                    logger.warning(f"    - {error}")

            # Close vector DB
            await vector_db.close()
        except KeyboardInterrupt:
            logger.info("Interrupted by user")
            await vector_db.close()
            sys.exit(1)
        except Exception as e:
            logger.error(f"Indexing failed: {e}", exc_info=True)
            try:
                await vector_db.close()
            except Exception:
                pass
            sys.exit(1)

    asyncio.run(run_indexing())

