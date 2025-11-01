"""Code chunker for splitting code into appropriately sized chunks for embedding."""

import tiktoken
from typing import List, Optional


class CodeChunker:
    """Chunks code appropriately for embedding."""

    def __init__(self, max_tokens: int = 1000, model: str = "gpt-4"):
        self.max_tokens = max_tokens
        self.encoding = tiktoken.encoding_for_model(model)

    def _count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        try:
            return len(self.encoding.encode(text))
        except Exception:
            # Fallback: approximate 4 chars per token
            return len(text) // 4

    def chunk_code(self, chunks: List) -> List:
        """Split large chunks and merge small chunks to respect token limits."""
        result = []
        current_chunks = []
        current_tokens = 0

        for chunk in chunks:
            chunk_tokens = self._count_tokens(chunk.text)

            # If chunk exceeds limit, split it
            if chunk_tokens > self.max_tokens:
                # Flush current accumulated chunks
                if current_chunks:
                    merged = self._merge_chunks(current_chunks)
                    result.append(merged)
                    current_chunks = []
                    current_tokens = 0

                # Split large chunk
                split_chunks = self._split_chunk(chunk)
                result.extend(split_chunks)
            else:
                # Try to add to current batch
                if current_tokens + chunk_tokens <= self.max_tokens:
                    current_chunks.append(chunk)
                    current_tokens += chunk_tokens
                else:
                    # Flush and start new batch
                    if current_chunks:
                        merged = self._merge_chunks(current_chunks)
                        result.append(merged)
                    current_chunks = [chunk]
                    current_tokens = chunk_tokens

        # Flush remaining chunks
        if current_chunks:
            merged = self._merge_chunks(current_chunks)
            result.append(merged)

        return result

    def _split_chunk(self, chunk) -> List:
        """Split a large chunk into smaller pieces."""
        lines = chunk.text.split("\n")
        result = []
        current_lines = []
        current_tokens = 0

        for line in lines:
            line_tokens = self._count_tokens(line)
            if current_tokens + line_tokens > self.max_tokens and current_lines:
                # Create chunk from current lines
                new_chunk = type(chunk)(
                    text="\n".join(current_lines),
                    file_path=chunk.file_path,
                    start_line=chunk.start_line,
                    end_line=chunk.start_line + len(current_lines) - 1,
                    chunk_type=chunk.chunk_type,
                    language=chunk.language,
                    name=chunk.name,
                )
                result.append(new_chunk)
                current_lines = []
                current_tokens = 0

            current_lines.append(line)
            current_tokens += line_tokens

        # Add remaining lines
        if current_lines:
            new_chunk = type(chunk)(
                text="\n".join(current_lines),
                file_path=chunk.file_path,
                start_line=chunk.start_line + len(result) * 50,
                end_line=chunk.end_line,
                chunk_type=chunk.chunk_type,
                language=chunk.language,
                name=chunk.name,
            )
            result.append(new_chunk)

        return result

    def _merge_chunks(self, chunks: List):
        """Merge multiple small chunks into one."""
        if not chunks:
            return []

        if len(chunks) == 1:
            return chunks[0]

        # Merge text
        merged_text = "\n\n".join(chunk.text for chunk in chunks)
        first_chunk = chunks[0]
        last_chunk = chunks[-1]

        # Create merged chunk
        merged = type(first_chunk)(
            text=merged_text,
            file_path=first_chunk.file_path,
            start_line=first_chunk.start_line,
            end_line=last_chunk.end_line,
            chunk_type=first_chunk.chunk_type,
            language=first_chunk.language,
            name=first_chunk.name,
        )

        return merged

