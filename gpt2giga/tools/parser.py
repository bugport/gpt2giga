"""Code parser for extracting meaningful chunks from code files."""

import ast
from typing import List, Optional, Dict, Any
from pathlib import Path


class CodeChunk:
    """A chunk of code with metadata."""

    def __init__(
        self,
        text: str,
        file_path: str,
        start_line: int,
        end_line: int,
        chunk_type: str,
        language: str,
        name: Optional[str] = None,
    ):
        self.text = text
        self.file_path = file_path
        self.start_line = start_line
        self.end_line = end_line
        self.chunk_type = chunk_type  # 'function', 'class', 'module', 'line'
        self.language = language
        self.name = name  # Function/class name


class CodeParser:
    """Parses code files and extracts meaningful chunks."""

    def __init__(self):
        self.supported_languages = {
            ".py": "python",
            ".js": "javascript",
            ".ts": "typescript",
            ".jsx": "javascript",
            ".tsx": "typescript",
            ".java": "java",
            ".go": "go",
            ".rs": "rust",
            ".cpp": "cpp",
            ".cc": "cpp",
            ".cxx": "cpp",
            ".c": "c",
            ".h": "c",
            ".hpp": "cpp",
            ".hxx": "cpp",
            ".sql": "sql",
        }

    def _get_language(self, file_path: str) -> str:
        """Detect language from file extension."""
        ext = Path(file_path).suffix.lower()
        return self.supported_languages.get(ext, "unknown")

    def _parse_python(self, file_path: str, content: str) -> List[CodeChunk]:
        """Parse Python file using AST."""
        chunks = []
        try:
            tree = ast.parse(content, filename=file_path)
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    start = node.lineno
                    end = node.end_lineno if hasattr(node, "end_lineno") else start
                    func_code = "\n".join(content.split("\n")[start - 1 : end])
                    docstring = ast.get_docstring(node) or ""
                    chunks.append(
                        CodeChunk(
                            text=func_code,
                            file_path=file_path,
                            start_line=start,
                            end_line=end,
                            chunk_type="function",
                            language="python",
                            name=node.name,
                        )
                    )
                elif isinstance(node, ast.ClassDef):
                    start = node.lineno
                    end = node.end_lineno if hasattr(node, "end_lineno") else start
                    class_code = "\n".join(content.split("\n")[start - 1 : end])
                    docstring = ast.get_docstring(node) or ""
                    chunks.append(
                        CodeChunk(
                            text=class_code,
                            file_path=file_path,
                            start_line=start,
                            end_line=end,
                            chunk_type="class",
                            language="python",
                            name=node.name,
                        )
                    )
        except SyntaxError:
            # Fallback to line-based chunking
            pass

        # If no AST chunks found, use line-based fallback
        if not chunks:
            lines = content.split("\n")
            chunk_size = 50  # lines per chunk
            for i in range(0, len(lines), chunk_size):
                chunk_lines = lines[i : i + chunk_size]
                chunks.append(
                    CodeChunk(
                        text="\n".join(chunk_lines),
                        file_path=file_path,
                        start_line=i + 1,
                        end_line=min(i + chunk_size, len(lines)),
                        chunk_type="line",
                        language="python",
                    )
                )

        return chunks

    def _parse_other_language(
        self, file_path: str, content: str, language: str
    ) -> List[CodeChunk]:
        """Parse non-Python files with line-based chunking."""
        lines = content.split("\n")
        chunk_size = 50  # lines per chunk
        chunks = []

        for i in range(0, len(lines), chunk_size):
            chunk_lines = lines[i : i + chunk_size]
            chunks.append(
                CodeChunk(
                    text="\n".join(chunk_lines),
                    file_path=file_path,
                    start_line=i + 1,
                    end_line=min(i + chunk_size, len(lines)),
                    chunk_type="line",
                    language=language,
                )
            )

        return chunks

    def parse_file(self, file_path: str, content: str) -> List[CodeChunk]:
        """Parse file and return code chunks."""
        language = self._get_language(file_path)

        if language == "python":
            return self._parse_python(file_path, content)
        elif language != "unknown":
            return self._parse_other_language(file_path, content, language)
        else:
            # Unknown language: still create a simple chunk
            return [
                CodeChunk(
                    text=content,
                    file_path=file_path,
                    start_line=1,
                    end_line=len(content.split("\n")),
                    chunk_type="line",
                    language="unknown",
                )
            ]

