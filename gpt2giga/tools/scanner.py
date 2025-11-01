"""Codebase scanner for discovering and filtering code files."""

import hashlib
import os
from pathlib import Path
from typing import List, Optional
import fnmatch


class FileInfo:
    """Information about a code file."""

    def __init__(
        self,
        path: str,
        file_hash: str,
        size: int,
        extension: str,
    ):
        self.path = path
        self.hash = file_hash
        self.size = size
        self.extension = extension


class CodebaseScanner:
    """Scans directories and filters code files."""

    def __init__(
        self,
        include_patterns: Optional[List[str]] = None,
        exclude_patterns: Optional[List[str]] = None,
        max_file_size: int = 10 * 1024 * 1024,  # 10MB default
    ):
        self.include_patterns = include_patterns or []
        self.exclude_patterns = exclude_patterns or []
        self.max_file_size = max_file_size
        self._gitignore_patterns = []

    def _load_gitignore(self, root_path: Path) -> List[str]:
        """Load patterns from .gitignore and .codebaseignore files."""
        patterns = []
        for ignore_file in [".gitignore", ".codebaseignore"]:
            ignore_path = root_path / ignore_file
            if ignore_path.exists():
                try:
                    with ignore_path.open("r", encoding="utf-8") as f:
                        for line in f:
                            line = line.strip()
                            if line and not line.startswith("#"):
                                patterns.append(line)
                except Exception:
                    pass
        return patterns

    def _matches_pattern(self, path: str, patterns: List[str]) -> bool:
        """Check if path matches any pattern."""
        # Convert to relative path for matching
        rel_path = path
        for pattern in patterns:
            if fnmatch.fnmatch(path, pattern) or fnmatch.fnmatch(
                os.path.basename(path), pattern
            ):
                return True
            # Also check if any parent directory matches
            path_parts = Path(path).parts
            for i in range(len(path_parts)):
                subpath = "/".join(path_parts[i:])
                if fnmatch.fnmatch(subpath, pattern):
                    return True
        return False

    def _calculate_file_hash(self, file_path: Path) -> str:
        """Calculate SHA256 hash of file content."""
        sha256 = hashlib.sha256()
        try:
            with file_path.open("rb") as f:
                # Read in chunks for large files
                while chunk := f.read(8192):
                    sha256.update(chunk)
            return sha256.hexdigest()
        except Exception:
            return ""

    def _should_include_file(
        self, file_path: Path, root_path: Path, gitignore_patterns: List[str]
    ) -> bool:
        """Check if file should be included based on patterns."""
        # Check exclude patterns first
        if self.exclude_patterns:
            if self._matches_pattern(str(file_path), self.exclude_patterns):
                return False

        # Check gitignore patterns
        if gitignore_patterns:
            rel_path = str(file_path.relative_to(root_path))
            if self._matches_pattern(rel_path, gitignore_patterns):
                return False

        # Check file size
        try:
            size = file_path.stat().st_size
            if size > self.max_file_size:
                return False
        except Exception:
            return False

        # Check include patterns
        if self.include_patterns:
            ext = file_path.suffix.lower()
            matches = any(
                fnmatch.fnmatch(ext, pattern)
                or fnmatch.fnmatch(file_path.name, pattern)
                for pattern in self.include_patterns
            )
            if not matches:
                return False

        return True

    def scan_directory(self, path: str) -> List[FileInfo]:
        """Scan directory recursively and return file information."""
        root_path = Path(path).resolve()
        if not root_path.exists() or not root_path.is_dir():
            return []

        gitignore_patterns = self._load_gitignore(root_path)
        files = []

        for file_path in root_path.rglob("*"):
            if not file_path.is_file():
                continue

            if self._should_include_file(file_path, root_path, gitignore_patterns):
                try:
                    file_hash = self._calculate_file_hash(file_path)
                    size = file_path.stat().st_size
                    extension = file_path.suffix.lower()
                    files.append(
                        FileInfo(
                            path=str(file_path),
                            file_hash=file_hash,
                            size=size,
                            extension=extension,
                        )
                    )
                except Exception:
                    # Skip files we can't process
                    continue

        return files

