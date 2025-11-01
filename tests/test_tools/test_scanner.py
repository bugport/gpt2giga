"""Unit tests for CodebaseScanner."""

import os
import tempfile
from pathlib import Path

import pytest

from gpt2giga.tools.scanner import CodebaseScanner, FileInfo


class TestCodebaseScanner:
    """Test cases for CodebaseScanner."""

    def test_scan_directory_basic(self, tmp_path):
        """Test basic directory scanning."""
        # Create test files
        (tmp_path / "file1.py").write_text("def hello(): pass\n")
        (tmp_path / "file2.js").write_text("function test() {}\n")
        (tmp_path / "subdir").mkdir()
        (tmp_path / "subdir" / "file3.py").write_text("class Test:\n    pass\n")

        scanner = CodebaseScanner(include_patterns=[".py", ".js"])
        files = scanner.scan_directory(str(tmp_path))

        assert len(files) == 3
        paths = {f.path for f in files}
        assert str(tmp_path / "file1.py") in paths
        assert str(tmp_path / "file2.js") in paths
        assert str(tmp_path / "subdir" / "file3.py") in paths

    def test_scan_directory_with_exclude_patterns(self, tmp_path):
        """Test scanning with exclude patterns."""
        # Create test files
        (tmp_path / "file1.py").write_text("def hello(): pass\n")
        (tmp_path / "node_modules").mkdir()
        (tmp_path / "node_modules" / "dep.js").write_text("export {};\n")
        (tmp_path / "venv").mkdir()
        (tmp_path / "venv" / "lib.py").write_text("import sys\n")

        scanner = CodebaseScanner(
            include_patterns=[".py", ".js"],
            exclude_patterns=["node_modules", "venv"],
        )
        files = scanner.scan_directory(str(tmp_path))

        assert len(files) == 1
        assert files[0].path == str(tmp_path / "file1.py")

    def test_scan_directory_with_gitignore(self, tmp_path):
        """Test scanning with .gitignore support."""
        # Create .gitignore
        (tmp_path / ".gitignore").write_text("*.pyc\n__pycache__/\n")

        # Create test files
        (tmp_path / "file1.py").write_text("def hello(): pass\n")
        (tmp_path / "file1.pyc").write_bytes(b"\x00\x01")
        (tmp_path / "__pycache__").mkdir()
        (tmp_path / "__pycache__" / "file1.pyc").write_bytes(b"\x00\x01")

        scanner = CodebaseScanner(include_patterns=[".py"])
        files = scanner.scan_directory(str(tmp_path))

        # Should only find file1.py, not .pyc files
        assert len(files) == 1
        assert files[0].path == str(tmp_path / "file1.py")
        assert files[0].extension == ".py"

    def test_scan_directory_file_hash(self, tmp_path):
        """Test that file hashes are calculated correctly."""
        content = "def hello():\n    return 'world'\n"
        (tmp_path / "test.py").write_text(content)

        scanner = CodebaseScanner()
        files = scanner.scan_directory(str(tmp_path))

        assert len(files) == 1
        assert files[0].hash != ""
        assert len(files[0].hash) == 64  # SHA256 hex length

    def test_scan_directory_size_limit(self, tmp_path):
        """Test that files exceeding size limit are excluded."""
        # Create a large file (simulate)
        large_content = "x" * (11 * 1024 * 1024)  # 11MB
        (tmp_path / "large.py").write_text(large_content)

        scanner = CodebaseScanner(
            include_patterns=[".py"], max_file_size=10 * 1024 * 1024
        )
        files = scanner.scan_directory(str(tmp_path))

        # Large file should be excluded
        assert len(files) == 0

    def test_scan_directory_nonexistent_path(self):
        """Test scanning nonexistent path."""
        scanner = CodebaseScanner()
        files = scanner.scan_directory("/nonexistent/path/12345")

        assert len(files) == 0

    def test_scan_directory_empty(self, tmp_path):
        """Test scanning empty directory."""
        scanner = CodebaseScanner()
        files = scanner.scan_directory(str(tmp_path))

        assert len(files) == 0

    def test_scan_directory_cpp_and_sql(self, tmp_path):
        """Test scanning C++ and SQL files (as per requirements)."""
        (tmp_path / "main.cpp").write_text("#include <iostream>\n")
        (tmp_path / "header.h").write_text("#ifndef HEADER_H\n")
        (tmp_path / "header.hpp").write_text("#pragma once\n")
        (tmp_path / "query.sql").write_text("SELECT * FROM users;\n")
        (tmp_path / "file.py").write_text("def test(): pass\n")

        scanner = CodebaseScanner(
            include_patterns=[".cpp", ".h", ".hpp", ".sql", ".py"]
        )
        files = scanner.scan_directory(str(tmp_path))

        assert len(files) == 5
        extensions = {f.extension for f in files}
        assert ".cpp" in extensions
        assert ".h" in extensions
        assert ".hpp" in extensions
        assert ".sql" in extensions
        assert ".py" in extensions

    def test_file_info_attributes(self, tmp_path):
        """Test FileInfo attributes are set correctly."""
        (tmp_path / "test.py").write_text("print('hello')\n")

        scanner = CodebaseScanner()
        files = scanner.scan_directory(str(tmp_path))

        assert len(files) == 1
        file_info = files[0]
        assert file_info.path == str(tmp_path / "test.py")
        assert file_info.extension == ".py"
        assert file_info.size > 0
        assert file_info.hash != ""

