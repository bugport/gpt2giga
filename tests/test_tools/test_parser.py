"""Unit tests for CodeParser."""

import tempfile
from pathlib import Path

import pytest

from gpt2giga.tools.parser import CodeParser, CodeChunk


class TestCodeParser:
    """Test cases for CodeParser."""

    def test_parse_python_function(self):
        """Test parsing Python function."""
        content = '''def hello_world():
    """A simple greeting function."""
    print("Hello, World!")
    return "greeting"
'''
        parser = CodeParser()
        chunks = parser.parse_file("/test/file.py", content)

        assert len(chunks) > 0
        # Should extract function
        func_chunks = [c for c in chunks if c.chunk_type == "function"]
        assert len(func_chunks) > 0
        assert "hello_world" in func_chunks[0].text

    def test_parse_python_class(self):
        """Test parsing Python class."""
        content = '''class MyClass:
    """A test class."""
    
    def __init__(self):
        self.value = 42
    
    def get_value(self):
        return self.value
'''
        parser = CodeParser()
        chunks = parser.parse_file("/test/file.py", content)

        assert len(chunks) > 0
        # Should extract class
        class_chunks = [c for c in chunks if c.chunk_type == "class"]
        assert len(class_chunks) > 0
        assert "MyClass" in class_chunks[0].text

    def test_parse_javascript(self):
        """Test parsing JavaScript file."""
        content = '''function testFunction() {
    console.log("test");
    return true;
}

const arrowFunc = () => {
    return "arrow";
};
'''
        parser = CodeParser()
        chunks = parser.parse_file("/test/file.js", content)

        assert len(chunks) > 0
        assert chunks[0].language == "javascript"
        assert chunks[0].chunk_type == "line"

    def test_parse_typescript(self):
        """Test parsing TypeScript file."""
        content = '''interface Test {
    name: string;
    value: number;
}

function process(data: Test): void {
    console.log(data.name);
}
'''
        parser = CodeParser()
        chunks = parser.parse_file("/test/file.ts", content)

        assert len(chunks) > 0
        assert chunks[0].language == "typescript"

    def test_parse_cpp(self):
        """Test parsing C++ file."""
        content = '''#include <iostream>

class MyClass {
public:
    void method() {
        std::cout << "Hello" << std::endl;
    }
};
'''
        parser = CodeParser()
        chunks = parser.parse_file("/test/file.cpp", content)

        assert len(chunks) > 0
        assert chunks[0].language == "cpp"

    def test_parse_sql(self):
        """Test parsing SQL file."""
        content = '''SELECT * FROM users WHERE id = 1;

INSERT INTO users (name, email) VALUES ('John', 'john@example.com');

UPDATE users SET name = 'Jane' WHERE id = 2;
'''
        parser = CodeParser()
        chunks = parser.parse_file("/test/query.sql", content)

        assert len(chunks) > 0
        assert chunks[0].language == "sql"

    def test_parse_python_syntax_error(self):
        """Test that syntax errors fallback to line-based chunking."""
        content = '''def broken
    invalid syntax here
'''
        parser = CodeParser()
        chunks = parser.parse_file("/test/broken.py", content)

        # Should still return chunks (fallback to line-based)
        assert len(chunks) > 0
        assert chunks[0].chunk_type == "line"

    def test_chunk_metadata(self):
        """Test that chunks have correct metadata."""
        content = "def test():\n    pass\n"
        parser = CodeParser()
        chunks = parser.parse_file("/test/file.py", content)

        assert len(chunks) > 0
        chunk = chunks[0]
        assert chunk.file_path == "/test/file.py"
        assert chunk.start_line >= 1
        assert chunk.end_line >= chunk.start_line
        assert chunk.language == "python"
        assert chunk.chunk_type in ["function", "class", "line"]

    def test_parse_unknown_language(self):
        """Test parsing file with unknown language."""
        content = "some random content\nmore content\n"
        parser = CodeParser()
        chunks = parser.parse_file("/test/file.unknown", content)

        # Should still create chunks
        assert len(chunks) > 0
        assert chunks[0].language == "unknown"

    def test_parse_empty_file(self):
        """Test parsing empty file."""
        parser = CodeParser()
        chunks = parser.parse_file("/test/empty.py", "")

        # Should handle empty files gracefully
        assert isinstance(chunks, list)
        # May return empty list or single chunk

