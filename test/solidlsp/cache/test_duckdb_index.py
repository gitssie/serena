"""
Comprehensive tests for DuckDB-based symbol index.

This module tests all core functionality of the DuckdbIndex class,
including lifecycle management, storage, querying, and cache operations.
"""

import os
import tempfile
import shutil
from pathlib import Path
from typing import List

import pytest

from solidlsp.index import DuckdbIndex
from solidlsp import ls_types
from serena.symbol import NamePathMatcher


@pytest.fixture
def temp_db_dir():
    """Create a temporary directory for the test database."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    # Cleanup after tests
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def duckdb_index(temp_db_dir):
    """Create a DuckdbIndex instance for testing."""
    db_path = os.path.join(temp_db_dir, "test_symbols.duckdb")
    index = DuckdbIndex(db_path=db_path, schema_name="test_schema")
    yield index
    # Ensure cleanup
    if index.is_started():
        index.stop()


@pytest.fixture
def sample_symbol_tree():
    """Create a sample symbol tree for testing."""
    # Root class symbol
    class_symbol = {
        'id': 'symbol_1',
        'name': 'TestClass',
        'kind': 5,  # Class kind
        'range': {
            'start': {'line': 10, 'character': 0},
            'end': {'line': 30, 'character': 0}
        },
        'selectionRange': {
            'start': {'line': 10, 'character': 6},
            'end': {'line': 10, 'character': 15}
        },
        'children': []
    }
    
    # Method 1
    method1_symbol = {
        'id': 'symbol_2',
        'name': 'method_one',
        'kind': 6,  # Method kind
        'range': {
            'start': {'line': 15, 'character': 4},
            'end': {'line': 20, 'character': 0}
        },
        'selectionRange': {
            'start': {'line': 15, 'character': 8},
            'end': {'line': 15, 'character': 18}
        },
        'children': [],
        'detail': 'def method_one(self, arg1: str) -> None'
    }
    
    # Method 2
    method2_symbol = {
        'id': 'symbol_3',
        'name': 'method_two',
        'kind': 6,  # Method kind
        'range': {
            'start': {'line': 22, 'character': 4},
            'end': {'line': 28, 'character': 0}
        },
        'selectionRange': {
            'start': {'line': 22, 'character': 8},
            'end': {'line': 22, 'character': 18}
        },
        'children': [],
        'detail': 'def method_two(self) -> int'
    }
    
    # Nested variable in method2
    var_symbol = {
        'id': 'symbol_4',
        'name': 'result',
        'kind': 13,  # Variable kind
        'range': {
            'start': {'line': 23, 'character': 8},
            'end': {'line': 23, 'character': 18}
        },
        'selectionRange': {
            'start': {'line': 23, 'character': 8},
            'end': {'line': 23, 'character': 14}
        },
        'children': []
    }
    
    method2_symbol['children'].append(var_symbol)
    class_symbol['children'].extend([method1_symbol, method2_symbol])
    
    return [class_symbol]


class TestDuckdbIndexLifecycle:
    """Test lifecycle management (start, stop, is_started)."""
    
    def test_initial_state(self, duckdb_index):
        """Test that index is not started initially."""
        assert not duckdb_index.is_started()
    
    def test_start_creates_database(self, duckdb_index):
        """Test that starting creates the database file."""
        duckdb_index.start()
        assert duckdb_index.is_started()
        assert os.path.exists(duckdb_index.db_path)
    
    def test_start_idempotent(self, duckdb_index):
        """Test that calling start multiple times is safe."""
        duckdb_index.start()
        duckdb_index.start()  # Should not raise
        assert duckdb_index.is_started()
    
    def test_stop_closes_connection(self, duckdb_index):
        """Test that stop closes the connection."""
        duckdb_index.start()
        duckdb_index.stop()
        assert not duckdb_index.is_started()
    
    def test_stop_idempotent(self, duckdb_index):
        """Test that calling stop multiple times is safe."""
        duckdb_index.start()
        duckdb_index.stop()
        duckdb_index.stop()  # Should not raise
        assert not duckdb_index.is_started()
    
    def test_restart(self, duckdb_index):
        """Test that index can be restarted."""
        duckdb_index.start()
        duckdb_index.stop()
        duckdb_index.start()
        assert duckdb_index.is_started()


class TestDuckdbIndexStorage:
    """Test storage operations (store_doc_symbols)."""
    
    def test_store_doc_symbols_requires_start(self, duckdb_index, sample_symbol_tree):
        """Test that storing symbols requires index to be started."""
        with pytest.raises(RuntimeError, match="Index not started"):
            duckdb_index.store_doc_symbols(
                "test/file.py",
                "hash123",
                sample_symbol_tree
            )
    
    def test_store_and_retrieve_symbols(self, duckdb_index, sample_symbol_tree):
        """Test basic store and retrieve workflow."""
        duckdb_index.start()
        
        # Store symbols
        duckdb_index.store_doc_symbols(
            "test/sample.py",
            "abc123def456",
            sample_symbol_tree
        )
        
        # Retrieve symbols
        result = duckdb_index.get_doc_symbols("test/sample.py", "abc123def456")
        
        assert result is not None
        assert result['content_hash'] == "abc123def456"
        assert 'file_symbol' in result
        file_symbol = result['file_symbol']
        assert file_symbol['name'] == "sample.py"
        assert len(file_symbol['children']) == 1
        
        # Verify class symbol
        class_sym = file_symbol['children'][0]
        assert class_sym['name'] == 'TestClass'
        assert class_sym['kind'] == 5
        assert len(class_sym['children']) == 2
    
    def test_store_overwrites_old_data(self, duckdb_index):
        """Test that storing new symbols overwrites old data."""
        duckdb_index.start()
        
        # Store first version
        old_symbols = [{
            'id': 'old_1',
            'name': 'OldClass',
            'kind': 5,
            'range': {
                'start': {'line': 0, 'character': 0},
                'end': {'line': 5, 'character': 0}
            },
            'selectionRange': {
                'start': {'line': 0, 'character': 0},
                'end': {'line': 0, 'character': 8}
            },
            'children': []
        }]
        
        duckdb_index.store_doc_symbols("test/file.py", "hash1", old_symbols)
        
        # Store new version
        new_symbols = [{
            'id': 'new_1',
            'name': 'NewClass',
            'kind': 5,
            'range': {
                'start': {'line': 0, 'character': 0},
                'end': {'line': 10, 'character': 0}
            },
            'selectionRange': {
                'start': {'line': 0, 'character': 0},
                'end': {'line': 0, 'character': 8}
            },
            'children': []
        }]
        
        duckdb_index.store_doc_symbols("test/file.py", "hash2", new_symbols)
        
        # Verify only new symbols exist
        result = duckdb_index.get_doc_symbols("test/file.py", "hash2")
        assert result is not None
        file_symbol = result['file_symbol']
        assert len(file_symbol['children']) == 1
        assert file_symbol['children'][0]['name'] == 'NewClass'
    
    def test_store_multiple_documents(self, duckdb_index):
        """Test storing symbols from multiple documents."""
        duckdb_index.start()
        
        # Store first document
        doc1_symbols = [{
            'id': 'doc1_symbol',
            'name': 'ClassA',
            'kind': 5,
            'range': {
                'start': {'line': 0, 'character': 0},
                'end': {'line': 5, 'character': 0}
            },
            'selectionRange': {
                'start': {'line': 0, 'character': 0},
                'end': {'line': 0, 'character': 6}
            },
            'children': []
        }]
        
        duckdb_index.store_doc_symbols("src/module_a.py", "hash_a", doc1_symbols)
        
        # Store second document
        doc2_symbols = [{
            'id': 'doc2_symbol',
            'name': 'ClassB',
            'kind': 5,
            'range': {
                'start': {'line': 0, 'character': 0},
                'end': {'line': 5, 'character': 0}
            },
            'selectionRange': {
                'start': {'line': 0, 'character': 0},
                'end': {'line': 0, 'character': 6}
            },
            'children': []
        }]
        
        duckdb_index.store_doc_symbols("src/module_b.py", "hash_b", doc2_symbols)
        
        # Verify both documents exist
        result_a = duckdb_index.get_doc_symbols("src/module_a.py", "hash_a")
        result_b = duckdb_index.get_doc_symbols("src/module_b.py", "hash_b")
        
        assert result_a is not None
        assert result_b is not None
        assert result_a['file_symbol']['children'][0]['name'] == 'ClassA'
        assert result_b['file_symbol']['children'][0]['name'] == 'ClassB'


class TestDuckdbIndexRetrieval:
    """Test retrieval operations (get_doc_symbols, is_doc_cached)."""
    
    def test_get_doc_symbols_not_found(self, duckdb_index):
        """Test retrieving non-existent document returns None."""
        duckdb_index.start()
        result = duckdb_index.get_doc_symbols("nonexistent.py", "hash")
        assert result is None
    
    def test_get_doc_symbols_hash_mismatch(self, duckdb_index):
        """Test that hash mismatch returns None."""
        duckdb_index.start()
        
        symbols = [{
            'id': 'sym1',
            'name': 'MyClass',
            'kind': 5,
            'range': {
                'start': {'line': 0, 'character': 0},
                'end': {'line': 5, 'character': 0}
            },
            'selectionRange': {
                'start': {'line': 0, 'character': 0},
                'end': {'line': 0, 'character': 7}
            },
            'children': []
        }]
        
        duckdb_index.store_doc_symbols("test.py", "original_hash", symbols)
        
        # Try to retrieve with different hash
        result = duckdb_index.get_doc_symbols("test.py", "different_hash")
        assert result is None
    
    def test_is_doc_cached_true(self, duckdb_index):
        """Test is_doc_cached returns True for cached document."""
        duckdb_index.start()
        
        symbols = [{
            'id': 'sym1',
            'name': 'MyClass',
            'kind': 5,
            'range': {
                'start': {'line': 0, 'character': 0},
                'end': {'line': 5, 'character': 0}
            },
            'selectionRange': {
                'start': {'line': 0, 'character': 0},
                'end': {'line': 0, 'character': 7}
            },
            'children': []
        }]
        
        duckdb_index.store_doc_symbols("cached.py", "hash123", symbols)
        assert duckdb_index.is_doc_cached("cached.py", "hash123")
    
    def test_is_doc_cached_false_not_found(self, duckdb_index):
        """Test is_doc_cached returns False for non-existent document."""
        duckdb_index.start()
        assert not duckdb_index.is_doc_cached("notfound.py", "hash")
    
    def test_is_doc_cached_false_hash_mismatch(self, duckdb_index):
        """Test is_doc_cached returns False for hash mismatch."""
        duckdb_index.start()
        
        symbols = [{
            'id': 'sym1',
            'name': 'MyClass',
            'kind': 5,
            'range': {
                'start': {'line': 0, 'character': 0},
                'end': {'line': 5, 'character': 0}
            },
            'selectionRange': {
                'start': {'line': 0, 'character': 0},
                'end': {'line': 0, 'character': 7}
            },
            'children': []
        }]
        
        duckdb_index.store_doc_symbols("test.py", "hash1", symbols)
        assert not duckdb_index.is_doc_cached("test.py", "hash2")


class TestDuckdbIndexQuery:
    """Test query operations (query_symbols)."""
    
    @staticmethod
    def _pattern_to_regex(pattern: str) -> str:
        """Helper to convert pattern to regex for testing."""
        from serena.symbol import NamePathMatcher
        is_absolute = pattern.startswith('/')
        return NamePathMatcher._convert_name_path_to_regex(pattern, is_absolute)
    
    def test_query_all_symbols(self, duckdb_index, sample_symbol_tree):
        """Test querying all symbols with empty pattern."""
        duckdb_index.start()
        duckdb_index.store_doc_symbols("module.py", "hash1", sample_symbol_tree)
        
        results = duckdb_index.query_symbols(
            name_path_regex=self._pattern_to_regex("")
        )
        
        assert len(results) == 1
        assert results[0]['name'] == "module.py"
    
    def test_query_by_exact_name(self, duckdb_index, sample_symbol_tree):
        """Test querying by exact symbol name."""
        duckdb_index.start()
        duckdb_index.store_doc_symbols("module.py", "hash1", sample_symbol_tree)
        
        results = duckdb_index.query_symbols(
            name_path_regex=self._pattern_to_regex("TestClass")
        )
        
        assert len(results) == 1
        file_sym = results[0]
        assert len(file_sym['children']) == 1
        assert file_sym['children'][0]['name'] == 'TestClass'
    
    def test_query_by_name_path(self, duckdb_index, sample_symbol_tree):
        """Test querying by name path (parent/child)."""
        duckdb_index.start()
        duckdb_index.store_doc_symbols("module.py", "hash1", sample_symbol_tree)
        
        results = duckdb_index.query_symbols(
            name_path_regex=self._pattern_to_regex("TestClass/method_one")
        )
        
        assert len(results) == 1
        file_sym = results[0]
        # Should filter to only the class containing method_one
        class_sym = file_sym['children'][0]
        assert class_sym['name'] == 'TestClass'
        # Should have filtered children
        assert any(c['name'] == 'method_one' for c in class_sym['children'])
    
    def test_query_with_substring_matching(self, duckdb_index, sample_symbol_tree):
        """Test substring matching in queries (default grep-like behavior)."""
        duckdb_index.start()
        duckdb_index.store_doc_symbols("module.py", "hash1", sample_symbol_tree)
        
        results = duckdb_index.query_symbols(
            name_path_regex=self._pattern_to_regex("method")
        )
        
        assert len(results) == 1
        file_sym = results[0]
        class_sym = file_sym['children'][0]
        # Should find both methods
        method_names = [c['name'] for c in class_sym['children']]
        assert 'method_one' in method_names
        assert 'method_two' in method_names
    
    def test_query_with_include_kinds(self, duckdb_index, sample_symbol_tree):
        """Test filtering by included symbol kinds."""
        duckdb_index.start()
        duckdb_index.store_doc_symbols("module.py", "hash1", sample_symbol_tree)
        
        # Query only methods (kind 6)
        results = duckdb_index.query_symbols(
            name_path_regex=self._pattern_to_regex(""),
            include_kinds=[6]  # Method kind
        )
        
        assert len(results) == 1
        file_sym = results[0]
        class_sym = file_sym['children'][0]
        # Should only have methods, not the class itself at top level
        # But class should be present as container
        assert class_sym['name'] == 'TestClass'
        method_names = [c['name'] for c in class_sym['children'] if c['kind'] == 6]
        assert len(method_names) == 2
    
    def test_query_with_exclude_kinds(self, duckdb_index, sample_symbol_tree):
        """Test filtering by excluded symbol kinds."""
        duckdb_index.start()
        duckdb_index.store_doc_symbols("module.py", "hash1", sample_symbol_tree)
        
        # Exclude variables (kind 13)
        results = duckdb_index.query_symbols(
            name_path_regex=self._pattern_to_regex(""),
            exclude_kinds=[13]  # Variable kind
        )
        
        assert len(results) == 1
        file_sym = results[0]
        class_sym = file_sym['children'][0]
        # Check that 'result' variable is not present in method_two
        for method in class_sym['children']:
            if method['name'] == 'method_two':
                # Variables should be filtered out
                var_count = sum(1 for c in method.get('children', []) if c['kind'] == 13)
                assert var_count == 0
    
    def test_query_within_relative_path(self, duckdb_index, sample_symbol_tree):
        """Test querying within specific directory path."""
        duckdb_index.start()
        
        # Store in different directories
        duckdb_index.store_doc_symbols("src/core/module.py", "hash1", sample_symbol_tree)
        duckdb_index.store_doc_symbols("tests/test_module.py", "hash2", sample_symbol_tree)
        
        # Query only in src/core - use direct regex pattern
        results = duckdb_index.query_symbols(
            name_path_regex=self._pattern_to_regex(""),
            relative_path_regex="src/core"  # Direct regex pattern for path matching
        )
        
        assert len(results) == 1
        assert "core" in results[0]['relativePath']


class TestDuckdbIndexCacheManagement:
    """Test cache management operations (invalidate, clear_all, stats)."""
    
    def test_invalidate_doc(self, duckdb_index, sample_symbol_tree):
        """Test invalidating a specific document."""
        duckdb_index.start()
        
        duckdb_index.store_doc_symbols("file1.py", "hash1", sample_symbol_tree)
        duckdb_index.store_doc_symbols("file2.py", "hash2", sample_symbol_tree)
        
        # Invalidate file1
        duckdb_index.invalidate_doc("file1.py")
        
        # file1 should not be cached anymore
        assert not duckdb_index.is_doc_cached("file1.py", "hash1")
        
        # file2 should still be cached
        assert duckdb_index.is_doc_cached("file2.py", "hash2")
    
    def test_clear_all(self, duckdb_index, sample_symbol_tree):
        """Test clearing all cached data."""
        duckdb_index.start()
        
        duckdb_index.store_doc_symbols("file1.py", "hash1", sample_symbol_tree)
        duckdb_index.store_doc_symbols("file2.py", "hash2", sample_symbol_tree)
        
        # Clear all
        duckdb_index.clear_all()
        
        # Both files should not be cached anymore
        assert not duckdb_index.is_doc_cached("file1.py", "hash1")
        assert not duckdb_index.is_doc_cached("file2.py", "hash2")
    
    def test_get_cache_stats(self, duckdb_index, sample_symbol_tree):
        """Test getting cache statistics."""
        duckdb_index.start()
        
        # Initially empty
        stats = duckdb_index.get_cache_stats()
        assert stats['total_docs'] == 0
        assert stats['backend'] == 'duckdb'
        
        # Store some documents
        duckdb_index.store_doc_symbols("file1.py", "hash1", sample_symbol_tree)
        duckdb_index.store_doc_symbols("file2.py", "hash2", sample_symbol_tree)
        
        # Check updated stats
        stats = duckdb_index.get_cache_stats()
        assert stats['total_docs'] == 2
        assert stats['total_symbols'] > 0
    
    def test_get_all_cached_docs(self, duckdb_index, sample_symbol_tree):
        """Test getting list of all cached document paths."""
        duckdb_index.start()
        
        # Initially empty
        docs = duckdb_index.get_all_cached_docs()
        assert len(docs) == 0
        
        # Store documents
        duckdb_index.store_doc_symbols("src/file1.py", "hash1", sample_symbol_tree)
        duckdb_index.store_doc_symbols("tests/file2.py", "hash2", sample_symbol_tree)
        
        # Check docs list
        docs = duckdb_index.get_all_cached_docs()
        assert len(docs) == 2
        assert "src/file1.py" in docs
        assert "tests/file2.py" in docs


class TestDuckdbIndexEdgeCases:
    """Test edge cases and error handling."""
    
    def test_empty_symbol_tree(self, duckdb_index):
        """Test storing document with no symbols."""
        duckdb_index.start()
        
        duckdb_index.store_doc_symbols("empty.py", "hash", [])
        
        result = duckdb_index.get_doc_symbols("empty.py", "hash")
        assert result is not None
        assert len(result['file_symbol']['children']) == 0
    
    def test_deeply_nested_symbols(self, duckdb_index):
        """Test handling deeply nested symbol hierarchies."""
        duckdb_index.start()
        
        # Create deeply nested structure
        def create_nested(depth, name_prefix):
            if depth == 0:
                return []
            
            symbol = {
                'id': f'symbol_{name_prefix}_{depth}',
                'name': f'Level{depth}',
                'kind': 5,
                'range': {
                    'start': {'line': depth, 'character': 0},
                    'end': {'line': depth + 1, 'character': 0}
                },
                'selectionRange': {
                    'start': {'line': depth, 'character': 0},
                    'end': {'line': depth, 'character': 6}
                },
                'children': create_nested(depth - 1, name_prefix)
            }
            return [symbol]
        
        deep_symbols = create_nested(10, "test")
        
        duckdb_index.store_doc_symbols("deep.py", "hash", deep_symbols)
        
        result = duckdb_index.get_doc_symbols("deep.py", "hash")
        assert result is not None
        
        # Verify nesting
        current = result['file_symbol']['children'][0]
        depth = 0
        while current:
            depth += 1
            if current.get('children'):
                current = current['children'][0]
            else:
                break
        
        assert depth == 10
    
    def test_special_characters_in_path(self, duckdb_index, sample_symbol_tree):
        """Test handling paths with special characters."""
        duckdb_index.start()
        
        special_path = "src/special-file_v1.2.py"
        duckdb_index.store_doc_symbols(special_path, "hash", sample_symbol_tree)
        
        result = duckdb_index.get_doc_symbols(special_path, "hash")
        assert result is not None
        assert result['file_symbol']['relativePath'] == special_path
    
    def test_unicode_in_symbol_names(self, duckdb_index):
        """Test handling unicode characters in symbol names."""
        duckdb_index.start()
        
        unicode_symbol = [{
            'id': 'unicode_1',
            'name': '测试类',  # Chinese characters
            'kind': 5,
            'range': {
                'start': {'line': 0, 'character': 0},
                'end': {'line': 5, 'character': 0}
            },
            'selectionRange': {
                'start': {'line': 0, 'character': 0},
                'end': {'line': 0, 'character': 3}
            },
            'children': []
        }]
        
        duckdb_index.store_doc_symbols("unicode.py", "hash", unicode_symbol)
        
        result = duckdb_index.get_doc_symbols("unicode.py", "hash")
        assert result is not None
        assert result['file_symbol']['children'][0]['name'] == '测试类'
    
    def test_concurrent_operations(self, duckdb_index, sample_symbol_tree):
        """Test thread safety of index operations."""
        import threading
        
        duckdb_index.start()
        
        errors = []
        
        def store_symbols(path_num):
            try:
                path = f"file{path_num}.py"
                duckdb_index.store_doc_symbols(path, f"hash{path_num}", sample_symbol_tree)
            except Exception as e:
                errors.append(e)
        
        # Create multiple threads
        threads = [threading.Thread(target=store_symbols, args=(i,)) for i in range(10)]
        
        # Start all threads
        for t in threads:
            t.start()
        
        # Wait for completion
        for t in threads:
            t.join()
        
        # Check no errors occurred
        assert len(errors) == 0
        
        # Verify all files were stored
        stats = duckdb_index.get_cache_stats()
        assert stats['total_docs'] == 10


class TestDuckdbIndexSchemaIsolation:
    """Test schema isolation between different index instances."""
    
    def test_different_schemas_isolated(self, temp_db_dir, sample_symbol_tree):
        """Test that different schemas don't interfere with each other."""
        db_path = os.path.join(temp_db_dir, "shared.duckdb")
        
        # Create two indexes with different schemas
        index1 = DuckdbIndex(db_path, "schema1")
        index2 = DuckdbIndex(db_path, "schema2")
        
        index1.start()
        index2.start()
        
        try:
            # Store different data in each schema
            index1.store_doc_symbols("file1.py", "hash1", sample_symbol_tree)
            
            modified_symbols = sample_symbol_tree.copy()
            modified_symbols[0]['name'] = 'DifferentClass'
            index2.store_doc_symbols("file2.py", "hash2", modified_symbols)
            
            # Verify isolation
            result1 = index1.get_doc_symbols("file1.py", "hash1")
            result2 = index2.get_doc_symbols("file2.py", "hash2")
            
            assert result1 is not None
            assert result2 is not None
            
            # schema1 should not see schema2's data
            assert index1.get_doc_symbols("file2.py", "hash2") is None
            # schema2 should not see schema1's data
            assert index2.get_doc_symbols("file1.py", "hash1") is None
            
        finally:
            index1.stop()
            index2.stop()
