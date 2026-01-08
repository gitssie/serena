"""
Test the create_symbol_index factory method with different backends.
"""

import os
import tempfile
import shutil
from pathlib import Path

import pytest

from solidlsp.index import create_symbol_index


@pytest.fixture
def temp_base_dir():
    """Create a temporary base directory."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


class TestSymbolIndexFactory:
    """Test the factory method for creating symbol indexes."""
    
    def test_kuzu_creates_language_specific_directory(self, temp_base_dir):
        """Test that Kuzu backend creates separate directory for each language."""
        # Create indexes for two languages
        index_python = create_symbol_index(
            base_dir=temp_base_dir,
            language_id="python",
            backend="kuzu"
        )
        
        index_javascript = create_symbol_index(
            base_dir=temp_base_dir,
            language_id="javascript",
            backend="kuzu"
        )
        
        try:
            index_python.start()
            index_javascript.start()
            
            # Check that separate directories were created
            python_dir = Path(temp_base_dir) / "python"
            javascript_dir = Path(temp_base_dir) / "javascript"
            
            assert python_dir.exists()
            assert javascript_dir.exists()
            
            # Each should have its own database
            assert (python_dir / "symbol_db").exists()
            assert (javascript_dir / "symbol_db").exists()
            
        finally:
            index_python.stop()
            index_javascript.stop()
    
    def test_duckdb_creates_shared_file_with_schemas(self, temp_base_dir):
        """Test that DuckDB backend uses shared file with schema isolation."""
        # Create indexes for two languages
        index_python = create_symbol_index(
            base_dir=temp_base_dir,
            language_id="python",
            backend="duckdb"
        )
        
        index_javascript = create_symbol_index(
            base_dir=temp_base_dir,
            language_id="javascript",
            backend="duckdb"
        )
        
        try:
            index_python.start()
            index_javascript.start()
            
            # Check that only one database file exists
            db_file = Path(temp_base_dir) / "symbol_db.duckdb"
            assert db_file.exists()
            
            # Verify both schemas exist by checking cache stats
            python_stats = index_python.get_cache_stats()
            javascript_stats = index_javascript.get_cache_stats()
            
            assert python_stats['backend'] == 'duckdb'
            assert javascript_stats['backend'] == 'duckdb'
            
        finally:
            index_python.stop()
            index_javascript.stop()
    
    def test_kuzu_and_duckdb_can_coexist(self, temp_base_dir):
        """Test that both backends can be used simultaneously."""
        index_kuzu = create_symbol_index(
            base_dir=temp_base_dir,
            language_id="python",
            backend="kuzu"
        )
        
        index_duckdb = create_symbol_index(
            base_dir=temp_base_dir,
            language_id="python",
            backend="duckdb"
        )
        
        try:
            index_kuzu.start()
            index_duckdb.start()
            
            # Both should start successfully
            assert index_kuzu.is_started()
            assert index_duckdb.is_started()
            
            # Check file structure
            python_dir = Path(temp_base_dir) / "python"
            db_file = Path(temp_base_dir) / "symbol_db.duckdb"
            
            assert python_dir.exists()  # Kuzu directory
            assert db_file.exists()     # DuckDB file
            
        finally:
            index_kuzu.stop()
            index_duckdb.stop()
