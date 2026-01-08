"""
Tests for the search_symbols method with Kùzu cache integration.

These tests validate that search_symbols correctly queries and returns symbols
using the Kùzu graph database cache, with proper parent references and path matching.
"""

import os
import pytest

from solidlsp import SolidLanguageServer
from solidlsp.ls_config import Language
from solidlsp.ls_types import SymbolKind
from serena.symbol import LanguageServerSymbolRetriever
from test.conftest import language_tests_enabled, get_repo_path

pytestmark = [pytest.mark.java, pytest.mark.skipif(not language_tests_enabled(Language.JAVA), reason="Java tests disabled")]


class TestJavaSearchSymbols:
    """Test search_symbols method with various pattern matching scenarios."""

    @staticmethod
    def _index_project(project: str, log_level: str = "DEBUG", timeout: float = 10.0) -> None:
        """Index the given project path so symbol caches are populated for tests.

        This mirrors the minimal behavior from the CLI _index_project: it loads
        the project, creates a language server manager and requests document
        symbols for all source files, saving caches periodically.
        """
        import logging
        import collections
        from sensai.util.logging import configure
        from serena.config.serena_config import SerenaConfig
        from serena.project import Project
        from tqdm import tqdm

        lvl = logging.getLevelNamesMapping()[log_level.upper()]
        configure(level=lvl)
        serena_config = SerenaConfig.from_config_file()
        proj = Project.load(os.path.abspath(project))
        ls_mgr = proj.create_language_server_manager(
            log_level=lvl, ls_timeout=timeout, ls_specific_settings=serena_config.ls_specific_settings
        )
        try:
            appender = proj.init_index_cache(clear_data=True,batch_mode=True)
            files = proj.gather_source_files()
            collected_exceptions: list[Exception] = []
            files_failed = []
            language_file_counts: dict = collections.defaultdict(lambda: 0)
            for i, f in enumerate(tqdm(files, desc="Indexing")):
                try:
                    ls = ls_mgr.get_language_server(f)
                    ls.build_index(f, appender=appender.get(ls.language))
                    language_file_counts[ls.language] += 1
                except Exception as e:
                    collected_exceptions.append(e)
                    files_failed.append(f)
                if (i + 1) % 10 == 0:
                    ls_mgr.save_all_caches()
            
            for app in appender.values():
                app.commit()
            
            if len(files_failed) > 0:
                # Don't raise in tests; just log
                import logging as _logging
                _logging.getLogger(__name__).warning("Indexing failed for %d files", len(files_failed))
            
            # Wait for LSP diagnostics to be processed
            import time
            logging.info("Waiting 5 seconds for LSP diagnostics notifications...")
        finally:
            ls_mgr.stop_all()

    def test_index_project(self) -> None:
        """Test the _index_project helper method."""
        repo_path = get_repo_path(Language.JAVA)
        self._index_project(str(repo_path))

    @pytest.mark.parametrize("language_server", [Language.JAVA], indirect=True)
    def test_search_simple_name(self, language_server: SolidLanguageServer) -> None:
        """Test searching for symbols by simple name."""
        # Request document symbols and find methods named "printHello" within the file
        file_path = os.path.join("src", "main", "java", "test_repo", "Utils.java")
        doc_symbols = language_server.request_document_symbols(file_path)
        found = [s for s in doc_symbols.iter_symbols() if s.get("name") == "printHello"]

        assert len(found) > 0, "Should find at least one printHello method"

        # Verify symbol has location and parent reference
        for sym in found:
            assert "location" in sym, "Symbol should have location"
            assert "relativePath" in sym["location"], "Symbol location should have relativePath"
            assert sym["location"]["relativePath"].replace("\\", "/").endswith("Utils.java"), "printHello should be in Utils.java"

    @pytest.mark.parametrize("language_server", [Language.JAVA], indirect=True)
    def test_search_with_class_path(self, language_server: SolidLanguageServer) -> None:
        """Test searching for symbols with class/method path."""
        # Search for printHello method in Utils class
        # Should return Utils class with printHello in its children
        symbols = language_server.search_symbols("Utils/printHello")
        
        assert len(symbols) > 0, "Should find Utils class containing printHello method"
        assert symbols[0]["name"] == "Utils", "Root symbol should be Utils class"
        assert symbols[0]["kind"] == SymbolKind.Class, "Root symbol should be a class"
        
        # Verify children contain printHello
        children = symbols[0].get("children", [])
        assert len(children) > 0, "Utils class should have children"
        print_hello_methods = [c for c in children if c["name"] == "printHello"]
        assert len(print_hello_methods) > 0, "Utils class should contain printHello method"

    @pytest.mark.parametrize("language_server", [Language.JAVA], indirect=True)
    def test_search_absolute_path(self, language_server: SolidLanguageServer) -> None:
        """Test searching with absolute name path."""
        # Absolute path requires exact match from file root
        # Should return Utils class with printHello in its children
        symbols = language_server.search_symbols("/Utils/printHello")
        
        assert len(symbols) > 0, "Should find Utils class with absolute path"
        assert symbols[0]["name"] == "Utils", "Root symbol should be Utils class"
        children_names = [c["name"] for c in symbols[0].get("children", [])]
        assert "printHello" in children_names, "Utils class should contain printHello method"

    @pytest.mark.parametrize("language_server", [Language.JAVA], indirect=True)
    def test_search_substring_matching(self, language_server: SolidLanguageServer) -> None:
        """Test substring matching for method names."""
        # Find all methods containing "print"
        symbols = language_server.search_symbols("print", substring_matching=True)
        
        assert len(symbols) > 0, "Should find Utils class with absolute path"
        assert symbols[0]["name"] == "Utils", "Root symbol should be Utils class"
        children_names = [c["name"] for c in symbols[0].get("children", [])]
        assert "printHello" in children_names, "Utils class should contain printHello method"

    @pytest.mark.parametrize("language_server", [Language.JAVA], indirect=True)
    def test_search_with_kind_filter(self, language_server: SolidLanguageServer) -> None:
        """Test filtering symbols by kind."""
        # Search for all classes (kind=5)
        symbols = language_server.search_symbols(
            "Utils",
            include_kinds=[SymbolKind.Class]
        )
        
        assert len(symbols) > 0, "Should find Utils class"
        assert all(s["kind"] == SymbolKind.Class for s in symbols), "All results should be classes"
        assert symbols[0]["name"] == "Utils", "Should find Utils class"

    @pytest.mark.parametrize("language_server", [Language.JAVA], indirect=True)
    def test_search_exclude_kinds(self, language_server: SolidLanguageServer) -> None:
        """Test excluding symbol kinds."""
        # Search for Main but exclude classes, should only get methods/fields
        symbols = language_server.search_symbols(
            "Main",
            exclude_kinds=[SymbolKind.Class]
        )
        
        # Should not include the Main class itself
        assert not any(s["kind"] == SymbolKind.Class for s in symbols), "Should not include classes"

    @pytest.mark.parametrize("language_server", [Language.JAVA], indirect=True)
    def test_search_within_file(self, language_server: SolidLanguageServer) -> None:
        """Test searching within a specific file."""
        file_path = os.path.join("src", "main", "java", "test_repo", "Utils.java")
        
        # Search for methods only in Utils.java
        symbols = language_server.search_symbols(
            "printHello",
            within_relative_path=file_path
        )
        
        assert len(symbols) > 0, "Should find printHello in Utils.java"
        for sym in symbols:
            rel_path = sym["location"]["relativePath"].replace("\\", "/")
            assert rel_path.endswith("Utils.java"), "All results should be from Utils.java"

    @pytest.mark.parametrize("language_server", [Language.JAVA], indirect=True)
    def test_search_within_directory(self, language_server: SolidLanguageServer) -> None:
        """Test searching within a specific directory."""
        dir_path = os.path.join("src", "main", "java", "test_repo")
        
        # Search for all symbols in test_repo directory
        symbols = language_server.search_symbols(
            "Main",
            within_relative_path=dir_path
        )
        
        assert len(symbols) > 0, "Should find symbols in test_repo directory"
        for sym in symbols:
            rel_path = sym["location"]["relativePath"].replace("\\", "/")
            assert "test_repo" in rel_path, "All results should be from test_repo directory"

    @pytest.mark.parametrize("language_server", [Language.JAVA], indirect=True)
    def test_search_nonexistent_symbol(self, language_server: SolidLanguageServer) -> None:
        """Test searching for non-existent symbol returns empty list."""
        symbols = language_server.search_symbols("NonExistentMethod12345")
        
        assert len(symbols) == 0, "Should return empty list for non-existent symbol"

    @pytest.mark.parametrize("language_server", [Language.JAVA], indirect=True)
    def test_search_parent_chain_integrity(self, language_server: SolidLanguageServer) -> None:
        """Test that parent references are correctly set up the chain."""
        # Search for a method path and verify complete parent chain
        # Should return Utils class with printHello as child
        symbols = language_server.search_symbols("Utils/printHello")
        
        assert len(symbols) > 0, "Should find Utils class"
        utils_class = symbols[0]
        assert utils_class["name"] == "Utils", "Root symbol should be Utils class"
        
        # Verify Utils class has parent reference to file
        assert "parent" in utils_class, "Utils class should have parent reference"
        file_parent = utils_class["parent"]
        assert file_parent is not None, "Should have file parent"
        assert file_parent["kind"] == SymbolKind.File, "Parent should be file"
        
        # Verify printHello method exists in children and has parent reference
        children = utils_class.get("children", [])
        print_hello = next((c for c in children if c["name"] == "printHello"), None)
        assert print_hello is not None, "Should find printHello in children"
        assert "parent" in print_hello, "printHello should have parent reference"
        assert print_hello["parent"]["name"] == "Utils", "printHello's parent should be Utils"

    @pytest.mark.parametrize("language_server", [Language.JAVA], indirect=True) 
    def test_search_class_with_methods(self, language_server: SolidLanguageServer) -> None:
        """Test searching returns symbols with children populated."""
        # Search for Utils class
        symbols = language_server.search_symbols(
            "Utils",
            include_kinds=[SymbolKind.Class]
        )
        
        assert len(symbols) > 0, "Should find Utils class"
        utils_class = symbols[0]
        
        # Should have children (methods)
        assert "children" in utils_class, "Class should have children field"
        assert "Utils" == utils_class['name'], "Should find Utils class"

    @pytest.mark.parametrize("language_server", [Language.JAVA], indirect=True)
    def test_request_text_document_diagnostics(self, language_server: SolidLanguageServer) -> None:
        """Test requesting diagnostics for a specific file using JDTLS push notifications."""
        # 指定一个具体的Java文件路径（相对路径）
        file_path = os.path.join("src", "main", "java", "test_repo", "Utils.java")
        
        # 调用 request_text_document_diagnostics 方法
        # JDTLS uses push-based diagnostics, so this waits for notifications
        diagnostics = language_server.request_text_document_diagnostics(file_path, timeout=10.0)
        
        # 验证返回的是列表类型
        assert isinstance(diagnostics, list), "Should return a list of diagnostics"
        
        # 如果有诊断信息，验证其结构
        for diagnostic in diagnostics:
            assert "severity" in diagnostic, "Diagnostic should have severity"
            assert "message" in diagnostic, "Diagnostic should have message"
            assert "range" in diagnostic, "Diagnostic should have range"
            assert "uri" in diagnostic, "Diagnostic should have uri"
            
            # 验证 range 结构
            assert "start" in diagnostic["range"], "Range should have start"
            assert "end" in diagnostic["range"], "Range should have end"

    @pytest.mark.parametrize("language_server", [Language.JAVA], indirect=True)
    def test_request_text_document_diagnostics_with_errors(self, language_server: SolidLanguageServer) -> None:
        """Test requesting diagnostics for a file with intentional compilation errors."""
        # 创建包含错误的临时Java文件
        file_path = os.path.join("src", "main", "java", "test_repo", "ErrorFile.java")
        repo_path = get_repo_path(Language.JAVA)
        full_path = os.path.join(repo_path, file_path)
        
        error_content = """package test_repo;

public class ErrorFile {
    // Intentional error: missing semicolon
    public static void testMethod() {
        String test = "hello"
    }
    
    // Intentional error: undefined variable
    public void anotherMethod() {
        System.out.println(undefinedVariable);
    }
    
    // Intentional error: incompatible types
    public int wrongReturnType() {
        return "string instead of int";
    }
}
"""
        
        try:
            # 创建错误文件
            os.makedirs(os.path.dirname(full_path), exist_ok=True)
            with open(full_path, "w", encoding="utf-8") as f:
                f.write(error_content)
            
            # 调用 request_text_document_diagnostics 方法
            diagnostics = language_server.request_text_document_diagnostics(file_path, timeout=30.0)
            
            # 验证返回的是列表类型
            assert isinstance(diagnostics, list), "Should return a list of diagnostics"
            
            # 应该包含错误诊断信息（因为文件有编译错误）
            assert len(diagnostics) > 0, "Should have at least one diagnostic for error file"
            
            # 验证诊断信息包含错误级别
            error_found = False
            for diagnostic in diagnostics:
                assert "severity" in diagnostic, "Diagnostic should have severity"
                assert "message" in diagnostic, "Diagnostic should have message"
                assert "range" in diagnostic, "Diagnostic should have range"
                assert "uri" in diagnostic, "Diagnostic should have uri"
                
                # 检查是否有错误级别的诊断（severity=1 表示 Error）
                if diagnostic.get("severity") == 1:
                    error_found = True
            
            assert error_found, "Should have at least one error-level diagnostic"
            
        finally:
            # 清理：删除临时文件
            if os.path.exists(full_path):
                os.remove(full_path)

    @pytest.mark.parametrize("language_server", [Language.JAVA], indirect=True)
    def test_request_dir_overview(self, language_server: SolidLanguageServer) -> None:
        """Test request_dir_overview returns top-level symbols for all files in a directory."""
        dir_path = os.path.join("src", "main", "java", "test_repo")
        
        # Get directory overview
        dir_overview = language_server.request_dir_overview(dir_path)
        
        assert len(dir_overview) > 0, "Should return overview for files in directory"
        
        # Verify structure: mapping from relative paths to lists of top-level symbols
        for rel_path, symbols in dir_overview.items():
            assert isinstance(symbols, list), "Value should be a list of symbols"
            assert len(symbols) > 0, f"File {rel_path} should have at least one top-level symbol"
            
            # Verify each symbol has proper structure
            for symbol in symbols:
                assert "name" in symbol, "Symbol should have name"
                assert "kind" in symbol, "Symbol should have kind"
                assert "location" in symbol, "Symbol should have location"
                
        # Verify we got expected files
        normalized_paths = [p.replace("\\", "/") for p in dir_overview.keys()]
        assert any("Utils.java" in p for p in normalized_paths), "Should include Utils.java"
        assert any("Main.java" in p for p in normalized_paths), "Should include Main.java"

    @pytest.mark.parametrize("language_server", [Language.JAVA], indirect=True)
    def test_request_overview_with_file(self, language_server: SolidLanguageServer) -> None:
        """Test request_overview with a file path returns symbols for that file."""
        file_path = os.path.join("src", "main", "java", "test_repo", "Utils.java")
        
        # Get overview for single file
        file_overview = language_server.request_overview(file_path)
        
        assert len(file_overview) == 1, "Should return overview for one file"
        assert file_path in file_overview or file_path.replace("\\", "/") in [k.replace("\\", "/") for k in file_overview.keys()], "Should include Utils.java"
        
        symbols = list(file_overview.values())[0]
        assert len(symbols) > 0, "File should have at least one top-level symbol"
        
        # Verify symbols structure
        for symbol in symbols:
            assert "name" in symbol, "Symbol should have name"
            assert "kind" in symbol, "Symbol should have kind"
    
    @pytest.mark.parametrize("language_server", [Language.JAVA], indirect=True)
    def test_request_overview_with_directory(self, language_server: SolidLanguageServer) -> None:
        """Test request_overview with a directory path returns symbols for all files."""
        dir_path = os.path.join("src", "main", "java", "test_repo")
        
        # Get overview for directory
        dir_overview = language_server.request_overview(dir_path)
        
        assert len(dir_overview) > 0, "Should return overview for directory"
        
        # Should contain multiple files
        normalized_paths = [p.replace("\\", "/") for p in dir_overview.keys()]
        assert any("Utils.java" in p for p in normalized_paths), "Should include Utils.java"
        assert any("Main.java" in p for p in normalized_paths), "Should include Main.java"


class TestLanguageServerSymbolRetriever:
    """Test LanguageServerSymbolRetriever methods with Kùzu cache integration."""

    @pytest.mark.parametrize("language_server", [Language.JAVA], indirect=True)
    def test_symbol_retriever_find(self, language_server: SolidLanguageServer) -> None:
        """Test LanguageServerSymbolRetriever.find() method."""
        retriever = LanguageServerSymbolRetriever(language_server)
        
        # Find methods named "printHello"
        symbols = retriever.find("printHello")
        
        assert len(symbols) > 0, "Should find at least one printHello method"
        
        # Verify symbol properties
        for symbol in symbols:
            assert symbol.name == "printHello", "Symbol name should be printHello"
            assert symbol.symbol_kind == SymbolKind.Method, "Symbol should be a method"
            assert symbol.location.relative_path is not None, "Symbol should have relative path"
    
    @pytest.mark.parametrize("language_server", [Language.JAVA], indirect=True)
    def test_symbol_retriever_find_unique(self, language_server: SolidLanguageServer) -> None:
        """Test LanguageServerSymbolRetriever.find_unique() method."""
        retriever = LanguageServerSymbolRetriever(language_server)
        
        # Find unique symbol with class path (using overload index)
        file_path = os.path.join("src", "main", "java", "test_repo", "Utils.java")
        symbol = retriever.find_unique("Utils/printHello[0]", within_relative_path=file_path)
        
        assert symbol is not None, "Should find unique printHello method"
        assert symbol.name == "printHello", "Symbol name should be printHello"
        assert symbol.symbol_kind == SymbolKind.Method, "Symbol should be a method"
        assert symbol.overload_idx == 0, "Symbol should be overload index 0"
        
        # Verify location
        assert symbol.location.relative_path is not None, "Symbol should have relative path"
        assert "Utils.java" in symbol.location.relative_path.replace("\\", "/"), "Symbol should be in Utils.java"
    
    @pytest.mark.parametrize("language_server", [Language.JAVA], indirect=True)
    def test_symbol_retriever_get_symbol_overview(self, language_server: SolidLanguageServer) -> None:
        """Test LanguageServerSymbolRetriever.get_symbol_overview() method."""
        retriever = LanguageServerSymbolRetriever(language_server)
        
        # Get overview for a directory
        dir_path = os.path.join("src", "main", "java", "test_repo")
        overview = retriever.get_symbol_overview(dir_path, depth=0)
        
        assert len(overview) > 0, "Should return overview for files in directory"
        
        # Verify structure: mapping from file paths to lists of symbol dicts
        for file_path, symbols in overview.items():
            assert isinstance(symbols, list), "Value should be a list of symbol dicts"
            assert len(symbols) > 0, f"File {file_path} should have at least one symbol"
            
            # Verify each symbol dict has proper structure
            for symbol_dict in symbols:
                assert "name" in symbol_dict, "Symbol dict should have name"
                assert "kind" in symbol_dict, "Symbol dict should have kind"
                # Should not include relativePath when include_relative_path=False (default in get_symbol_overview)
        
        # Verify we got expected files
        normalized_paths = [p.replace("\\", "/") for p in overview.keys()]
        assert any("Utils.java" in p for p in normalized_paths), "Should include Utils.java"
        assert any("Main.java" in p for p in normalized_paths), "Should include Main.java"
    
    @pytest.mark.parametrize("language_server", [Language.JAVA], indirect=True)
    def test_symbol_retriever_get_symbol_overview_with_file(self, language_server: SolidLanguageServer) -> None:
        """Test get_symbol_overview with a single file path."""
        retriever = LanguageServerSymbolRetriever(language_server)
        
        # Get overview for a single file
        file_path = os.path.join("src", "main", "java", "test_repo", "Utils.java")
        overview = retriever.get_symbol_overview(file_path, depth=0)
        
        assert len(overview) == 1, "Should return overview for one file"
        
        # Get the symbols for the file
        symbols = list(overview.values())[0]
        assert len(symbols) > 0, "File should have at least one top-level symbol"
        
        # Verify the Utils class is present
        symbol_names = [s["name"] for s in symbols]
        assert "Utils" in symbol_names, "Should include Utils class"
    
    @pytest.mark.parametrize("language_server", [Language.JAVA], indirect=True)
    def test_symbol_retriever_get_symbol_overview_with_depth(self, language_server: SolidLanguageServer) -> None:
        """Test get_symbol_overview with depth parameter to include children."""
        retriever = LanguageServerSymbolRetriever(language_server)
        
        # Get overview with depth=1 to include children
        file_path = os.path.join("src", "main", "java", "test_repo", "Utils.java")
        overview = retriever.get_symbol_overview(file_path, depth=1)
        
        assert len(overview) == 1, "Should return overview for one file"
        
        symbols = list(overview.values())[0]
        assert len(symbols) > 0, "File should have at least one symbol"
        
        # Find the Utils class
        utils_class = next((s for s in symbols if s["name"] == "Utils"), None)
        assert utils_class is not None, "Should find Utils class"
        
        # With depth=1, should include children
        assert "children" in utils_class, "Utils class should have children field"
        children = utils_class.get("children", [])
        assert len(children) > 0, "Utils class should have at least one child method"
        
        # Verify printHello method is in children
        child_names = [c["name"] for c in children]
        assert "printHello" in child_names, "Should include printHello method as child"


    def test_search_ofield_in_crm_hc(self) -> None:
        """Test searching for OField symbol in E:\\worksapce\\upgrade\\crm-hc project."""
        import logging
        from sensai.util.logging import configure
        from serena.config.serena_config import SerenaConfig
        from serena.project import Project
        
        # Setup logging
        lvl = logging.DEBUG
        configure(level=lvl)
        
        # Enable DEBUG for kuzu_symbol_cache module
        logging.getLogger("solidlsp.util.kuzu_symbol_cache").setLevel(logging.DEBUG)
        
        # Load project configuration
        serena_config = SerenaConfig.from_config_file()
        project_path = "E:/worksapce/upgrade/crm-hc"
        proj = Project.load(os.path.abspath(project_path))
        
        # Create language server manager with rebuild_indexes=True
        timeout = 30.0
        ls_mgr = proj.create_language_server_manager(
            log_level=lvl, 
            ls_timeout=timeout, 
            ls_specific_settings=serena_config.ls_specific_settings
        )
        proj.init_index_cache()
        try:
            ls = ls_mgr._default_language_server
            # Search for OField symbol
            symbols = ls.search_symbols(name_path_pattern="OField") 
            # Verify results
            assert len(symbols) > 0, "Should find OField symbol in the project"
            
            for symbol in symbols:
                assert "name" in symbol, "Symbol should have name"
                assert "kind" in symbol, "Symbol should have kind"
                assert "location" in symbol, "Symbol should have location"
                
                # Log found symbol information
                logging.info(f"Found symbol: {symbol['name']} (kind: {symbol['kind']}) in {symbol['location'].get('relativePath', 'N/A')}")
        
        finally:
            # Always stop all language servers
            ls_mgr.stop_all()

