"""
Language server-related tools
"""

import os
from collections import defaultdict
from collections.abc import Sequence
from copy import copy
from itertools import count
from typing import Any

from serena.tools import (
    SUCCESS_RESULT,
    Tool,
    ToolMarkerSymbolicEdit,
    ToolMarkerSymbolicRead,
)
from serena.tools.tools_base import ToolMarkerOptional
from solidlsp.ls_types import DiagnosticSeverity, SymbolKind


# Counter for file reference numbering
_file_ref_counter = count(1)


def _sanitize_symbol_dict(symbol_dict: dict[str, Any]) -> dict[str, Any]:
    """
    Sanitize a symbol dictionary inplace by removing unnecessary information.
    """
    # We replace the location entry, which repeats line information already included in body_location
    # and has unnecessary information on column, by just the relative path.
    symbol_dict = copy(symbol_dict)
    s_relative_path = symbol_dict.get("location", {}).get("relative_path")
    if s_relative_path is not None:
        symbol_dict["relative_path"] = s_relative_path
    symbol_dict.pop("location", None)
    # also remove name, name_path should be enough
    symbol_dict.pop("name")
    return symbol_dict


def _compress_symbols(symbol_dicts: list[dict[str, Any]]) -> list[dict[str, Any] | str]:
    """
    Compress symbols by creating a file path mapping to reduce token consumption.
    
    Modifies symbol_dicts in-place by replacing 'relative_path' with short 'file_ref'.
    Returns a flat list with file mapping string as first element.
    
    :param symbol_dicts: List of sanitized symbol dictionaries (modified in-place)
    :return: Flat list: [file_mapping_string, symbol1, symbol2, ...]
    """
    if not symbol_dicts:
        return []
    
    # Build mapping and modify symbols in single pass
    path_to_key = {}
    for symbol_dict in symbol_dicts:
        relative_path = symbol_dict.get("relative_path")
        if relative_path:
            if relative_path not in path_to_key:
                path_to_key[relative_path] = f"e{next(_file_ref_counter)}"
            symbol_dict["relative_path"] = path_to_key[relative_path]
    
    # Create compact file mapping string
    files_mapping = "Files: " + ", ".join(f"{key}= {path}" for path, key in sorted(path_to_key.items()))
    
    return [files_mapping] + symbol_dicts


class RestartLanguageServerTool(Tool, ToolMarkerOptional):
    """Restarts the language server, may be necessary when edits not through Serena happen."""

    def apply(self) -> str:
        """Use this tool only on explicit user request or after confirmation.
        It may be necessary to restart the language server if it hangs.
        """
        self.agent.reset_language_server_manager()
        return SUCCESS_RESULT


class GetSymbolsOverviewTool(Tool, ToolMarkerSymbolicRead):
    """
    Gets an overview of the top-level symbols defined in a given file.
    """

    def apply(self, relative_path: str, depth: int = 0, max_answer_chars: int = -1) -> str:
        """
        Use this tool to get a high-level understanding of the code symbols in a file.
        This should be the first tool to call when you want to understand a new file, unless you already know
        what you are looking for.

        :param relative_path: the relative path to the file to get the overview of
        :param depth: depth up to which descendants of top-level symbols shall be retrieved
            (e.g. 1 retrieves immediate children). Default 0.
        :param max_answer_chars: if the overview is longer than this number of characters,
            no content will be returned. -1 means the default value from the config will be used.
            Don't adjust unless there is really no other way to get the content required for the task.
        :return: a JSON object containing symbols grouped by kind in a compact format.
        """
        symbol_retriever = self.create_language_server_symbol_retriever()
        file_path = os.path.join(self.project.project_root, relative_path)

        # The symbol overview is capable of working with both files and directories,
        # but we want to ensure that the user provides a file path.
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File or directory {relative_path} does not exist in the project.")
        if os.path.isdir(file_path):
            raise ValueError(f"Expected a file path, but got a directory path: {relative_path}. ")
        result = symbol_retriever.get_symbol_overview(relative_path, depth=depth)[relative_path]
        # Transform to compact format
        compact_result = self._transform_symbols_to_compact_format(result)
        result_json_str = self._to_json(compact_result)
        return result_json_str

    @staticmethod
    def _transform_symbols_to_compact_format(symbols: list[dict[str, Any]]) -> dict[str, list]:
        """
        Transform symbol overview from verbose format to compact grouped format.

        Groups symbols by kind and uses names instead of full symbol objects.
        For symbols with children, creates nested dictionaries.

        The name_path can be inferred from the hierarchical structure:
        - Top-level symbols: name_path = name
        - Nested symbols: name_path = parent_name + "/" + name
        For example, "convert" under class "ProjectType" has name_path "ProjectType/convert".
        """
        result = defaultdict(list)

        for symbol in symbols:
            kind = symbol.get("kind", "Unknown")
            name = symbol.get("name", "unknown")
            children = symbol.get("children", [])

            if children:
                # Symbol has children: create nested dict {name: children_dict}
                children_dict = GetSymbolsOverviewTool._transform_symbols_to_compact_format(children)
                result[kind].append({name: children_dict})
            else:
                # Symbol has no children: just add the name
                result[kind].append(name)

        return result


class FindSymbolTool(Tool, ToolMarkerSymbolicRead):
    """
    Searches for symbols using grep-style pattern matching.
    """

    # noinspection PyDefaultArgument
    def apply(
        self,
        pattern: str,
        search_in: str = "",
        include_body: bool = False,
        include_kinds: list[int] = [],  # noqa: B006
        exclude_kinds: list[int] = [],  # noqa: B006
        max_answer_chars: int = 50000,
    ) -> str:
        """
        Searches for symbols using regular expressions (like grep).
        
        **Usage (Like grep)**:
        ```
        find_symbol("UserService")                        # Search everywhere
        find_symbol("UserService", "src/main/java")      # Search in directory
        find_symbol("UserService", "UserService.java")   # Search in file
        find_symbol("get.*", ".*Controller.java")        # Both are regex
        ```
        
        **Symbol Name Paths**:
        A name path is the hierarchical path to a symbol within a source file:
        - Method in class: "MyClass/myMethod"
        - Nested class method: "OuterClass/InnerClass/method"
        - Overloaded method: "MyClass/myMethod#1" (0-based index using # symbol)

        **Pattern Examples**:
        - "UserService" → matches any symbol containing "UserService"
        - "^UserService$" → matches exactly "UserService" at file root
        - "get.*Mapping" → matches "getRequestMapping", "getPathMapping"
        - ".*Service$" → matches symbols ending with "Service"
        - "^com/example/.*" → matches symbols in com.example package
        
        **Search Scope (search_in)**:
        - Empty/omitted: searches entire workspace
        - File path: "src/UserService.java" (if file exists, searches only that file)
        - Directory: "src/main/java" (searches all files in directory)
        - Regex: ".*\\.java$" (matches files by regex pattern)
        
        **Note**: This tool returns matched symbols without their children. 
        Use `get_symbols_overview` tool to see the detailed structure of a specific file/symbol.

        :param pattern: Regular expression to match symbol name paths (what to search for)
        :param search_in: Where to search (file path, directory, or regex). Empty = search everywhere.
        :param include_body: Include symbol source code in results (use carefully, increases size)
        :param include_kinds: List of LSP symbol kinds to include (e.g., [5] for classes, [12] for functions).
            Common: 5=class, 6=method, 12=function, 13=variable, 10=enum, 11=interface.
            Full list: 1=file, 2=module, 3=namespace, 4=package, 5=class, 6=method, 7=property, 8=field,
            9=constructor, 10=enum, 11=interface, 12=function, 13=variable, 14=constant, 22=enum member,
            23=struct. Empty = include all.
        :param exclude_kinds: Symbol kinds to exclude (takes precedence over include_kinds)
        :param max_answer_chars: Max result size in characters (default 50000).
        :return: JSON list of matching symbols with metadata (name_path, kind, location, body)
        """
        parsed_include_kinds: Sequence[SymbolKind] | None = [SymbolKind(k) for k in include_kinds] if include_kinds else None
        parsed_exclude_kinds: Sequence[SymbolKind] | None = [SymbolKind(k) for k in exclude_kinds] if exclude_kinds else None
        symbol_retriever = self.create_language_server_symbol_retriever()
        symbols = symbol_retriever.find(
            pattern=pattern,
            search_in=search_in or None,
            include_kinds=parsed_include_kinds,
            exclude_kinds=parsed_exclude_kinds,
        )
        # depth=0 to only return the matched symbols, no children
        symbol_dicts = [_sanitize_symbol_dict(s.to_dict(kind=True, location=True, depth=0, include_body=include_body)) for s in symbols]
        # Compress symbols with path mapping to reduce token consumption
        compressed_result = _compress_symbols(symbol_dicts)
        result = self._to_json(compressed_result)
        return self._limit_length(result, max_answer_chars)


class FindReferencingSymbolsTool(Tool, ToolMarkerSymbolicRead):
    """
    Finds symbols that reference the given symbol using the language server backend
    """

    # noinspection PyDefaultArgument
    def apply(
        self,
        name_path: str,
        relative_path: str,
        include_kinds: list[int] = [],  # noqa: B006
        exclude_kinds: list[int] = [],  # noqa: B006
        max_answer_chars: int = -1,
    ) -> str:
        """
        Finds references to the symbol at the given `name_path`. The result will contain metadata about the referencing symbols
        as well as a short code snippet around the reference.

        :param name_path: for finding the symbol to find references for, same logic as in the `find_symbol` tool.
        :param relative_path: the relative path to the file containing the symbol for which to find references.
            Note that here you can't pass a directory but must pass a file.
        :param include_kinds: same as in the `find_symbol` tool.
        :param exclude_kinds: same as in the `find_symbol` tool.
        :param max_answer_chars: same as in the `find_symbol` tool.
        :return: a list of JSON objects with the symbols referencing the requested symbol
        """
        include_body = False  # It is probably never a good idea to include the body of the referencing symbols
        parsed_include_kinds: Sequence[SymbolKind] | None = [SymbolKind(k) for k in include_kinds] if include_kinds else None
        parsed_exclude_kinds: Sequence[SymbolKind] | None = [SymbolKind(k) for k in exclude_kinds] if exclude_kinds else None
        symbol_retriever = self.create_language_server_symbol_retriever()
        references_in_symbols = symbol_retriever.find_referencing_symbols(
            name_path,
            relative_file_path=relative_path,
            include_body=include_body,
            include_kinds=parsed_include_kinds,
            exclude_kinds=parsed_exclude_kinds,
        )
        reference_dicts = []
        for ref in references_in_symbols:
            ref_dict = ref.symbol.to_dict(kind=True, location=True, depth=0, include_body=include_body)
            ref_dict = _sanitize_symbol_dict(ref_dict)
            if not include_body:
                ref_relative_path = ref.symbol.location.relative_path
                assert ref_relative_path is not None, f"Referencing symbol {ref.symbol.name} has no relative path, this is likely a bug."
                content_around_ref = self.project.retrieve_content_around_line(
                    relative_file_path=ref_relative_path, line=ref.line, context_lines_before=1, context_lines_after=1
                )
                ref_dict["content_around_reference"] = content_around_ref.to_display_string()
            reference_dicts.append(ref_dict)
        # Compress references with path mapping to reduce token consumption
        compressed_result = _compress_symbols(reference_dicts)
        result = self._to_json(compressed_result)
        return self._limit_length(result, max_answer_chars)


class ReplaceSymbolBodyTool(Tool, ToolMarkerSymbolicEdit):
    """
    Replaces the full definition of a symbol using the language server backend.
    """

    def apply(
        self,
        name_path: str,
        relative_path: str,
        body: str,
    ) -> str:
        r"""
        Replaces the body of the symbol with the given `name_path`.

        The tool shall be used to replace symbol bodies that have been previously retrieved
        (e.g. via `find_symbol`).
        IMPORTANT: Do not use this tool if you do not know what exactly constitutes the body of the symbol.

        :param name_path: for finding the symbol to replace, same logic as in the `find_symbol` tool.
        :param relative_path: the relative path to the file containing the symbol
        :param body: the new symbol body. The symbol body is the definition of a symbol
            in the programming language, including e.g. the signature line for functions.
            IMPORTANT: The body does NOT include any preceding docstrings/comments or imports, in particular.
        """
        code_editor = self.create_code_editor()
        code_editor.replace_body(
            name_path,
            relative_file_path=relative_path,
            body=body,
        )
        return SUCCESS_RESULT


class InsertAfterSymbolTool(Tool, ToolMarkerSymbolicEdit):
    """
    Inserts content after the end of the definition of a given symbol.
    """

    def apply(
        self,
        name_path: str,
        relative_path: str,
        body: str,
    ) -> str:
        """
        Inserts the given body/content after the end of the definition of the given symbol (via the symbol's location).
        A typical use case is to insert a new class, function, method, field or variable assignment.

        :param name_path: name path of the symbol after which to insert content (definitions in the `find_symbol` tool apply)
        :param relative_path: the relative path to the file containing the symbol
        :param body: the body/content to be inserted. The inserted code shall begin with the next line after
            the symbol.
        """
        code_editor = self.create_code_editor()
        code_editor.insert_after_symbol(name_path, relative_file_path=relative_path, body=body)
        return SUCCESS_RESULT


class InsertBeforeSymbolTool(Tool, ToolMarkerSymbolicEdit):
    """
    Inserts content before the beginning of the definition of a given symbol.
    """

    def apply(
        self,
        name_path: str,
        relative_path: str,
        body: str,
    ) -> str:
        """
        Inserts the given content before the beginning of the definition of the given symbol (via the symbol's location).
        A typical use case is to insert a new class, function, method, field or variable assignment; or
        a new import statement before the first symbol in the file.

        :param name_path: name path of the symbol before which to insert content (definitions in the `find_symbol` tool apply)
        :param relative_path: the relative path to the file containing the symbol
        :param body: the body/content to be inserted before the line in which the referenced symbol is defined
        """
        code_editor = self.create_code_editor()
        code_editor.insert_before_symbol(name_path, relative_file_path=relative_path, body=body)
        return SUCCESS_RESULT


class RenameSymbolTool(Tool, ToolMarkerSymbolicEdit):
    """
    Renames a symbol throughout the codebase using language server refactoring capabilities.
    """

    def apply(
        self,
        name_path: str,
        relative_path: str,
        new_name: str,
    ) -> str:
        """
        Renames the symbol with the given `name_path` to `new_name` throughout the entire codebase.
        Note: for languages with method overloading, like Java, name_path may have to include a method's
        signature to uniquely identify a method.

        :param name_path: name path of the symbol to rename (definitions in the `find_symbol` tool apply)
        :param relative_path: the relative path to the file containing the symbol to rename
        :param new_name: the new name for the symbol
        :return: result summary indicating success or failure
        """
        code_editor = self.create_code_editor()
        status_message = code_editor.rename_symbol(name_path, relative_file_path=relative_path, new_name=new_name)
        return status_message


class DiagnosticsTool(Tool, ToolMarkerSymbolicRead):
    """
    Requests diagnostics for a given file from the language server.
    """

    def apply(self, relative_path: str, max_answer_chars: int = -1) -> str:
        """
        Retrieves diagnostics (errors, warnings, information, hints, etc.) for the given file from the language server.
        Diagnostics include compilation errors, warnings, linting issues, type errors, and code quality suggestions.
        The diagnostic 'code' field often contains error codes that can be used to identify specific issues.

        :param relative_path: the relative path to the file to retrieve diagnostics for
        :param max_answer_chars: maximum number of characters for the result. If the result is longer than this,
            an error message will be returned. Use -1 for the default limit from the config.
        :return: a JSON array of diagnostic objects. Each diagnostic contains:
            - severity: integer (1=Error, 2=Warning, 3=Information, 4=Hint)
            - message: string describing the issue or status
            - range: optional object with start/end line and character positions
            
            Special status messages:
            - "Diagnostics check completed successfully - no issues found" (severity=3): Analysis passed with no problems
            - "Timeout ({n}s) waiting for diagnostics" (severity=2): Language server timeout occurred
        """
        symbol_retriever = self.create_language_server_symbol_retriever()
        diagnostics = symbol_retriever.request_text_document_diagnostics(relative_path, timeout=30)
        diagnostics_list = []
        for diag in diagnostics:
            item = {
                "message": diag["message"]
            }
            # severity is optional - convert to readable string if present
            if "severity" in diag:
                severity_value = diag["severity"]
                try:
                    severity_enum = DiagnosticSeverity(severity_value)
                    item["severity"] = severity_enum.name
                except (ValueError, KeyError):
                    item["severity"] = severity_value
            # range is optional - only include if present
            if "range" in diag:
                item["range"] = diag["range"]
            diagnostics_list.append(item)
        result = self._to_json(diagnostics_list)
        return self._limit_length(result, max_answer_chars)

