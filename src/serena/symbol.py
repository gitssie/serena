import json
import logging
import os
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterator, Sequence
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any, NotRequired, Self, TypedDict, Union

from sensai.util.string import ToStringMixin

from solidlsp import SolidLanguageServer
from solidlsp.ls import ReferenceInSymbol as LSPReferenceInSymbol
from solidlsp.ls_types import Position, SymbolKind, UnifiedSymbolInformation, Diagnostic

from .ls_manager import LanguageServerManager
from .project import Project

if TYPE_CHECKING:
    from .agent import SerenaAgent

log = logging.getLogger(__name__)
NAME_PATH_SEP = "/"


@dataclass
class LanguageServerSymbolLocation:
    """
    Represents the (start) location of a symbol identifier, which, within Serena, uniquely identifies the symbol.
    """

    relative_path: str | None
    """
    the relative path of the file containing the symbol; if None, the symbol is defined outside of the project's scope
    """
    line: int | None
    """
    the line number in which the symbol identifier is defined (if the symbol is a function, class, etc.);
    may be None for some types of symbols (e.g. SymbolKind.File)
    """
    column: int | None
    """
    the column number in which the symbol identifier is defined (if the symbol is a function, class, etc.);
    may be None for some types of symbols (e.g. SymbolKind.File)
    """

    def __post_init__(self) -> None:
        # Keep forward slashes for consistent cross-platform path representation
        pass

    def to_dict(self, include_relative_path: bool = True) -> dict[str, Any]:
        result = asdict(self)
        if not include_relative_path:
            result.pop("relative_path", None)
        return result

    def has_position_in_file(self) -> bool:
        return self.relative_path is not None and self.line is not None and self.column is not None


@dataclass
class PositionInFile:
    """
    Represents a character position within a file
    """

    line: int
    """
    the 0-based line number in the file
    """
    col: int
    """
    the 0-based column
    """

    def to_lsp_position(self) -> Position:
        """
        Convert to LSP Position.
        """
        return Position(line=self.line, character=self.col)


class Symbol(ToStringMixin, ABC):
    @abstractmethod
    def get_body_start_position(self) -> PositionInFile | None:
        pass

    @abstractmethod
    def get_body_end_position(self) -> PositionInFile | None:
        pass

    def get_body_start_position_or_raise(self) -> PositionInFile:
        """
        Get the start position of the symbol body, raising an error if it is not defined.
        """
        pos = self.get_body_start_position()
        if pos is None:
            raise ValueError(f"Body start position is not defined for {self}")
        return pos

    def get_body_end_position_or_raise(self) -> PositionInFile:
        """
        Get the end position of the symbol body, raising an error if it is not defined.
        """
        pos = self.get_body_end_position()
        if pos is None:
            raise ValueError(f"Body end position is not defined for {self}")
        return pos

    @abstractmethod
    def is_neighbouring_definition_separated_by_empty_line(self) -> bool:
        """
        :return: whether a symbol definition of this symbol's kind is usually separated from the
            previous/next definition by at least one empty line.
        """


class NamePathMatcher(ToStringMixin):
    """
    Matches name paths of symbols against Linux grep-style regular expressions.

    A name path is a path in the symbol tree *within a source file*.
    For example, the method `my_method` defined in class `MyClass` would have the name path `MyClass/my_method`.
    
    This class provides in-memory filtering using regular expressions that are consistent with
    DuckDB's regexp_matches() function. It's used to filter symbols retrieved from LSP when
    the cache is invalidated.
    
    Pattern matching:
     * Supports standard Python/POSIX regular expressions (e.g., "get.*Mappings", ".*Action/.*")
     * Case-insensitive by default (can be overridden with case_sensitive parameter)
     * Matches anywhere in the name path (use ^ and $ for anchoring)
     
    Examples:
     * "getUser" - matches any name path containing "getuser" (case-insensitive)
     * "^MyClass/.*" - matches name paths starting with "MyClass/"
     * ".*Action/execute$" - matches name paths ending with "Action/execute"
     * "get[A-Z].*" - matches name paths like "getUser", "getData" (uppercase after 'get')
    """

    def __init__(self, regex_pattern: str, case_sensitive: bool = False) -> None:
        """
        :param regex_pattern: Regular expression pattern to match against symbol name paths.
            This should be a standard Python/POSIX regex pattern.
        :param case_sensitive: Whether the matching should be case-sensitive. Default is False.
        """
        import re
        
        assert regex_pattern, "regex_pattern must not be empty"
        self._expr = regex_pattern
        self._case_sensitive = case_sensitive
        
        # Compile the regex pattern for efficiency
        flags = 0 if case_sensitive else re.IGNORECASE
        try:
            self._compiled_regex = re.compile(regex_pattern, flags)
        except re.error as e:
            raise ValueError(f"Invalid regular expression pattern '{regex_pattern}': {e}")

    def _tostring_includes(self) -> list[str]:
        return ["_expr"]

    def matches_ls_symbol(self, symbol: "LanguageServerSymbol") -> bool:
        """
        Check if a LanguageServerSymbol matches the regex pattern.
        
        :param symbol: The symbol to match against
        :return: True if the symbol's name path matches the pattern
        """
        return self.matches_name_path(symbol.get_name_path())
    
    def matches_name_path(self, name_path: str) -> bool:
        """
        Check if a name path string matches the regex pattern.
        
        :param name_path: The name path string to match (e.g., "MyClass/myMethod")
        :return: True if the name path matches the pattern
        """
        return bool(self._compiled_regex.search(name_path))

    def matches_components(self, symbol_name_path_parts: list[str], overload_idx: int | None) -> bool:
        """
        Match symbol name path parts against the regex pattern.
        This method is kept for backward compatibility but delegates to matches_name_path.
        
        :param symbol_name_path_parts: List of name path components (e.g., ["MyClass", "myMethod"])
        :param overload_idx: Overload index (currently ignored as overload handling is removed)
        :return: True if the name path matches the pattern
        """
        name_path = NAME_PATH_SEP.join(symbol_name_path_parts)
        return self.matches_name_path(name_path)


class LanguageServerSymbol(Symbol, ToStringMixin):
    def __init__(self, symbol_root_from_ls: UnifiedSymbolInformation) -> None:
        self.symbol_root = symbol_root_from_ls

    def _tostring_includes(self) -> list[str]:
        return []

    def _tostring_additional_entries(self) -> dict[str, Any]:
        return dict(name=self.name, kind=self.kind, num_children=len(self.symbol_root["children"]))

    @property
    def name(self) -> str:
        return self.symbol_root["name"]

    @property
    def kind(self) -> str:
        return SymbolKind(self.symbol_kind).name

    @property
    def symbol_kind(self) -> SymbolKind:
        return self.symbol_root["kind"]

    def is_low_level(self) -> bool:
        """
        :return: whether the symbol is a low-level symbol (variable, constant, etc.), which typically represents data
            rather than structure and therefore is not relevant in a high-level overview of the code.
        """
        return self.symbol_kind >= SymbolKind.Variable.value

    @property
    def overload_idx(self) -> int | None:
        return self.symbol_root.get("overload_idx")

    def is_neighbouring_definition_separated_by_empty_line(self) -> bool:
        return self.symbol_kind in (SymbolKind.Function, SymbolKind.Method, SymbolKind.Class, SymbolKind.Interface, SymbolKind.Struct)

    @property
    def relative_path(self) -> str | None:
        location = self.symbol_root.get("location")
        if location:
            return location.get("relativePath")
        return None

    @property
    def location(self) -> LanguageServerSymbolLocation:
        """
        :return: the start location of the actual symbol identifier
        """
        return LanguageServerSymbolLocation(relative_path=self.relative_path, line=self.line, column=self.column)

    @property
    def body_start_position(self) -> Position | None:
        location = self.symbol_root.get("location")
        if location:
            range_info = location.get("range")
            if range_info:
                start_pos = range_info.get("start")
                if start_pos:
                    return start_pos
        return None

    @property
    def body_end_position(self) -> Position | None:
        location = self.symbol_root.get("location")
        if location:
            range_info = location.get("range")
            if range_info:
                end_pos = range_info.get("end")
                if end_pos:
                    return end_pos
        return None

    def get_body_start_position(self) -> PositionInFile | None:
        start_pos = self.body_start_position
        if start_pos is None:
            return None
        return PositionInFile(line=start_pos["line"], col=start_pos["character"])

    def get_body_end_position(self) -> PositionInFile | None:
        end_pos = self.body_end_position
        if end_pos is None:
            return None
        return PositionInFile(line=end_pos["line"], col=end_pos["character"])

    def get_body_line_numbers(self) -> tuple[int | None, int | None]:
        start_pos = self.body_start_position
        end_pos = self.body_end_position
        start_line = start_pos["line"] if start_pos else None
        end_line = end_pos["line"] if end_pos else None
        return start_line, end_line

    @property
    def line(self) -> int | None:
        """
        :return: the line in which the symbol identifier is defined.
        """
        if "selectionRange" in self.symbol_root:
            return self.symbol_root["selectionRange"]["start"]["line"]
        else:
            # line is expected to be undefined for some types of symbols (e.g. SymbolKind.File)
            return None

    @property
    def column(self) -> int | None:
        if "selectionRange" in self.symbol_root:
            return self.symbol_root["selectionRange"]["start"]["character"]
        else:
            # precise location is expected to be undefined for some types of symbols (e.g. SymbolKind.File)
            return None

    @property
    def body(self) -> str | None:
        return self.symbol_root.get("body")

    def get_name_path(self) -> str:
        """
        Get the name path of the symbol, e.g. "class/method/inner_function" or
        "class/method#1" (overloaded method with identifying index).
        
        Uses # symbol for overload index instead of [] to avoid regex special characters.
        """
        name_path = self.symbol_root.get('name_path',None)
        if name_path is None:
            name_path = NAME_PATH_SEP.join(self.get_name_path_parts())
            if "overload_idx" in self.symbol_root and self.symbol_root['overload_idx'] > 0:
                name_path += f"#{self.symbol_root['overload_idx']}"
        return name_path

    def get_name_path_parts(self) -> list[str]:
        """
        Get the parts of the name path of the symbol (e.g. ["class", "method", "inner_function"]).
        """
        ancestors_within_file = list(self.iter_ancestors(up_to_symbol_kind=SymbolKind.File))
        ancestors_within_file.reverse()
        return [a.name for a in ancestors_within_file] + [self.name]

    def iter_children(self) -> Iterator[Self]:
        for c in self.symbol_root["children"]:
            yield self.__class__(c)

    def iter_ancestors(self, up_to_symbol_kind: SymbolKind | None = None) -> Iterator[Self]:
        """
        Iterate over all ancestors of the symbol, starting with the parent and going up to the root or
        the given symbol kind.

        :param up_to_symbol_kind: if provided, iteration will stop *before* the first ancestor of the given kind.
            A typical use case is to pass `SymbolKind.File` or `SymbolKind.Package`.
        """
        parent = self.get_parent()
        if parent is not None:
            if up_to_symbol_kind is None or parent.symbol_kind != up_to_symbol_kind:
                yield parent
                yield from parent.iter_ancestors(up_to_symbol_kind=up_to_symbol_kind)

    def get_parent(self) -> Self | None:
        parent_root = self.symbol_root.get("parent")
        if parent_root is None:
            return None
        return self.__class__(parent_root)

    def find(
        self,
        name_path_pattern: str,
        include_kinds: Sequence[SymbolKind] | None = None,
        exclude_kinds: Sequence[SymbolKind] | None = None,
    ) -> list[Self]:
        """
        Find all symbols within the symbol's subtree that match the given name path pattern.

        :param name_path_pattern: Regular expression pattern to match against symbol name paths (e.g., "get.*Mappings", ".*Action/.*").
            Uses standard Python/POSIX regex syntax. Case-insensitive by default.
        :param include_kinds: an optional sequence of ints representing the LSP symbol kind.
            If provided, only symbols of the given kinds will be included in the result.
        :param exclude_kinds: If provided, symbols of the given kinds will be excluded from the result.
        """
        result = []
        name_path_matcher = NamePathMatcher(name_path_pattern)

        def should_include(s: "LanguageServerSymbol") -> bool:
            if include_kinds is not None and s.symbol_kind not in include_kinds:
                return False
            if exclude_kinds is not None and s.symbol_kind in exclude_kinds:
                return False
            return name_path_matcher.matches_ls_symbol(s)

        def traverse(s: "LanguageServerSymbol") -> None:
            if should_include(s):
                result.append(s)
                return
            for c in s.iter_children():
                traverse(c)

        traverse(self)
        return result

    def to_dict(
        self,
        kind: bool = False,
        location: bool = False,
        depth: int = 0,
        include_body: bool = False,
        include_children_body: bool = False,
        include_relative_path: bool = True,
        child_inclusion_predicate: Callable[[Self], bool] | None = None,
    ) -> dict[str, Any]:
        """
        Converts the symbol to a dictionary.

        :param kind: whether to include the kind of the symbol
        :param location: whether to include the location of the symbol
        :param depth: the depth up to which to include child symbols (0 = do not include children)
        :param include_body: whether to include the body of the top-level symbol.
        :param include_children_body: whether to also include the body of the children.
            Note that the body of the children is part of the body of the parent symbol,
            so there is usually no need to set this to True unless you want process the output
            and pass the children without passing the parent body to the LM.
        :param include_relative_path: whether to include the relative path of the symbol in the location
            entry. Relative paths of the symbol's children are always excluded.
        :param child_inclusion_predicate: an optional predicate that decides whether a child symbol
            should be included.
        :return: a dictionary representation of the symbol
        """
        result: dict[str, Any] = {"name": self.name, "name_path": self.get_name_path()}

        if kind:
            result["kind"] = self.kind

        if location:
            result["location"] = self.location.to_dict(include_relative_path=include_relative_path)
            body_start_line, body_end_line = self.get_body_line_numbers()
            result["body_location"] = [body_start_line, body_end_line]

        if include_body:
            if self.body is None:
                log.warning("Requested body for symbol, but it is not present. The symbol might have been loaded with include_body=False.")
            result["body"] = self.body

        if child_inclusion_predicate is None:
            child_inclusion_predicate = lambda s: True

        def included_children(s: Self) -> list[dict[str, Any]]:
            children = []
            for c in s.iter_children():
                if not child_inclusion_predicate(c):
                    continue
                children.append(
                    c.to_dict(
                        kind=kind,
                        location=location,
                        depth=depth - 1,
                        child_inclusion_predicate=child_inclusion_predicate,
                        include_body=include_children_body,
                        include_children_body=include_children_body,
                        # all children have the same relative path as the parent
                        include_relative_path=False,
                    )
                )
            return children

        if depth > 0:
            children = included_children(self)
            if len(children) > 0:
                result["children"] = included_children(self)

        return result


@dataclass
class ReferenceInLanguageServerSymbol(ToStringMixin):
    """
    Represents the location of a reference to another symbol within a symbol/file.

    The contained symbol is the symbol within which the reference is located,
    not the symbol that is referenced.
    """

    symbol: LanguageServerSymbol
    """
    the symbol within which the reference is located
    """
    line: int
    """
    the line number in which the reference is located (0-based)
    """
    character: int
    """
    the column number in which the reference is located (0-based)
    """

    @classmethod
    def from_lsp_reference(cls, reference: LSPReferenceInSymbol) -> Self:
        return cls(symbol=LanguageServerSymbol(reference.symbol), line=reference.line, character=reference.character)

    def get_relative_path(self) -> str | None:
        return self.symbol.location.relative_path


class LanguageServerSymbolRetriever:
    def __init__(self, ls: SolidLanguageServer | LanguageServerManager, agent: Union["SerenaAgent", None] = None) -> None:
        """
        :param ls: the language server or language server manager to use for symbol retrieval and editing operations.
        :param agent: the agent to use (only needed for marking files as modified). You can pass None if you don't
            need an agent to be aware of file modifications performed by the symbol manager.
        """
        if isinstance(ls, SolidLanguageServer):
            ls_manager = LanguageServerManager({ls.language: ls})
        else:
            ls_manager = ls
        assert isinstance(ls_manager, LanguageServerManager)
        self._ls_manager: LanguageServerManager = ls_manager
        self.agent = agent

    def get_root_path(self) -> str:
        return self._ls_manager.get_root_path()

    def get_language_server(self, relative_path: str) -> SolidLanguageServer:
        return self._ls_manager.get_language_server(relative_path)

    def find(
        self,
        pattern: str,
        search_in: str | None = None,
        include_kinds: Sequence[SymbolKind] | None = None,
        exclude_kinds: Sequence[SymbolKind] | None = None,
    ) -> list[LanguageServerSymbol]:
        """
        Finds symbols using grep-style pattern matching.
        
        :param pattern: Regular expression to match symbol name paths (like grep pattern)
        :param search_in: Where to search - file path, directory, or regex pattern (like grep file argument)
        :param include_kinds: Optional list of symbol kinds to include
        :param exclude_kinds: Optional list of symbol kinds to exclude
        :return: List of matching symbols
        """
        
        symbols: list[LanguageServerSymbol] = []
        for lang_server in self._ls_manager.iter_language_servers():
            # Use the language server's search_symbols method
            # It handles file vs directory distinction internally
            symbol_roots = lang_server.search_symbols(
                name_path_regex=pattern,
                include_kinds=include_kinds,
                exclude_kinds=exclude_kinds,
                relative_path_regex=search_in
            )
            # Now we can directly use the filtered symbols without additional find() call
            for root in symbol_roots:
                symbols.extend(
                    LanguageServerSymbol(root).find(
                        pattern, include_kinds=include_kinds, exclude_kinds=exclude_kinds
                    )
                )
        return symbols

    def find_unique(
        self,
        name_path_pattern: str,
        include_kinds: Sequence[SymbolKind] | None = None,
        exclude_kinds: Sequence[SymbolKind] | None = None,
        within_relative_path: str | None = None,
    ) -> LanguageServerSymbol:
        symbol_candidates = self.find(
            name_path_pattern,
            search_in=within_relative_path,
            include_kinds=include_kinds,
            exclude_kinds=exclude_kinds,
        )
        if len(symbol_candidates) == 1:
            return symbol_candidates[0]
        elif len(symbol_candidates) == 0:
            raise ValueError(f"No symbol matching '{name_path_pattern}' found")
        else:
            # There are multiple candidates.
            # If only one of the candidates has the given pattern as its exact name path, return that one
            exact_matches = [s for s in symbol_candidates if s.get_name_path() == name_path_pattern]
            if len(exact_matches) == 1:
                return exact_matches[0]
            # otherwise, raise an error
            include_rel_path = within_relative_path is not None
            raise ValueError(
                f"Found multiple {len(symbol_candidates)} symbols matching '{name_path_pattern}'. "
                "They are: \n"
                + json.dumps([s.to_dict(kind=True, include_relative_path=include_rel_path) for s in symbol_candidates], indent=2)
            )

    def find_by_location(self, location: LanguageServerSymbolLocation) -> LanguageServerSymbol | None:
        if location.relative_path is None:
            return None
        lang_server = self.get_language_server(location.relative_path)
        document_symbols = lang_server.request_document_symbols(location.relative_path)
        for symbol_dict in document_symbols.iter_symbols():
            symbol = LanguageServerSymbol(symbol_dict)
            if symbol.location == location:
                return symbol
        return None

    def find_referencing_symbols(
        self,
        name_path: str,
        relative_file_path: str,
        include_body: bool = False,
        include_kinds: Sequence[SymbolKind] | None = None,
        exclude_kinds: Sequence[SymbolKind] | None = None,
    ) -> list[ReferenceInLanguageServerSymbol]:
        """
        Find all symbols that reference the specified symbol, which is assumed to be unique.

        :param name_path: the name path of the symbol to find. (While this can be a matching pattern, it should
            usually be the full path to ensure uniqueness.)
        :param relative_file_path: the relative path of the file in which the referenced symbol is defined.
        :param include_body: whether to include the body of all symbols in the result.
            Not recommended, as the referencing symbols will often be files, and thus the bodies will be very long.
        :param include_kinds: which kinds of symbols to include in the result.
        :param exclude_kinds: which kinds of symbols to exclude from the result.
        """
        symbol = self.find_unique(name_path, within_relative_path=relative_file_path)
        return self.find_referencing_symbols_by_location(
            symbol.location, include_body=include_body, include_kinds=include_kinds, exclude_kinds=exclude_kinds
        )

    def find_referencing_symbols_by_location(
        self,
        symbol_location: LanguageServerSymbolLocation,
        include_body: bool = False,
        include_kinds: Sequence[SymbolKind] | None = None,
        exclude_kinds: Sequence[SymbolKind] | None = None,
    ) -> list[ReferenceInLanguageServerSymbol]:
        """
        Find all symbols that reference the symbol at the given location.

        :param symbol_location: the location of the symbol for which to find references.
            Does not need to include an end_line, as it is unused in the search.
        :param include_body: whether to include the body of all symbols in the result.
            Not recommended, as the referencing symbols will often be files, and thus the bodies will be very long.
            Note: you can filter out the bodies of the children if you set include_children_body=False
            in the to_dict method.
        :param include_kinds: an optional sequence of ints representing the LSP symbol kind.
            If provided, only symbols of the given kinds will be included in the result.
        :param exclude_kinds: If provided, symbols of the given kinds will be excluded from the result.
            Takes precedence over include_kinds.
        :return: a list of symbols that reference the given symbol
        """
        if not symbol_location.has_position_in_file():
            raise ValueError("Symbol location does not contain a valid position in a file")
        assert symbol_location.relative_path is not None
        assert symbol_location.line is not None
        assert symbol_location.column is not None
        lang_server = self.get_language_server(symbol_location.relative_path)
        references = lang_server.request_referencing_symbols(
            relative_file_path=symbol_location.relative_path,
            line=symbol_location.line,
            column=symbol_location.column,
            include_imports=False,
            include_self=False,
            include_body=include_body,
            include_file_symbols=True,
        )

        if include_kinds is not None:
            references = [s for s in references if s.symbol["kind"] in include_kinds]

        if exclude_kinds is not None:
            references = [s for s in references if s.symbol["kind"] not in exclude_kinds]

        return [ReferenceInLanguageServerSymbol.from_lsp_reference(r) for r in references]

    def get_symbol_overview(self, relative_path: str, depth: int = 0) -> dict[str, list[dict]]:
        """
        :param relative_path: the path of the file or directory for which to get the symbol overview
        :param depth: the depth up to which to include child symbols (0 = only top-level symbols)
        :return: a mapping from file paths to lists of symbol dictionaries.
            For the case where a file is passed, the mapping will contain a single entry.
        """
        lang_server = self.get_language_server(relative_path)
        path_to_unified_symbols = lang_server.request_overview(relative_path)

        def child_inclusion_predicate(s: LanguageServerSymbol) -> bool:
            return not s.is_low_level()

        result = {}
        for file_path, unified_symbols in path_to_unified_symbols.items():
            symbols_in_file = []
            for us in unified_symbols:
                symbol = LanguageServerSymbol(us)
                symbols_in_file.append(
                    symbol.to_dict(
                        depth=depth,
                        kind=True,
                        include_relative_path=False,
                        location=False,
                        child_inclusion_predicate=child_inclusion_predicate,
                    )
                )
            result[file_path] = symbols_in_file

        return result

    def request_text_document_diagnostics(self, relative_path: str, timeout: float = 15.0) -> list[Diagnostic]:
        """
        Request diagnostics for the given file.

        :param relative_path: the relative path of the file to retrieve diagnostics for
        :param timeout: timeout in seconds for the request
        :return: a list of diagnostics for the file
        """
        lang_server = self.get_language_server(relative_path)
        return lang_server.request_text_document_diagnostics(relative_path, timeout=timeout)


class JetBrainsSymbol(Symbol):
    class SymbolDict(TypedDict):
        name_path: str
        relative_path: str
        type: str
        text_range: NotRequired[dict]
        body: NotRequired[str]
        children: NotRequired[list["JetBrainsSymbol.SymbolDict"]]

    def __init__(self, symbol_dict: SymbolDict, project: Project) -> None:
        """
        :param symbol_dict: dictionary as returned by the JetBrains plugin client.
        """
        self._project = project
        self._dict = symbol_dict
        self._cached_file_content: str | None = None
        self._cached_body_start_position: PositionInFile | None = None
        self._cached_body_end_position: PositionInFile | None = None

    def _tostring_includes(self) -> list[str]:
        return []

    def _tostring_additional_entries(self) -> dict[str, Any]:
        return dict(name_path=self.get_name_path(), relative_path=self.get_relative_path(), type=self._dict["type"])

    def get_name_path(self) -> str:
        return self._dict["name_path"]

    def get_relative_path(self) -> str:
        return self._dict["relative_path"]

    def get_file_content(self) -> str:
        if self._cached_file_content is None:
            path = os.path.join(self._project.project_root, self.get_relative_path())
            with open(path, encoding=self._project.project_config.encoding) as f:
                self._cached_file_content = f.read()
        return self._cached_file_content

    def is_position_in_file_available(self) -> bool:
        return "text_range" in self._dict

    def get_body_start_position(self) -> PositionInFile | None:
        if not self.is_position_in_file_available():
            return None
        if self._cached_body_start_position is None:
            pos = self._dict["text_range"]["start_pos"]
            line, col = pos["line"], pos["col"]
            self._cached_body_start_position = PositionInFile(line=line, col=col)
        return self._cached_body_start_position

    def get_body_end_position(self) -> PositionInFile | None:
        if not self.is_position_in_file_available():
            return None
        if self._cached_body_end_position is None:
            pos = self._dict["text_range"]["end_pos"]
            line, col = pos["line"], pos["col"]
            self._cached_body_end_position = PositionInFile(line=line, col=col)
        return self._cached_body_end_position

    def is_neighbouring_definition_separated_by_empty_line(self) -> bool:
        # NOTE: Symbol types cannot really be differentiated, because types are not handled in a language-agnostic way.
        return False
