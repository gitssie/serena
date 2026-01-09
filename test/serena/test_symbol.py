import pytest
import re

from serena.symbol import NamePathMatcher


class TestSymbolNameMatching:
    """
    Tests for NamePathMatcher with Linux grep-style regular expression support.
    The behavior should be consistent with DuckDB's regexp_matches() function.
    """
    
    def _create_assertion_error_message(
        self,
        regex_pattern: str,
        name_path: str,
        expected_result: bool,
        actual_result: bool,
    ) -> str:
        """Helper to create a detailed error message for assertions."""
        return (
            f"Pattern '{regex_pattern}' vs Name path '{name_path}'. "
            f"Expected: {expected_result}, Got: {actual_result}"
        )

    @pytest.mark.parametrize(
        "regex_pattern, name_path, expected",
        [
            # Basic substring matching (grep-style default behavior)
            pytest.param("foo", "foo", True, id="'foo' matches 'foo'"),
            pytest.param("foo", "MyClass/foo", True, id="'foo' matches 'MyClass/foo'"),
            pytest.param("foo", "MyClass/foobar", True, id="'foo' matches 'MyClass/foobar'"),
            pytest.param("foo", "MyClass/barfoo", True, id="'foo' matches 'MyClass/barfoo'"),
            pytest.param("foo", "MyClass/bar", False, id="'foo' does not match 'MyClass/bar'"),
            
            # Anchored matching
            pytest.param("^foo", "foo", True, id="'^foo' matches 'foo' at start"),
            pytest.param("^foo", "MyClass/foo", False, id="'^foo' does not match 'MyClass/foo'"),
            pytest.param("foo$", "MyClass/foo", True, id="'foo$' matches 'MyClass/foo' at end"),
            pytest.param("foo$", "MyClass/foobar", False, id="'foo$' does not match 'MyClass/foobar'"),
            pytest.param("^foo$", "foo", True, id="'^foo$' matches exactly 'foo'"),
            pytest.param("^foo$", "foobar", False, id="'^foo$' does not match 'foobar'"),
            
            # Path matching
            pytest.param("MyClass/foo", "MyClass/foo", True, id="'MyClass/foo' matches 'MyClass/foo'"),
            pytest.param("MyClass/foo", "com/example/MyClass/foo", True, id="'MyClass/foo' matches 'com/example/MyClass/foo'"),
            pytest.param("^MyClass/foo$", "MyClass/foo", True, id="'^MyClass/foo$' matches exactly 'MyClass/foo'"),
            pytest.param("^MyClass/foo$", "com/example/MyClass/foo", False, id="'^MyClass/foo$' does not match 'com/example/MyClass/foo'"),
            
            # Wildcard patterns (.*)
            # Wildcard patterns (.*)
            pytest.param("get.*", "getUser", True, id="'get.*' matches 'getUser'"),
            pytest.param("get.*", "MyClass/getUserById", True, id="'get.*' matches 'MyClass/getUserById'"),
            pytest.param("get.*", "setUser", False, id="'get.*' does not match 'setUser'"),
            pytest.param(".*Action/.*", "AdminAction/execute", True, id="'.*Action/.*' matches 'AdminAction/execute'"),
            pytest.param(".*Action/.*", "MyController/handle", False, id="'.*Action/.*' does not match 'MyController/handle'"),
            
            # Character class
            pytest.param("get[A-Z].*", "getUser", True, id="'get[A-Z].*' matches 'getUser'"),
            pytest.param("get[A-Z].*", "getuser", False, id="'get[A-Z].*' does not match 'getuser' (case-sensitive in character class)"),
            
            # Optional patterns
            pytest.param("handle?", "handle", True, id="'handle?' matches 'handle'"),
            pytest.param("handle?", "handler", True, id="'handle?' matches 'handler'"),
            
            # Case insensitivity (default)
            pytest.param("myclass", "MyClass/myMethod", True, id="'myclass' matches 'MyClass/myMethod' (case-insensitive)"),
        ],
    )
    def test_regex_matching(self, regex_pattern, name_path, expected):
        """Tests regex matching for symbol name paths."""
        matcher = NamePathMatcher(regex_pattern)
        result = matcher.matches_name_path(name_path)
        error_msg = self._create_assertion_error_message(regex_pattern, name_path, expected, result)
        assert result == expected, error_msg
    
    @pytest.mark.parametrize(
        "regex_pattern, symbol_name_path_parts, expected",
        [
            # Basic matching using matches_components
            pytest.param("foo", ["MyClass", "foo"], True, id="'foo' matches ['MyClass', 'foo']"),
            pytest.param("foo", ["MyClass", "bar"], False, id="'foo' does not match ['MyClass', 'bar']"),
            pytest.param("^MyClass/foo$", ["MyClass", "foo"], True, id="'^MyClass/foo$' matches ['MyClass', 'foo'] exactly"),
            pytest.param("^MyClass/foo$", ["com", "MyClass", "foo"], False, id="'^MyClass/foo$' does not match ['com', 'MyClass', 'foo']"),
        ],
    )
    def test_matches_components(self, regex_pattern, symbol_name_path_parts, expected):
        """Tests matches_components method (backward compatibility)."""
        matcher = NamePathMatcher(regex_pattern)
        result = matcher.matches_components(symbol_name_path_parts, None)
        name_path = "/".join(symbol_name_path_parts)
        error_msg = self._create_assertion_error_message(regex_pattern, name_path, expected, result)
        assert result == expected, error_msg
    
    def test_invalid_regex_pattern(self):
        """Tests that invalid regex patterns raise ValueError."""
        with pytest.raises(ValueError, match="Invalid regular expression pattern"):
            NamePathMatcher("[invalid")
    
    def test_case_sensitive_matching(self):
        """Tests case-sensitive matching."""
        matcher = NamePathMatcher("myclass", case_sensitive=True)
        assert matcher.matches_name_path("MyClass/myMethod") is False
        assert matcher.matches_name_path("myclass/myMethod") is True
