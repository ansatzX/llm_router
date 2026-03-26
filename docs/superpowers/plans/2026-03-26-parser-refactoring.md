# LLM Router Parser Module Refactoring Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refactor monolithic MCP parsing logic into a modular parser system with robust error handling

**Architecture:** Create separate parser module with data classes, error types, XML/JSON parsers, validator, and unified API. Follow TDD approach - write tests first, then implementation. Keep old parsing code until new system is validated.

**Tech Stack:** Python 3.10+, dataclasses, json_repair, pytest, re (regex)

**Spec Reference:** `docs/superpowers/specs/2026-03-26-parser-refactoring-design.md`

---

## File Structure

**New Files to Create:**
```
llm_router/parser/
├── __init__.py        (Public API: parse_tool_calls())
├── base.py            (ToolCall, ParseResult, ToolCallParser ABC)
├── errors.py          (ParseError, XMLParseError, JSONParseError, ValidationError)
├── xml_parser.py      (XMLParser implementation)
├── json_parser.py     (JSONParser implementation)
└── validator.py       (validate_tool_call, sanitize_arguments)

tests/
├── test_parser_base.py
├── test_xml_parser.py
├── test_json_parser.py
├── test_validator.py
└── test_parser_integration.py
```

**Files to Modify:**
```
llm_router/server.py        (Update to use parse_tool_calls())
llm_router/mcp_converter.py (Remove parsing functions, keep prompt generation)
```

---

## Task 1: Setup Test Infrastructure

**Files:**
- Create: `tests/__init__.py` (empty file for package)
- Create: `tests/conftest.py` (pytest fixtures)

- [ ] **Step 1: Create tests directory and init file**

```bash
mkdir -p tests
touch tests/__init__.py
```

- [ ] **Step 2: Create pytest configuration**

Create `tests/conftest.py`:
```python
"""Pytest configuration and fixtures for parser tests."""
import pytest
```

- [ ] **Step 3: Verify pytest works**

Run: `pytest --version`
Expected: pytest version output

- [ ] **Step 4: Commit test infrastructure**

```bash
git add tests/
git commit -m "test: add test infrastructure for parser module"
```

---

## Task 2: Create Error Types

**Files:**
- Create: `llm_router/parser/__init__.py` (empty for now)
- Create: `llm_router/parser/errors.py`
- Create: `tests/test_parser_errors.py`

- [ ] **Step 1: Write failing test for ParseError enum**

Create `tests/test_parser_errors.py`:
```python
"""Tests for parser error types."""
import pytest
from llm_router.parser.errors import ParseErrorType, ParseError


def test_parse_error_type_enum():
    """Test ParseErrorType enum values."""
    assert ParseErrorType.EMPTY_INPUT.value == "empty_input"
    assert ParseErrorType.INVALID_XML.value == "invalid_xml"
    assert ParseErrorType.INVALID_JSON.value == "invalid_json"
    assert ParseErrorType.MISSING_TOOL_NAME.value == "missing_tool_name"
    assert ParseErrorType.INVALID_ARGUMENTS.value == "invalid_arguments"


def test_parse_error_creation():
    """Test ParseError instantiation."""
    error = ParseError(
        ParseErrorType.INVALID_XML,
        "Test message",
        "context snippet"
    )
    assert error.error_type == ParseErrorType.INVALID_XML
    assert error.message == "Test message"
    assert error.context == "context snippet"
    assert "invalid_xml" in str(error)


def test_parse_error_str_representation():
    """Test ParseError string representation includes context."""
    error = ParseError(
        ParseErrorType.INVALID_JSON,
        "Malformed JSON",
        "long context string that should be truncated to 100 chars"
    )
    error_str = str(error)
    assert "[invalid_json]" in error_str
    assert "Malformed JSON" in error_str
    assert "context:" in error_str
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_parser_errors.py -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'llm_router.parser'"

- [ ] **Step 3: Create parser package directory**

```bash
mkdir -p llm_router/parser
touch llm_router/parser/__init__.py
```

- [ ] **Step 4: Implement error types**

Create `llm_router/parser/errors.py`:
```python
"""Parser error types and exceptions."""
from enum import Enum


class ParseErrorType(Enum):
    """Types of parsing errors."""
    EMPTY_INPUT = "empty_input"
    INVALID_XML = "invalid_xml"
    INVALID_JSON = "invalid_json"
    MISSING_TOOL_NAME = "missing_tool_name"
    INVALID_ARGUMENTS = "invalid_arguments"
    UNEXPECTED_FORMAT = "unexpected_format"
    SIZE_LIMIT_EXCEEDED = "size_limit_exceeded"


class ParseError(Exception):
    """Base exception for parsing errors."""

    def __init__(self, error_type: ParseErrorType, message: str, context: str = ""):
        self.error_type = error_type
        self.message = message
        self.context = context
        super().__init__(f"[{error_type.value}] {message}")

    def __str__(self):
        context_preview = self.context[:100] if self.context else ""
        return f"{super().__str__()} (context: {context_preview}...)"


class XMLParseError(ParseError):
    """XML parsing error."""

    def __init__(self, message: str, context: str = ""):
        super().__init__(ParseErrorType.INVALID_XML, message, context)


class JSONParseError(ParseError):
    """JSON parsing error."""

    def __init__(self, message: str, context: str = ""):
        super().__init__(ParseErrorType.INVALID_JSON, message, context)


class ValidationError(ParseError):
    """Validation error."""

    def __init__(self, message: str, context: str = ""):
        super().__init__(ParseErrorType.MISSING_TOOL_NAME, message, context)
```

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/test_parser_errors.py -v`
Expected: PASS (3 tests)

- [ ] **Step 6: Commit error types**

```bash
git add llm_router/parser/ tests/test_parser_errors.py
git commit -m "feat(parser): add error types for parsing"
```

---

## Task 3: Create Base Data Classes

**Files:**
- Create: `llm_router/parser/base.py`
- Create: `tests/test_parser_base.py`

- [ ] **Step 1: Write failing test for ToolCall dataclass**

Create `tests/test_parser_base.py`:
```python
"""Tests for parser base data classes."""
import pytest
from llm_router.parser.base import ToolCall, ParseResult


def test_tool_call_creation():
    """Test ToolCall dataclass creation."""
    tool_call = ToolCall(
        tool_name="get_weather",
        arguments={"location": "London"},
        server_name="tools"
    )
    assert tool_call.tool_name == "get_weather"
    assert tool_call.arguments == {"location": "London"}
    assert tool_call.server_name == "tools"


def test_tool_call_default_server_name():
    """Test ToolCall default server_name."""
    tool_call = ToolCall(
        tool_name="test_tool",
        arguments={}
    )
    assert tool_call.server_name == "tools"


def test_tool_call_to_openai_format():
    """Test ToolCall conversion to OpenAI format."""
    tool_call = ToolCall(
        tool_name="read_file",
        arguments={"path": "/test/file.txt"}
    )
    openai_format = tool_call.to_openai_format()

    assert openai_format["type"] == "function"
    assert openai_format["function"]["name"] == "read_file"
    assert openai_format["function"]["arguments"] == '{"path": "/test/file.txt"}'
    assert openai_format["id"].startswith("call_")


def test_parse_result_ok():
    """Test ParseResult.ok factory method."""
    tool_call = ToolCall(tool_name="test", arguments={})
    result = ParseResult.ok([tool_call])

    assert result.success is True
    assert len(result.tool_calls) == 1
    assert result.errors == []
    assert result.warnings == []


def test_parse_result_ok_with_warnings():
    """Test ParseResult.ok with warnings."""
    tool_call = ToolCall(tool_name="test", arguments={})
    result = ParseResult.ok([tool_call], warnings=["Test warning"])

    assert result.success is True
    assert result.warnings == ["Test warning"]


def test_parse_result_error():
    """Test ParseResult.error factory method."""
    result = ParseResult.error(["Error 1", "Error 2"])

    assert result.success is False
    assert result.tool_calls == []
    assert result.errors == ["Error 1", "Error 2"]
    assert result.warnings == []


def test_parse_result_error_with_warnings():
    """Test ParseResult.error with warnings."""
    result = ParseResult.error(
        ["Error message"],
        warnings=["Warning message"]
    )

    assert result.success is False
    assert result.errors == ["Error message"]
    assert result.warnings == ["Warning message"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_parser_base.py -v`
Expected: FAIL with "cannot import name 'ToolCall'"

- [ ] **Step 3: Implement base data classes**

Create `llm_router/parser/base.py`:
```python
"""Parser base classes and data structures."""
from dataclasses import dataclass
from typing import Optional
from abc import ABC, abstractmethod
import json
import uuid


@dataclass
class ToolCall:
    """Represents a parsed tool call."""
    tool_name: str
    arguments: dict
    server_name: str = "tools"

    def to_openai_format(self) -> dict:
        """Convert to OpenAI tool_calls format."""
        return {
            "id": f"call_{uuid.uuid4().hex[:8]}",
            "type": "function",
            "function": {
                "name": self.tool_name,
                "arguments": json.dumps(self.arguments, ensure_ascii=False)
            }
        }


@dataclass
class ParseResult:
    """Result of parsing tool calls."""
    success: bool
    tool_calls: list[ToolCall]
    errors: list[str]
    warnings: list[str]

    @classmethod
    def ok(cls, tool_calls: list[ToolCall], warnings: list[str] = None) -> 'ParseResult':
        """Create successful parse result."""
        return cls(
            success=True,
            tool_calls=tool_calls,
            errors=[],
            warnings=warnings or []
        )

    @classmethod
    def error(cls, errors: list[str], warnings: list[str] = None) -> 'ParseResult':
        """Create failed parse result."""
        return cls(
            success=False,
            tool_calls=[],
            errors=errors,
            warnings=warnings or []
        )


class ToolCallParser(ABC):
    """Abstract base class for tool call parsers."""

    @abstractmethod
    def parse(self, content: str) -> ParseResult:
        """Parse content and return result with all found tool calls."""
        pass

    @property
    @abstractmethod
    def format_name(self) -> str:
        """Parser format name for debugging."""
        pass
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_parser_base.py -v`
Expected: PASS (7 tests)

- [ ] **Step 5: Commit base classes**

```bash
git add llm_router/parser/base.py tests/test_parser_base.py
git commit -m "feat(parser): add base data classes ToolCall and ParseResult"
```

---

## Task 4: Create Validator

**Files:**
- Create: `llm_router/parser/validator.py`
- Create: `tests/test_validator.py`

- [ ] **Step 1: Write failing test for validator**

Create `tests/test_validator.py`:
```python
"""Tests for tool call validator."""
import pytest
from llm_router.parser.base import ToolCall
from llm_router.parser.validator import validate_tool_call, sanitize_arguments


def test_validate_valid_tool_call():
    """Test validation of valid tool call."""
    tool_call = ToolCall(
        tool_name="test_tool",
        arguments={"param": "value"},
        server_name="test_server"
    )
    is_valid, warning = validate_tool_call(tool_call)

    assert is_valid is True
    assert warning is None


def test_validate_empty_tool_name():
    """Test validation fails for empty tool name."""
    tool_call = ToolCall(
        tool_name="",
        arguments={}
    )
    is_valid, warning = validate_tool_call(tool_call)

    assert is_valid is False
    assert "empty" in warning.lower()


def test_validate_non_string_tool_name():
    """Test validation fails for non-string tool name."""
    tool_call = ToolCall(
        tool_name=123,  # type: ignore
        arguments={}
    )
    is_valid, warning = validate_tool_call(tool_call)

    assert is_valid is False
    assert "string" in warning.lower()


def test_validate_non_dict_arguments():
    """Test validation fails for non-dict arguments."""
    tool_call = ToolCall(
        tool_name="test",
        arguments="not a dict"  # type: ignore
    )
    is_valid, warning = validate_tool_call(tool_call)

    assert is_valid is False
    assert "dict" in warning.lower()


def test_validate_whitespace_tool_name():
    """Test validation fails for whitespace-only tool name."""
    tool_call = ToolCall(
        tool_name="   ",
        arguments={}
    )
    is_valid, warning = validate_tool_call(tool_call)

    assert is_valid is False
    assert "whitespace" in warning.lower()


def test_validate_none_values_in_arguments():
    """Test warning for None values in arguments."""
    tool_call = ToolCall(
        tool_name="test",
        arguments={"param1": "value", "param2": None}
    )
    is_valid, warning = validate_tool_call(tool_call)

    assert is_valid is True
    assert warning is not None
    assert "None" in warning


def test_sanitize_arguments_removes_none():
    """Test sanitize_arguments removes None values."""
    args = {"a": 1, "b": None, "c": "value", "d": None}
    sanitized = sanitize_arguments(args)

    assert sanitized == {"a": 1, "c": "value"}


def test_sanitize_arguments_empty_dict():
    """Test sanitize_arguments with empty dict."""
    assert sanitize_arguments({}) == {}


def test_sanitize_arguments_non_dict():
    """Test sanitize_arguments returns empty dict for non-dict."""
    assert sanitize_arguments("not a dict") == {}
    assert sanitize_arguments(None) == {}
    assert sanitize_arguments([1, 2, 3]) == {}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_validator.py -v`
Expected: FAIL with "cannot import name 'validate_tool_call'"

- [ ] **Step 3: Implement validator**

Create `llm_router/parser/validator.py`:
```python
"""Tool call validation functions."""
from typing import Tuple, Optional
from .base import ToolCall


def validate_tool_call(tool_call: ToolCall) -> Tuple[bool, Optional[str]]:
    """
    Validate tool call data completeness.

    Args:
        tool_call: Tool call to validate

    Returns:
        Tuple of (is_valid, warning_message)
        - is_valid: True if tool call can be used
        - warning_message: Warning if valid but has issues, error if invalid
    """
    warnings = []

    # Check tool_name
    if not tool_call.tool_name:
        return False, "Tool name is empty"

    if not isinstance(tool_call.tool_name, str):
        return False, f"Tool name must be string, got {type(tool_call.tool_name).__name__}"

    if not tool_call.tool_name.strip():
        return False, "Tool name is whitespace only"

    # Check arguments
    if not isinstance(tool_call.arguments, dict):
        return False, f"Arguments must be dict, got {type(tool_call.arguments).__name__}"

    # Check for None values in arguments (warning, not error)
    if tool_call.arguments:
        none_keys = [k for k, v in tool_call.arguments.items() if v is None]
        if none_keys:
            warnings.append(f"Arguments contain None values for keys: {none_keys}")

    # Return result
    warning_msg = "; ".join(warnings) if warnings else None
    return True, warning_msg


def sanitize_arguments(arguments: dict) -> dict:
    """
    Clean arguments by removing None values.

    Args:
        arguments: Raw arguments dict

    Returns:
        Cleaned arguments dict with None values removed
    """
    if not isinstance(arguments, dict):
        return {}

    return {k: v for k, v in arguments.items() if v is not None}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_validator.py -v`
Expected: PASS (9 tests)

- [ ] **Step 5: Commit validator**

```bash
git add llm_router/parser/validator.py tests/test_validator.py
git commit -m "feat(parser): add tool call validator"
```

---

## Task 5: Create XML Parser

**Files:**
- Create: `llm_router/parser/xml_parser.py`
- Create: `tests/test_xml_parser.py`

- [ ] **Step 1: Write failing test for XML parser - valid MCP XML**

Create `tests/test_xml_parser.py`:
```python
"""Tests for XML parser."""
import pytest
from llm_router.parser.xml_parser import XMLParser
from llm_router.parser.base import ToolCall


def test_xml_parser_format_name():
    """Test XMLParser format name."""
    parser = XMLParser()
    assert parser.format_name == "xml"


def test_parse_valid_mcp_xml():
    """Test parsing valid MCP XML format."""
    parser = XMLParser()
    content = """
<use_mcp_tool>
<server_name>tools</server_name>
<tool_name>get_weather</tool_name>
<arguments>
{"location": "London"}
</arguments>
</use_mcp_tool>
"""
    result = parser.parse(content)

    assert result.success is True
    assert len(result.tool_calls) == 1

    tool_call = result.tool_calls[0]
    assert tool_call.tool_name == "get_weather"
    assert tool_call.arguments == {"location": "London"}
    assert tool_call.server_name == "tools"


def test_parse_mcp_xml_missing_server_name():
    """Test MCP XML with missing server_name uses default."""
    parser = XMLParser()
    content = """
<use_mcp_tool>
<tool_name>test_tool</tool_name>
<arguments>{"param": "value"}</arguments>
</use_mcp_tool>
"""
    result = parser.parse(content)

    assert result.success is True
    assert result.tool_calls[0].server_name == "tools"


def test_parse_tool_call_xml_format():
    """Test parsing TOOL_CALL XML format."""
    parser = XMLParser()
    content = """[TOOL_CALL]
{"name": "read_file", "arguments": {"path": "/test.txt"}}
[/TOOL_CALL]"""
    result = parser.parse(content)

    assert result.success is True
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].tool_name == "read_file"


def test_parse_simple_tool_xml():
    """Test parsing simple <tool> XML format."""
    parser = XMLParser()
    content = """
<tool>
<function_name>calculate</function_name>
<parameters>{"expr": "2+2"}</parameters>
</tool>
"""
    result = parser.parse(content)

    assert result.success is True
    assert result.tool_calls[0].tool_name == "calculate"
    assert result.tool_calls[0].arguments == {"expr": "2+2"}


def test_parse_multiple_tool_calls():
    """Test parsing multiple tool calls in same format."""
    parser = XMLParser()
    content = """
<use_mcp_tool>
<tool_name>tool1</tool_name>
<arguments>{"a": 1}</arguments>
</use_mcp_tool>
<use_mcp_tool>
<tool_name>tool2</tool_name>
<arguments>{"b": 2}</arguments>
</use_mcp_tool>
"""
    result = parser.parse(content)

    assert result.success is True
    assert len(result.tool_calls) == 2
    assert result.tool_calls[0].tool_name == "tool1"
    assert result.tool_calls[1].tool_name == "tool2"


def test_parse_malformed_json_arguments():
    """Test parsing with malformed JSON uses json_repair."""
    parser = XMLParser()
    content = """
<use_mcp_tool>
<tool_name>test</tool_name>
<arguments>{location: "London", unit: "celsius"}</arguments>
</use_mcp_tool>
"""
    result = parser.parse(content)

    assert result.success is True
    assert result.tool_calls[0].arguments == {"location": "London", "unit": "celsius"}


def test_parse_missing_tool_name():
    """Test parsing XML with missing tool_name (format fallback to JSON)."""
    parser = XMLParser()
    content = """
<use_mcp_tool>
<arguments>{"param": "value"}</arguments>
</use_mcp_tool>
"""
    result = parser.parse(content)

    # XMLParser should fail (no valid tool calls)
    # But parse_tool_calls() will try JSON parser next
    assert result.success is False
    assert len(result.tool_calls) == 0
    assert len(result.errors) > 0


def test_parse_no_xml_found():
    """Test parsing content with no XML returns empty result."""
    parser = XMLParser()
    result = parser.parse("This is just plain text with no XML")

    assert result.success is False
    assert len(result.tool_calls) == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_xml_parser.py -v`
Expected: FAIL with "cannot import name 'XMLParser'"

- [ ] **Step 3: Implement XML parser**

Create `llm_router/parser/xml_parser.py`:
```python
"""XML format parser for tool calls."""
import re
import json
from typing import Optional
from .base import ToolCallParser, ToolCall, ParseResult
from .errors import XMLParseError, ValidationError
from .validator import validate_tool_call


class XMLParser(ToolCallParser):
    """Parser for XML format tool calls."""

    # XML patterns for different formats
    PATTERNS = {
        'mcp': r'<use_mcp_tool>(.*?)</use_mcp_tool>',
        'tool_call': r'\[TOOL_CALL\](.*?)\[/TOOL_CALL\]',
        'simple': r'<tool>(.*?)</tool>',
    }

    # Tag name fallbacks
    TOOL_NAME_TAGS = ['tool_name', 'function_name', 'name', 'function', 'action']
    ARGUMENT_TAGS = ['arguments', 'parameters', 'input', 'params', 'args']

    @property
    def format_name(self) -> str:
        return "xml"

    def parse(self, content: str) -> ParseResult:
        """Parse XML format tool calls."""
        # Size limit
        MAX_SIZE = 10 * 1024 * 1024
        if len(content) > MAX_SIZE:
            content = content[:MAX_SIZE]

        # Try each format in priority order
        for format_type, pattern in self.PATTERNS.items():
            matches = re.findall(pattern, content, re.DOTALL)

            if not matches:
                continue

            # Found this format, try to parse all occurrences
            tool_calls = []
            errors = []
            warnings = []

            for match_content in matches:
                try:
                    tool_call = self._parse_single_xml(
                        match_content,
                        has_server_name=(format_type == 'mcp')
                    )

                    if tool_call:
                        # Validate
                        is_valid, warning = validate_tool_call(tool_call)
                        if warning:
                            warnings.append(warning)

                        if is_valid:
                            tool_calls.append(tool_call)

                except (XMLParseError, ValidationError) as e:
                    errors.append(str(e))
                except Exception as e:
                    errors.append(f"Unexpected XML parse error: {e}")

            # Only return if we have VALID tool calls from this format
            if tool_calls:
                return ParseResult.ok(tool_calls, warnings)

            # This format matched but all tool calls failed validation
            # Continue to next format (don't return error yet)

        # No tool calls found in any XML format
        return ParseResult.error(["No valid XML tool calls found"])

    def _parse_single_xml(
        self,
        content: str,
        has_server_name: bool
    ) -> Optional[ToolCall]:
        """Parse single XML block."""
        # Extract server_name (if present)
        if has_server_name:
            server_name = self._extract_tag(content, ['server_name']) or "tools"
        else:
            server_name = "tools"

        # Extract tool_name (required)
        tool_name = self._extract_tag(content, self.TOOL_NAME_TAGS)
        if not tool_name:
            raise ValidationError(
                "Missing tool_name in XML",
                context=content[:100]
            )

        # Extract arguments
        args_text = self._extract_tag(content, self.ARGUMENT_TAGS)
        arguments = self._parse_arguments(args_text) if args_text else {}

        return ToolCall(
            tool_name=tool_name,
            arguments=arguments,
            server_name=server_name
        )

    def _extract_tag(self, content: str, tag_names: list) -> Optional[str]:
        """Extract content from first matching tag."""
        for tag in tag_names:
            match = re.search(rf'<{tag}>(.*?)</{tag}>', content, re.DOTALL)
            if match:
                return match.group(1).strip()
        return None

    def _parse_arguments(self, args_text: str) -> dict:
        """Parse JSON arguments with repair fallback."""
        from json_repair import repair_json

        try:
            return json.loads(args_text.strip())
        except json.JSONDecodeError:
            try:
                repaired = repair_json(args_text.strip())
                return json.loads(repaired)
            except Exception as e:
                raise XMLParseError(
                    f"Failed to parse arguments JSON: {e}",
                    context=args_text[:100]
                )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_xml_parser.py -v`
Expected: PASS (10 tests)

- [ ] **Step 5: Commit XML parser**

```bash
git add llm_router/parser/xml_parser.py tests/test_xml_parser.py
git commit -m "feat(parser): add XML parser for MCP/tool/simple formats"
```

---

## Task 6: Create JSON Parser

**Files:**
- Create: `llm_router/parser/json_parser.py`
- Create: `tests/test_json_parser.py`

- [ ] **Step 1: Write failing test for JSON parser**

Create `tests/test_json_parser.py`:
```python
"""Tests for JSON parser."""
import pytest
from llm_router.parser.json_parser import JSONParser


def test_json_parser_format_name():
    """Test JSONParser format name."""
    parser = JSONParser()
    assert parser.format_name == "json"


def test_parse_valid_json_with_name():
    """Test parsing valid JSON with 'name' field."""
    parser = JSONParser()
    content = '{"name": "get_weather", "arguments": {"location": "Tokyo"}}'
    result = parser.parse(content)

    assert result.success is True
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].tool_name == "get_weather"
    assert result.tool_calls[0].arguments == {"location": "Tokyo"}


def test_parse_valid_json_with_tool():
    """Test parsing valid JSON with 'tool' field."""
    parser = JSONParser()
    content = '{"tool": "read_file", "input": {"path": "/test.txt"}}'
    result = parser.parse(content)

    assert result.success is True
    assert result.tool_calls[0].tool_name == "read_file"
    assert result.tool_calls[0].arguments == {"path": "/test.txt"}


def test_parse_multiple_json_objects():
    """Test parsing multiple JSON objects."""
    parser = JSONParser()
    content = '''
First: {"name": "tool1", "arguments": {"a": 1}}
Second: {"name": "tool2", "arguments": {"b": 2}}
'''
    result = parser.parse(content)

    assert result.success is True
    assert len(result.tool_calls) == 2
    assert result.tool_calls[0].tool_name == "tool1"
    assert result.tool_calls[1].tool_name == "tool2"


def test_parse_malformed_json():
    """Test parsing malformed JSON uses json_repair."""
    parser = JSONParser()
    content = '{name: "test_tool", arguments: {param: "value"}}'
    result = parser.parse(content)

    assert result.success is True
    assert result.tool_calls[0].tool_name == "test_tool"


def test_parse_nested_json_string():
    """Test parsing JSON with nested JSON string in arguments."""
    parser = JSONParser()
    content = '{"name": "test", "arguments": "{\\"nested\\": \\"value\\"}"}'
    result = parser.parse(content)

    assert result.success is True
    assert result.tool_calls[0].arguments == {"nested": "value"}


def test_parse_missing_name_field():
    """Test parsing JSON without name/tool field fails."""
    parser = JSONParser()
    content = '{"some_field": "value"}'
    result = parser.parse(content)

    assert result.success is False
    assert len(result.tool_calls) == 0


def test_parse_non_object_json():
    """Test parsing non-object JSON fails."""
    parser = JSONParser()
    content = '["not", "an", "object"]'
    result = parser.parse(content)

    assert result.success is False


def test_parse_no_json_found():
    """Test parsing content with no JSON returns empty result."""
    parser = JSONParser()
    result = parser.parse("This is just plain text")

    assert result.success is False
    assert len(result.tool_calls) == 0


def test_parse_ignores_non_tool_json():
    """Test that JSON without tool-related fields is ignored."""
    parser = JSONParser()
    content = '''
{"config": {"setting": "value"}}
{"name": "real_tool", "arguments": {"x": 1}}
{"metadata": "ignored"}
'''
    result = parser.parse(content)

    assert result.success is True
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].tool_name == "real_tool"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_json_parser.py -v`
Expected: FAIL with "cannot import name 'JSONParser'"

- [ ] **Step 3: Implement JSON parser**

Create `llm_router/parser/json_parser.py`:
```python
"""JSON format parser for tool calls."""
import json
from typing import Optional
from .base import ToolCallParser, ToolCall, ParseResult
from .errors import JSONParseError, ValidationError
from .validator import validate_tool_call


class JSONParser(ToolCallParser):
    """Parser for JSON format tool calls."""

    @property
    def format_name(self) -> str:
        return "json"

    def parse(self, content: str) -> ParseResult:
        """Parse JSON format tool calls."""
        # Size limit
        MAX_SIZE = 10 * 1024 * 1024
        if len(content) > MAX_SIZE:
            content = content[:MAX_SIZE]

        tool_calls = []
        errors = []
        warnings = []

        # Extract all JSON objects
        json_objects = self._extract_json_objects(content)

        for json_str in json_objects:
            try:
                tool_call = self._parse_single_json(json_str)

                if tool_call:
                    # Validate
                    is_valid, warning = validate_tool_call(tool_call)
                    if warning:
                        warnings.append(warning)

                    if is_valid:
                        tool_calls.append(tool_call)

            except (JSONParseError, ValidationError) as e:
                errors.append(str(e))
            except Exception as e:
                errors.append(f"Unexpected JSON parse error: {e}")

        if tool_calls:
            return ParseResult.ok(tool_calls, warnings)
        else:
            if errors:
                return ParseResult.error(errors, warnings)
            else:
                return ParseResult.error(["No JSON tool calls found"], warnings)

    def _extract_json_objects(self, content: str) -> list:
        """Extract all potential JSON objects from content."""
        objects = []
        i = 0

        while i < len(content):
            if content[i] == '{':
                # Find matching closing brace
                depth = 1
                start = i
                i += 1

                while i < len(content) and depth > 0:
                    if content[i] == '{':
                        depth += 1
                    elif content[i] == '}':
                        depth -= 1
                    i += 1

                if depth == 0:
                    json_candidate = content[start:i]
                    # Filter: must contain tool-related fields
                    if '"name"' in json_candidate or '"tool"' in json_candidate:
                        objects.append(json_candidate)
            else:
                i += 1

        return objects

    def _parse_single_json(self, json_str: str) -> Optional[ToolCall]:
        """Parse single JSON object."""
        from json_repair import repair_json

        # Parse JSON
        try:
            obj = json.loads(json_str)
        except json.JSONDecodeError:
            try:
                repaired = repair_json(json_str)
                obj = json.loads(repaired)
            except Exception as e:
                raise JSONParseError(
                    f"Failed to parse JSON: {e}",
                    context=json_str[:100]
                )

        # Validate it's an object
        if not isinstance(obj, dict):
            raise JSONParseError(
                "JSON is not an object",
                context=json_str[:100]
            )

        # Extract tool_name (try multiple field names)
        tool_name = (
            obj.get("name") or
            obj.get("tool") or
            obj.get("tool_name")
        )

        if not tool_name:
            raise ValidationError(
                "Missing tool name in JSON",
                context=json_str[:100]
            )

        # Extract arguments (try multiple field names)
        arguments = (
            obj.get("arguments") or
            obj.get("input") or
            obj.get("params") or
            {}
        )

        # If arguments is a string, parse it
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except:
                try:
                    from json_repair import repair_json
                    repaired = repair_json(arguments)
                    arguments = json.loads(repaired)
                except:
                    arguments = {}

        # Ensure arguments is a dict
        if not isinstance(arguments, dict):
            arguments = {}

        return ToolCall(
            tool_name=str(tool_name),
            arguments=arguments,
            server_name="tools"
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_json_parser.py -v`
Expected: PASS (10 tests)

- [ ] **Step 5: Commit JSON parser**

```bash
git add llm_router/parser/json_parser.py tests/test_json_parser.py
git commit -m "feat(parser): add JSON parser for tool calls"
```

---

## Task 7: Create Public API

**Files:**
- Modify: `llm_router/parser/__init__.py`
- Create: `tests/test_parser_integration.py`

- [ ] **Step 1: Write failing test for public API**

Create `tests/test_parser_integration.py`:
```python
"""Integration tests for parser public API."""
import pytest
from llm_router.parser import parse_tool_calls, ToolCall, ParseResult


def test_parse_tool_calls_basic():
    """Test basic parse_tool_calls usage."""
    content = """
<use_mcp_tool>
<tool_name>test_tool</tool_name>
<arguments>{"param": "value"}</arguments>
</use_mcp_tool>
"""
    result = parse_tool_calls(content)

    assert isinstance(result, ParseResult)
    assert result.success is True
    assert len(result.tool_calls) == 1
    assert isinstance(result.tool_calls[0], ToolCall)


def test_parse_tool_calls_with_reasoning_content():
    """Test parsing with reasoning_content merged."""
    content = "Main response"
    reasoning = """
<use_mcp_tool>
<tool_name>hidden_tool</tool_name>
<arguments>{}</arguments>
</use_mcp_tool>
"""
    result = parse_tool_calls(content, reasoning)

    assert result.success is True
    assert result.tool_calls[0].tool_name == "hidden_tool"


def test_parse_tool_calls_format_priority():
    """Test XML parser has priority over JSON."""
    content = """
<use_mcp_tool>
<tool_name>xml_tool</tool_name>
<arguments>{}</arguments>
</use_mcp_tool>
{"name": "json_tool", "arguments": {}}
"""
    result = parse_tool_calls(content)

    # Should find XML first and return it
    assert result.success is True
    assert result.tool_calls[0].tool_name == "xml_tool"


def test_parse_tool_calls_json_fallback():
    """Test JSON parser is used when XML fails validation."""
    content = '{"name": "json_only_tool", "arguments": {"x": 1}}'
    result = parse_tool_calls(content)

    assert result.success is True
    assert result.tool_calls[0].tool_name == "json_only_tool"


def test_parse_tool_calls_format_fallback():
    """Test fallback to JSON when XML matches but validation fails."""
    content = """
<use_mcp_tool>
<arguments>{"x": 1}</arguments>
</use_mcp_tool>
{"name": "json_tool", "arguments": {"y": 2}}
"""
    result = parse_tool_calls(content)

    # Should fail XML (missing tool_name), succeed with JSON
    assert result.success is True
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].tool_name == "json_tool"


def test_parse_tool_calls_prefer_json():
    """Test prefer_format parameter."""
    content = """
{"name": "json_tool", "arguments": {}}
<use_mcp_tool>
<tool_name>xml_tool</tool_name>
<arguments>{}</arguments>
</use_mcp_tool>
"""
    result = parse_tool_calls(content, prefer_format="json")

    # Should try JSON first
    assert result.success is True
    assert result.tool_calls[0].tool_name == "json_tool"


def test_parse_tool_calls_no_tool_calls():
    """Test parsing content with no tool calls."""
    result = parse_tool_calls("Just regular text")

    assert result.success is False
    assert len(result.tool_calls) == 0
    assert len(result.errors) > 0


def test_parse_tool_calls_error_aggregation():
    """Test errors are aggregated from all parsers."""
    content = '{"invalid": "json"}'  # Missing name field

    result = parse_tool_calls(content)

    assert result.success is False
    assert len(result.errors) > 0


def test_parse_tool_calls_empty_content():
    """Test parsing empty content."""
    result = parse_tool_calls("")

    assert result.success is False


def test_exports():
    """Test that public API exports are correct."""
    from llm_router.parser import ToolCall, ParseResult, ParseError, parse_tool_calls

    assert callable(parse_tool_calls)
    assert ToolCall is not None
    assert ParseResult is not None
    assert ParseError is not None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_parser_integration.py -v`
Expected: FAIL with "cannot import name 'parse_tool_calls'"

- [ ] **Step 3: Implement public API**

Update `llm_router/parser/__init__.py`:
```python
"""
Parser module for extracting tool calls from LLM responses.

Public API:
- parse_tool_calls(): Main entry point for parsing
- ToolCall: Data class for tool call information
- ParseResult: Result container with tool_calls/errors/warnings
- ParseError: Base exception for parsing errors
"""

from .base import ToolCall, ParseResult
from .errors import ParseError

__all__ = [
    'ToolCall',
    'ParseResult',
    'ParseError',
    'parse_tool_calls',
]


def parse_tool_calls(
    content: str,
    reasoning_content: str = "",
    prefer_format: str = None
) -> ParseResult:
    """
    Parse tool calls from LLM response content.

    Args:
        content: Main response content
        reasoning_content: Reasoning content (MiroThinker may put tool calls here)
        prefer_format: Optional hint - "xml" or "json" to try that parser first

    Returns:
        ParseResult with success status, tool_calls, errors, and warnings

    Note:
        - Content and reasoning_content are merged before parsing
        - Returns first successful parser result (partial success is success)
        - Errors/warnings aggregated from failed parsers if all fail
    """
    from .xml_parser import XMLParser
    from .json_parser import JSONParser

    # Merge content
    full_content = f"{content}\n{reasoning_content}" if reasoning_content else content

    # Determine parser order
    if prefer_format == "json":
        parsers = [JSONParser(), XMLParser()]
    else:  # Default: XML first
        parsers = [XMLParser(), JSONParser()]

    # Try each parser
    all_errors = []
    all_warnings = []

    for parser in parsers:
        result = parser.parse(full_content)

        # If parser found any tool calls, return immediately
        if result.tool_calls:
            return result

        # Otherwise, collect errors and try next parser
        all_errors.extend(result.errors)
        all_warnings.extend(result.warnings)

    # All parsers failed to find tool calls
    return ParseResult.error(
        errors=all_errors or ["No tool calls found in any format"],
        warnings=all_warnings
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_parser_integration.py -v`
Expected: PASS (10 tests)

- [ ] **Step 5: Commit public API**

```bash
git add llm_router/parser/__init__.py tests/test_parser_integration.py
git commit -m "feat(parser): add public API parse_tool_calls()"
```

---

## Task 8: Integration - Update server.py

**Files:**
- Modify: `llm_router/server.py` (lines 74-86)

- [ ] **Step 1: Update imports in server.py**

Open `llm_router/server.py` and add import at top:
```python
from llm_router.parser import parse_tool_calls
```

- [ ] **Step 2: Replace old parsing logic**

Find lines 74-86 in `llm_router/server.py` and replace:
```python
# Old code (DELETE):
tool_calls = mcp_to_openai_tool_calls(response_text)
if not tool_calls and reasoning_text:
    tool_calls = mcp_to_openai_tool_calls(reasoning_text)

# New code (ADD):
parse_result = parse_tool_calls(response_text, reasoning_text)

# Log warnings
for warning in parse_result.warnings:
    logger.warning(f"Parse warning: {warning}")

# Log errors (don't interrupt flow)
for error in parse_result.errors:
    logger.error(f"Parse error: {error}")
```

- [ ] **Step 3: Update response building**

Find line 99 and replace:
```python
# Old code (DELETE):
if tool_calls:
    response["choices"][0]["message"]["tool_calls"] = tool_calls

# New code (ADD):
if parse_result.success:
    tool_calls_openai = [tc.to_openai_format() for tc in parse_result.tool_calls]
    response["choices"][0]["message"]["tool_calls"] = tool_calls_openai
```

- [ ] **Step 4: Update finish_reason**

Find line 92 and replace:
```python
# Old code (DELETE):
"finish_reason": "tool_calls" if tool_calls else choice.get("finish_reason", "stop")

# New code (ADD):
"finish_reason": "tool_calls" if parse_result.success else choice.get("finish_reason", "stop")
```

- [ ] **Step 5: Remove old import**

Remove this import from server.py:
```python
from .mcp_converter import (
    mcp_to_openai_tool_calls,  # DELETE this line
    ...
)
```

- [ ] **Step 6: Test with curl**

Start router:
```bash
cd D:/infer/llm_router
uv run llm-router --debug
```

Test request:
```bash
curl http://localhost:5001/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mirothinker-1.7-mini",
    "messages": [{"role": "user", "content": "What is the weather in London?"}],
    "tools": [{"type": "function", "function": {"name": "get_weather", "parameters": {"type": "object", "properties": {"location": {"type": "string"}}, "required": ["location"]}}}]
  }'
```

Expected: Response with tool_calls in OpenAI format

- [ ] **Step 7: Commit server integration**

```bash
git add llm_router/server.py
git commit -m "feat(server): integrate new parser module"
```

---

## Task 9: Cleanup - Remove old parsing code

**Files:**
- Modify: `llm_router/mcp_converter.py`

- [ ] **Step 1: Identify code to remove**

Open `llm_router/mcp_converter.py` and find these functions to DELETE:
- `extract_tool_calls_from_content()`
- `mcp_to_openai_tool_calls()`
- `strip_mcp_tags()`
- `_extract_tool_name_from_xml()`
- `_extract_arguments_from_xml()`
- `_extract_json_tool_calls()`
- `_extract_tag_content()`
- `_try_parse_json()`
- `_parse_arguments()`

Keep ONLY:
- `generate_mcp_system_prompt()`
- Tag name constants at top if used by `generate_mcp_system_prompt()`

- [ ] **Step 2: Remove parsing functions**

Delete all parsing-related functions, keeping only `generate_mcp_system_prompt()`.

Final `llm_router/mcp_converter.py` should be ~100 lines.

- [ ] **Step 3: Update imports**

Remove unused imports from `mcp_converter.py`:
```python
import uuid  # DELETE if not used
```

- [ ] **Step 4: Test that router still works**

Run integration test again:
```bash
curl http://localhost:5001/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mirothinker-1.7-mini",
    "messages": [{"role": "user", "content": "Test"}]
  }'
```

Expected: Normal response

- [ ] **Step 5: Commit cleanup**

```bash
git add llm_router/mcp_converter.py
git commit -m "refactor: remove old parsing code from mcp_converter"
```

---

## Task 10: Final Testing and Validation

**Files:**
- Run: All tests
- Run: Manual integration tests

- [ ] **Step 1: Run all parser tests**

Run: `pytest tests/ -v`
Expected: All tests PASS

- [ ] **Step 2: Check test coverage**

Run: `pytest tests/ --cov=llm_router/parser --cov-report=term-missing`
Expected: >90% coverage for parser module

- [ ] **Step 3: Test real MiroThinker integration**

Test 1 - Tool call:
```bash
curl http://localhost:5001/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mirothinker-1.7-mini",
    "messages": [{"role": "user", "content": "What is the weather in Tokyo?"}],
    "tools": [{"type": "function", "function": {"name": "get_weather", "description": "Get weather", "parameters": {"type": "object", "properties": {"location": {"type": "string"}}, "required": ["location"]}}}]
  }'
```

Expected: Valid tool_calls response

Test 2 - Normal chat:
```bash
curl http://localhost:5001/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mirothinker-1.7-mini",
    "messages": [{"role": "user", "content": "Say hello"}]
  }'
```

Expected: Normal text response

- [ ] **Step 4: Test error handling**

Test malformed tool call:
```bash
curl http://localhost:5001/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mirothinker-1.7-mini",
    "messages": [{"role": "user", "content": "Use tool"}],
    "tools": [{"type": "function", "function": {"name": "bad_tool", "parameters": {}}}]
  }'
```

Expected: Server doesn't crash, logs errors appropriately

- [ ] **Step 5: Check logs**

Check `llm_router.log` if debug mode is enabled:
- Should see parse errors/warnings logged
- Should see successful parse messages

- [ ] **Step 6: Final commit**

```bash
git add -A
git commit -m "test: validate parser module integration"
```

---

## Rollback Plan

If integration fails:

1. **Revert server.py changes:**
```bash
git checkout HEAD~2 -- llm_router/server.py
git commit -m "rollback: revert server.py to old parsing"
```

2. **Keep parser module for future use:**
Parser module is isolated and doesn't affect existing code if not imported.

3. **Alternative: Use feature flag:**
Add environment variable to switch between old and new parser:
```python
USE_NEW_PARSER = os.environ.get("USE_NEW_PARSER", "false").lower() == "true"

if USE_NEW_PARSER:
    parse_result = parse_tool_calls(...)
else:
    # Old parsing logic
    tool_calls = mcp_to_openai_tool_calls(...)
```

---

## Success Criteria Checklist

- [ ] All unit tests pass (>40 tests)
- [ ] Test coverage >90% for parser module
- [ ] Real MiroThinker integration works
- [ ] Error handling doesn't crash server
- [ ] Parsing failures logged appropriately
- [ ] Old code removed from mcp_converter.py
- [ ] Each parser file <200 lines
- [ ] No regression in existing functionality
- [ ] Documentation updated (README/CLAUDE.md)

---

**Plan Complete**

This plan creates a fully tested, modular parser system following TDD principles. Each task is bite-sized (2-5 minutes) and commits frequently. The old parsing code remains available for rollback until the new system is validated.
