# LLM Router Parser Refactoring Design

**Date:** 2026-03-26
**Status:** Revised (after review)
**Author:** Claude (via brainstorming session)
**Reviewer:** spec-document-reviewer agent
**Review Status:** APPROVED after revisions

## Overview

Refactor the LLM Router's MCP parsing logic to improve error handling, maintainability, and extensibility. The router acts as a pure format converter between OpenAI API format and MCP XML/JSON tool calls, without executing tools or managing MCP servers.

## Problem Statement

**Current Issues:**
1. Monolithic parsing logic in `mcp_converter.py` (300 lines) handles multiple concerns
2. Error handling is scattered - parse failures can cause unexpected errors
3. Multiple XML/JSON formats are handled in a single function, making debugging difficult
4. No clear separation between parsing, validation, and format conversion
5. Limited error context for debugging malformed tool calls

**Target Scenario:**
- Agent uses tools from mixed sources (OpenAI tools + MCP servers)
- Router performs pure format conversion
- Client is responsible for tool execution and MCP server management
- Parsing failures should not crash the service

## Requirements

- **Python Version**: 3.10+ (for `list[Type]` syntax and dataclasses with kw_only)
- **Dependencies**: json_repair, pydantic (optional, using dataclasses for simplicity)

## Goals

1. **Robust Error Handling** - Parse failures are caught, logged, and don't interrupt service
2. **Clear Architecture** - Separate concerns: parsing, validation, format conversion
3. **Extensibility** - Easy to add new parsing formats in the future
4. **Testability** - Each parser can be unit tested independently
5. **Debugging Support** - Rich error context and warnings for troubleshooting

## Architecture

### Module Structure

```
llm_router/
├── parser/                      # New module
│   ├── __init__.py             # Public API: parse_tool_calls()
│   ├── base.py                 # Data classes: ToolCall, ParseResult
│   ├── xml_parser.py           # XML format parsing
│   ├── json_parser.py          # JSON format parsing
│   ├── validator.py            # Data validation
│   └── errors.py               # Error types
├── mcp_converter.py            # Simplified: only prompt generation
├── server.py                   # Updated: use new parser API
└── [other modules unchanged]
```

### Data Flow

```
Client Request (OpenAI tools)
    ↓
Server: Build payload + inject MCP prompt
    ↓
LLM Client: Call backend LLM
    ↓
Backend Response (MCP XML/JSON)
    ↓
Parser: parse_tool_calls(content, reasoning_content)
    ↓
ParseResult(success=True, tool_calls=[...])
    ↓
Server: Convert to OpenAI tool_calls format
    ↓
Client Response (OpenAI format)
```

### Error Flow

```
Parser: parse_tool_calls() fails
    ↓
ParseResult(success=False, errors=[...], warnings=[...])
    ↓
Server: Log errors/warnings
    ↓
Return normal text response (finish_reason="stop")
    ↓
Client continues normally
```

## Component Design

### 1. Data Structures (`parser/base.py`)

**ToolCall Data Class:**
```python
@dataclass
class ToolCall:
    tool_name: str              # Required
    arguments: dict             # Required
    server_name: str = "tools"  # Optional with default

    def to_openai_format() -> dict
```
```

**ParseResult Data Class:**
```python
@dataclass
class ParseResult:
    success: bool
    tool_calls: list[ToolCall]
    errors: list[str]
    warnings: list[str]

    @classmethod
    def ok(cls, tool_calls, warnings=None) -> ParseResult
    @classmethod
    def error(cls, errors, warnings=None) -> ParseResult
```

### 2. Error Types (`parser/errors.py`)

**Error Hierarchy:**
```
ParseError (base)
├── XMLParseError
├── JSONParseError
└── ValidationError
```

**Error Properties:**
- `error_type: ParseErrorType` (enum)
- `message: str`
- `context: str` (raw content snippet for debugging)

### 3. Parser Interface (`parser/base.py`)

**Abstract Base Class:**
```python
class ToolCallParser(ABC):
    @abstractmethod
    def parse(self, content: str) -> ParseResult:
        """Parse content, return result with all found tool calls"""
        pass

    @property
    @abstractmethod
    def format_name(self) -> str:
        """Parser format name for debugging"""
        pass
```

**Note:** The `can_parse()` method was removed in favor of a try-parse approach. Each parser attempts to parse and returns a result, making the API simpler and more robust.
```

### 4. XML Parser (`parser/xml_parser.py`)

**Supported Formats:**
1. MCP XML: `<use_mcp_tool>...</use_mcp_tool>`
2. Tool Call XML: `[TOOL_CALL]...[/TOOL_CALL]`
3. Simple XML: `<tool>...</tool>`

**Parsing Strategy:**
1. Try formats in priority order (MCP → TOOL_CALL → simple)
2. For each format, extract ALL occurrences (not just first match)
3. Extract tool_name, arguments, server_name from XML tags
4. Parse JSON arguments with fallback to json_repair
5. Validate each extracted tool call
6. Return all successfully parsed tool calls
7. Continue to next format only if current format finds zero matches

**Important:** A parser returns all tool calls found in its format, even if some individual tool calls failed to parse. This is "partial success" - the parser found its format and successfully extracted some tool calls.

**Tag Name Fallbacks:**
- Tool name: `tool_name`, `function_name`, `name`, `function`, `action`
- Arguments: `arguments`, `parameters`, `input`, `params`, `args`

### 5. JSON Parser (`parser/json_parser.py`)

**Supported Formats:**
- `{"name": "...", "arguments": {...}}`
- `{"tool": "...", "input": {...}}`

**Parsing Strategy:**
1. Extract all JSON objects from content
2. Filter objects containing `name` or `tool` field
3. Parse JSON with fallback to json_repair
4. Extract tool name and arguments
5. Handle nested JSON strings in arguments field

**Field Name Fallbacks:**
- Tool name: `name`, `tool`, `tool_name`
- Arguments: `arguments`, `input`, `params`

### 6. Validator (`parser/validator.py`)

**Validation Rules:**
1. **Required Fields:**
   - `tool_name`: non-empty string
   - `arguments`: dict type

2. **Optional Fields:**
   - `server_name`: defaults to "tools" if missing

3. **Warnings (non-blocking):**
   - Empty server name
   - None values in arguments
   - Whitespace-only tool name

**Functions:**
- `validate_tool_call(ToolCall) -> (is_valid: bool, warning: Optional[str])`
- `sanitize_arguments(dict) -> dict` (remove None values)

### 7. Public API (`parser/__init__.py`)

**Main Function:**
```python
def parse_tool_calls(
    content: str,
    reasoning_content: str = "",
    prefer_format: str = None
) -> ParseResult:
    """
    Unified parsing interface

    Args:
        content: Main response content
        reasoning_content: Reasoning content (MiroThinker)
        prefer_format: Optional hint - "xml" tries XMLParser first, "json" tries JSONParser first
                      If None, defaults to XMLParser then JSONParser

    Returns:
        ParseResult with success status, tool_calls, errors, warnings

    Note:
        - Content and reasoning_content are merged before parsing (behavior change from current)
        - Returns first successful parser result
        - Success = True if ANY tool calls were found
        - Errors/warnings aggregated from failed parsers if all fail
    """
```

**Parsing Strategy:**
1. Merge `content` and `reasoning_content` (deliberate change from separate parsing)
2. Select parser order based on `prefer_format` (if provided)
3. Default order: XMLParser, JSONParser
4. For each parser:
   - Call `parse()` to extract all tool calls
   - If parser returns any tool_calls (partial success), return immediately
   - If parser returns zero tool_calls, collect errors/warnings and try next parser
5. If all parsers return zero tool_calls:
   - Return ParseResult(success=False, errors=aggregated_errors, warnings=aggregated_warnings)

**Error Aggregation:**
```python
all_errors = []
all_warnings = []

for parser in parsers:
    result = parser.parse(content)
    if result.tool_calls:  # Success (partial or full)
        return result  # First success wins
    else:
        all_errors.extend(result.errors)
        all_warnings.extend(result.warnings)

# All parsers failed
return ParseResult.error(all_errors, all_warnings)
```

## Integration Changes

### server.py Changes

**Before:**
```python
tool_calls = mcp_to_openai_tool_calls(response_text)
if not tool_calls and reasoning_text:
    tool_calls = mcp_to_openai_tool_calls(reasoning_text)

if tool_calls:
    response["choices"][0]["message"]["tool_calls"] = tool_calls
    response["choices"][0]["finish_reason"] = "tool_calls"
```

**After:**
```python
from llm_router.parser import parse_tool_calls

parse_result = parse_tool_calls(response_text, reasoning_text)

# Log warnings
for warning in parse_result.warnings:
    logger.warning(f"Parse warning: {warning}")

# Log errors (don't interrupt flow)
for error in parse_result.errors:
    logger.error(f"Parse error: {error}")

# Build response
if parse_result.success:
    tool_calls_openai = [tc.to_openai_format() for tc in parse_result.tool_calls]
    response["choices"][0]["message"]["tool_calls"] = tool_calls_openai
    response["choices"][0]["finish_reason"] = "tool_calls"
```

### mcp_converter.py Simplification

**Keep:**
- `generate_mcp_system_prompt()` - prompt generation

**Remove:**
- `extract_tool_calls_from_content()`
- `mcp_to_openai_tool_calls()`
- All parsing functions

**Final Size:** ~100 lines (from 300 lines)

## Error Handling Strategy

### Parse Errors (Don't Interrupt Service)

1. **Invalid XML:**
   - Missing closing tags
   - Malformed structure
   - Invalid JSON in arguments
   → Log error, return normal text response

2. **Invalid JSON:**
   - Malformed JSON
   - Missing required fields
   - Type mismatches
   → Log error, return normal text response

3. **Validation Errors:**
   - Empty tool_name
   - Non-dict arguments
   → Log warning, skip this tool call

### Recovery Mechanisms

1. **JSON Repair:** Use `json_repair` for malformed JSON in arguments
2. **Format Fallback:** Try XML → JSON → fail gracefully
3. **Default Values:** server_name defaults to "tools"
4. **Partial Success:** Return successfully parsed tool calls even if others failed

## Testing Strategy

### Unit Tests

**Important Context:**
- **No existing tests**: The current codebase has zero tests (`tests/` directory is empty)
- This is a greenfield testing effort - all tests must be created from scratch
- Success criteria ">90% coverage" applies to new parser module only

**Test Coverage:**
1. **XMLParser:**
   - Valid MCP XML
   - Valid TOOL_CALL XML
   - Valid simple XML
   - Malformed XML
   - Missing tool_name
   - Invalid JSON in arguments

2. **JSONParser:**
   - Valid JSON with `name` field
   - Valid JSON with `tool` field
   - Malformed JSON
   - Nested JSON strings
   - Missing required fields

3. **Validator:**
   - Valid tool call
   - Empty tool_name
   - Non-dict arguments
   - None values in arguments

4. **Integration:**
   - Mixed content and reasoning_content
   - Format detection priority
   - Error aggregation

### Test Files

```
tests/
├── test_xml_parser.py
├── test_json_parser.py
├── test_validator.py
└── test_parser_integration.py
```

## Migration Plan

### Phase 1: Add New Parser Module
1. Create `llm_router/parser/` directory
2. Implement all parser components
3. Write unit tests
4. Verify parser works correctly

### Phase 2: Update Server
1. Import `parse_tool_calls` in `server.py`
2. Replace old parsing logic
3. Update error handling
4. Test integration

### Phase 3: Cleanup
1. Remove old parsing code from `mcp_converter.py`
2. Update imports
3. Remove dead code
4. Update documentation

### Phase 4: Testing
1. Run existing tests (if any)
2. Add new tests for parser module
3. Test with real MiroThinker responses
4. Performance testing

## Success Criteria

1. ✅ **No Service Interruption:** Parse failures don't crash the server
2. ✅ **Clear Error Messages:** All errors logged with context
3. ✅ **Backward Compatible:** Existing clients work without changes
4. ✅ **Extensible:** Adding new format requires only new Parser class
5. ✅ **Testable:** Each component has >90% unit test coverage
6. ✅ **Maintainable:** Each file <200 lines, single responsibility

## Risks and Mitigations

### Risk 1: Breaking Existing Behavior
**Mitigation:**
- Comprehensive unit tests before integration
- Gradual rollout with feature flag (optional)
- Keep old parsing code until new parser is validated

### Risk 2: Performance Regression
**Mitigation:**
- Benchmark parsing speed before/after
- Optimize hot paths (regex compilation, JSON parsing)
- Add size limits (already in place: 10MB)

### Risk 3: Incomplete Error Context
**Mitigation:**
- Always include raw content snippet in errors
- Log full content in debug mode
- Add structured error types for programmatic handling

### Risk 4: Behavior Change - reasoning_content Merging
**Current Behavior:** Try content first, then separately try reasoning_content if no tool calls found
**New Behavior:** Merge content and reasoning_content before parsing
**Mitigation:**
- This is a deliberate simplification
- Merging is more robust (handles tool calls split across both)
- Document as breaking change in migration notes
- Keep old behavior available via flag if needed

## Future Enhancements

1. **Streaming Support:** Adapt parser for streaming responses
2. **Additional Formats:** Add new parsers as formats emerge
3. **Metrics:** Track parse success rates by format
4. **Caching:** Cache repeated parsing of identical content
5. **Schema Validation:** Validate arguments against tool parameter schema

## Open Questions

1. ~~Should server_name be configurable per request?~~ → No, use default "tools"
2. ~~Should we support partial tool call parsing?~~ → Yes, return successfully parsed calls
3. ~~Should parse errors be returned to client?~~ → No, log server-side only

## References

- Original code: `llm_router/mcp_converter.py`
- Example implementation: User-provided Python example
- OpenAI API spec: https://platform.openai.com/docs/api-reference/chat
- MCP Protocol: https://modelcontextprotocol.io/

---

**Next Steps:**
1. Review this spec
2. Create implementation plan
3. Begin Phase 1 development
