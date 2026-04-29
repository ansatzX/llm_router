"""XML format parser for tool calls.

This module provides parsing for multiple XML-based tool call formats including
MCP XML, TOOL_CALL XML, and simple XML tags.
"""

import json
from typing import Any

import regex as re  # Use regex library for better Unicode support
from json_repair import repair_json

from llm_router.parser.base import (
    MAX_CONTENT_SIZE,
    ParseResult,
    ToolCall,
    ToolCallParser,
    ToolNameResolver,
    validate_tool_arguments,
)
from llm_router.parser.errors import ValidationError, XMLParseError
from llm_router.parser.validator import validate_tool_call


class XMLParser(ToolCallParser):
    """Parser for XML format tool calls.

    Supports multiple XML formats:
    - MCP: <use_mcp_tool>...</use_mcp_tool>
    - TOOL_CALL: [TOOL_CALL]...[/TOOL_CALL]
    - Simple: <tool>...</tool>
    """

    # XML patterns for different formats
    PATTERNS: dict[str, str] = {
        'mcp': r'<use_mcp_tool>(.*?)</use_mcp_tool>',
        'codex_tool_calls': r'<tool_calls>(.*?)</tool_calls>',
        'bare_codex_tool_call': r'(<tool_call\s+name="[^"]+"[^>]*>.*?</tool_call>)',
        'tool_call': r'\[TOOL_CALL\](.*?)\[/TOOL_CALL\]',
        'simple': r'<tool>(.*?)</tool>',
    }

    # Tag name fallbacks
    TOOL_NAME_TAGS: list[str] = ['tool_name', 'function_name', 'name', 'function', 'action']
    ARGUMENT_TAGS: list[str] = ['arguments', 'parameters', 'input', 'params', 'args']

    def __init__(self, resolver: ToolNameResolver | None = None) -> None:
        """Initialize parser with optional tool name resolver.

        Args:
            resolver: Optional ToolNameResolver instance for name resolution.
        """
        self.resolver = resolver or ToolNameResolver()

    @property
    def format_name(self) -> str:
        """Parser format name for debugging.

        Returns:
            String 'xml' identifying this parser.
        """
        return "xml"

    def parse(self, content: str, available_tools: list[dict[str, Any]] | None = None) -> ParseResult:
        """Parse XML format tool calls.

        Args:
            content: Content string to parse for XML tool calls.
            available_tools: Optional list of available tools for validation.

        Returns:
            ParseResult with success status, tool_calls, errors, and warnings.
        """
        # Size limit
        if len(content) > MAX_CONTENT_SIZE:
            content = content[:MAX_CONTENT_SIZE]

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
                    parsed_calls = (
                        self._parse_codex_tool_calls(match_content, available_tools)
                        if format_type in ('codex_tool_calls', 'bare_codex_tool_call')
                        else [
                            self._parse_single_xml(
                                match_content,
                                has_server_name=(format_type == 'mcp'),
                                format_type=format_type,
                                available_tools=available_tools
                            )
                        ]
                    )

                    for tool_call in parsed_calls:
                        if not tool_call:
                            continue
                        # Validate basic structure
                        is_valid, warning = validate_tool_call(tool_call)
                        if warning:
                            warnings.append(warning)

                        if is_valid:
                            # Validate against tool schema
                            tool_schema = None
                            if available_tools:
                                for tool in available_tools:
                                    if tool.get("type") == "function":
                                        func = tool.get("function", {})
                                        if func.get("name") == tool_call.tool_name:
                                            tool_schema = tool
                                            break

                            schema_errors = validate_tool_arguments(
                                tool_call.tool_name,
                                tool_call.arguments,
                                tool_schema
                            )

                            if schema_errors:
                                # Schema validation failed - treat as error
                                errors.extend(schema_errors)
                            else:
                                # All validation passed
                                tool_calls.append(tool_call)

                except (XMLParseError, ValidationError) as e:
                    errors.append(str(e))
                except Exception as e:
                    errors.append(f"Unexpected XML parse error: {e}")

            # If we have schema validation errors, return them immediately
            # This allows Rollback to trigger for malformed arguments
            if errors and not tool_calls:
                return ParseResult.error(errors, warnings)

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
        has_server_name: bool,
        format_type: str = 'mcp',
        available_tools: list[dict[str, Any]] | None = None
    ) -> ToolCall | None:
        """Parse single XML block.

        Args:
            content: XML content string to parse.
            has_server_name: True if format includes server_name tag.
            format_type: Type of XML format ('mcp', 'tool_call', 'simple').
            available_tools: Optional list of available tools for validation.

        Returns:
            ToolCall instance if parsing succeeds, None otherwise.

        Raises:
            ValidationError: If required fields are missing.
            XMLParseError: If XML parsing fails.
        """
        # TOOL_CALL format contains JSON, not XML
        if format_type == 'tool_call':
            return self._parse_tool_call_json(content, available_tools)
        if format_type == 'codex_tool_calls':
            raise ValidationError(
                "Codex tool_calls blocks must be parsed as a group",
                context=content[:100],
            )

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

        # Resolve tool name using server_name
        resolved_name = self.resolver.resolve(server_name, tool_name, available_tools)

        # Extract arguments
        args_text = self._extract_tag(content, self.ARGUMENT_TAGS)
        arguments = self._parse_arguments(args_text) if args_text else {}

        return ToolCall(
            tool_name=resolved_name,
            arguments=arguments,
            server_name=server_name
        )

    def _parse_codex_tool_calls(
        self,
        content: str,
        available_tools: list[dict[str, Any]] | None = None,
    ) -> list[ToolCall]:
        """Parse Codex-style <tool_calls> XML blocks."""
        tool_calls = []
        for tool_match in re.finditer(
            r'<tool_call\s+name="([^"]+)"[^>]*>(.*?)</tool_call>',
            content,
            re.DOTALL,
        ):
            tool_name = tool_match.group(1).strip()
            body = tool_match.group(2)
            arguments: dict[str, Any] = {}
            for param_match in re.finditer(
                r'<parameter\s+name="([^"]+)"([^>]*)>(.*?)</parameter>',
                body,
                re.DOTALL,
            ):
                arguments[param_match.group(1)] = self._parse_codex_parameter(
                    param_match.group(2),
                    param_match.group(3),
                )

            resolved_name = self.resolver.resolve("tools", tool_name, available_tools)
            if available_tools and not self._tool_exists(resolved_name, available_tools):
                available_names = ", ".join(self._available_tool_names(available_tools))
                raise ValidationError(
                    f"Unknown tool '{tool_name}'. Available tools: {available_names}",
                    context=content[:100],
                )
            tool_calls.append(
                ToolCall(
                    tool_name=resolved_name,
                    arguments=arguments,
                    server_name="tools",
                )
            )
        return tool_calls

    def _parse_codex_parameter(self, attributes: str, text: str) -> Any:
        """Parse a Codex XML parameter value."""
        value = text.strip()
        string_match = re.search(r'\bstring="([^"]+)"', attributes)
        is_string = not string_match or string_match.group(1).lower() != "false"
        if is_string:
            return value
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value

    def _available_tool_names(self, available_tools: list[dict[str, Any]]) -> list[str]:
        names = []
        for tool in available_tools:
            if tool.get("type") == "function":
                func = tool.get("function", {})
                if name := func.get("name"):
                    names.append(name)
            elif name := tool.get("name"):
                names.append(name)
        return names

    def _tool_exists(
        self,
        tool_name: str,
        available_tools: list[dict[str, Any]],
    ) -> bool:
        return tool_name in self._available_tool_names(available_tools)

    def _parse_tool_call_json(self, content: str, available_tools: list[dict[str, Any]] | None = None) -> ToolCall | None:
        """Parse TOOL_CALL format with JSON content.

        Args:
            content: JSON content string from TOOL_CALL tag.
            available_tools: Optional list of available tools for validation.

        Returns:
            ToolCall instance if parsing succeeds.

        Raises:
            ValidationError: If required fields are missing.
            XMLParseError: If JSON parsing fails.
        """
        try:
            data = json.loads(content.strip())
        except json.JSONDecodeError:
            try:
                repaired = repair_json(content.strip())
                data = json.loads(repaired)
            except Exception as e:
                raise XMLParseError(
                    f"Failed to parse TOOL_CALL JSON: {e}",
                    context=content[:100]
                ) from None

        # Extract name (required)
        name = data.get('name')
        if not name:
            raise ValidationError(
                "Missing 'name' field in TOOL_CALL JSON",
                context=content[:100]
            )

        # Extract arguments
        arguments = data.get('arguments', {})
        if not isinstance(arguments, dict):
            arguments = {}

        return ToolCall(
            tool_name=name,
            arguments=arguments,
            server_name="tools"
        )

    def _extract_tag(self, content: str, tag_names: list[str]) -> str | None:
        """Extract content from first matching tag.

        Args:
            content: XML content string to search.
            tag_names: List of tag names to try in order.

        Returns:
            Tag content if found, None otherwise.
        """
        for tag in tag_names:
            match = re.search(rf'<{tag}>(.*?)</{tag}>', content, re.DOTALL)
            if match:
                return match.group(1).strip()
        return None

    def _parse_arguments(self, args_text: str) -> dict[str, Any]:
        """Parse JSON arguments with repair fallback.

        Args:
            args_text: JSON string containing tool arguments.

        Returns:
            Parsed arguments dictionary.

        Raises:
            XMLParseError: If JSON parsing fails even after repair.
        """
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
                ) from None
