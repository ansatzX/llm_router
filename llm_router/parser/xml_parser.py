"""XML format parser for tool calls."""
import json
import regex as re  # Use regex library for better Unicode support

from json_repair import repair_json

from .base import MAX_CONTENT_SIZE, ParseResult, ToolCall, ToolCallParser, ToolNameResolver, validate_tool_arguments
from .errors import ValidationError, XMLParseError
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

    def __init__(self, resolver: ToolNameResolver | None = None):
        """Initialize parser with optional tool name resolver."""
        self.resolver = resolver or ToolNameResolver()

    @property
    def format_name(self) -> str:
        return "xml"

    def parse(self, content: str, available_tools: list[dict] | None = None) -> ParseResult:
        """Parse XML format tool calls."""
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
                    tool_call = self._parse_single_xml(
                        match_content,
                        has_server_name=(format_type == 'mcp'),
                        format_type=format_type,
                        available_tools=available_tools
                    )

                    if tool_call:
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
        available_tools: list[dict] | None = None
    ) -> ToolCall | None:
        """Parse single XML block."""
        # TOOL_CALL format contains JSON, not XML
        if format_type == 'tool_call':
            return self._parse_tool_call_json(content, available_tools)

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

    def _parse_tool_call_json(self, content: str, available_tools: list[dict] | None = None) -> ToolCall | None:
        """Parse TOOL_CALL format with JSON content."""
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
        """Extract content from first matching tag."""
        for tag in tag_names:
            match = re.search(rf'<{tag}>(.*?)</{tag}>', content, re.DOTALL)
            if match:
                return match.group(1).strip()
        return None

    def _parse_arguments(self, args_text: str) -> dict:
        """Parse JSON arguments with repair fallback."""
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
