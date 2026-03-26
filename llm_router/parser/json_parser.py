"""JSON format parser for tool calls."""
import json

from .base import MAX_CONTENT_SIZE, ParseResult, ToolCall, ToolCallParser
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
        if len(content) > MAX_CONTENT_SIZE:
            content = content[:MAX_CONTENT_SIZE]

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
        """Extract all potential JSON objects from content.

        Handles braces inside string values correctly by tracking string
        boundaries and escape sequences.
        """
        objects = []
        i = 0

        while i < len(content):
            if content[i] == '{':
                # Find matching closing brace, accounting for strings
                depth = 0
                in_string = False
                escape_next = False
                start = i

                while i < len(content):
                    char = content[i]

                    if escape_next:
                        # Current character is escaped, skip it
                        escape_next = False
                    elif char == '\\':
                        # Next character will be escaped
                        escape_next = True
                    elif char == '"' and not escape_next:
                        # Toggle string state (unescaped quote)
                        in_string = not in_string
                    elif not in_string:
                        # Only count braces outside strings
                        if char == '{':
                            depth += 1
                        elif char == '}':
                            depth -= 1
                            if depth == 0:
                                # Found complete JSON object
                                json_candidate = content[start:i+1]
                                # Filter: must contain tool-related fields (with or without quotes)
                                if any(field in json_candidate for field in ['"name"', '"tool"', 'name:', 'tool:']):
                                    objects.append(json_candidate)
                                break
                    i += 1
            i += 1

        return objects

    def _parse_single_json(self, json_str: str) -> ToolCall | None:
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
                ) from e

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
            except (json.JSONDecodeError, ValueError):
                try:
                    from json_repair import repair_json
                    repaired = repair_json(arguments)
                    arguments = json.loads(repaired)
                except (json.JSONDecodeError, ValueError, Exception):
                    arguments = {}

        # Ensure arguments is a dict
        if not isinstance(arguments, dict):
            arguments = {}

        return ToolCall(
            tool_name=str(tool_name),
            arguments=arguments,
            server_name="tools"
        )
