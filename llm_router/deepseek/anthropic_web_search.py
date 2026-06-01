"""DeepSeek Anthropic web-search bridge for hosted Codex Responses turns."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse

import httpx

from llm_router.chat_adapter_base import ChatCompletionAdapterBase
from llm_router.debug_log import log_debug
from llm_router.llm_client import LLMRequestError
from llm_router.responses_state.usage import _responses_usage_from_provider

_ANTHROPIC_VERSION = "2023-06-01"
_WEB_SEARCH_TOOL_VERSION = "web_search_20250305"
_WEB_SEARCH_MAX_USES = 100
_WEB_SEARCH_DEFAULT_MAX_TOKENS = 524288
_MAX_PAUSE_TURN_RETRIES = 4


@dataclass
class DeepSeekAnthropicWebSearchResult:
    """Normalized Responses-shaped result from DeepSeek Anthropic web search."""

    output_items: list[dict[str, Any]]
    output_text: str | None
    usage: dict[str, Any]
    raw_response: dict[str, Any]
    tool_calls_list: list[dict[str, Any]] | None = None


@dataclass
class DeepSeekAnthropicSearchExecution:
    """Compact search-only execution result for the server loop."""

    queries: list[str]
    searches: list[dict[str, Any]]
    text: str | None
    usage: dict[str, Any]
    raw_response: dict[str, Any]
    raw_responses: list[dict[str, Any]]
    content_blocks: list[dict[str, Any]]


def _anthropic_messages_url(llm_base_url: str) -> str:
    parsed = urlparse(llm_base_url.rstrip("/"))
    path = parsed.path.rstrip("/")
    if path.endswith("/anthropic/v1/messages"):
        pass
    elif path.endswith("/anthropic/v1"):
        path = f"{path}/messages"
    elif path.endswith("/anthropic"):
        path = f"{path}/v1/messages"
    elif path.endswith("/v1"):
        path = path[:-3]
        path = f"{path}/anthropic/v1/messages"
    else:
        path = f"{path}/anthropic/v1/messages"
    return parsed._replace(path=path).geturl()


def _anthropic_tool_from_codex_web_search(tool: dict[str, Any]) -> dict[str, Any]:
    converted: dict[str, Any] = {
        "type": _WEB_SEARCH_TOOL_VERSION,
        "name": "web_search",
        "max_uses": _WEB_SEARCH_MAX_USES,
    }
    search_context_size = tool.get("search_context_size")
    if search_context_size in {"low", "medium", "high"}:
        converted["search_context_size"] = search_context_size
    filters = tool.get("filters")
    if isinstance(filters, dict):
        allowed_domains = filters.get("allowed_domains")
        if isinstance(allowed_domains, list) and allowed_domains:
            converted["allowed_domains"] = [
                domain for domain in allowed_domains
                if isinstance(domain, str) and domain
            ]
    location = tool.get("location") or tool.get("user_location")
    if isinstance(location, dict) and location:
        converted["user_location"] = {
            key: value
            for key, value in location.items()
            if key in {"type", "city", "region", "country", "timezone"}
            and isinstance(value, str)
            and value
        }
    return converted


def _anthropic_tool_choice_from_responses_tool_choice(
    tool_choice: Any,
) -> dict[str, Any] | None:
    if isinstance(tool_choice, str):
        if tool_choice in {"none", "auto", "any"}:
            return {"type": tool_choice}
        return None
    if not isinstance(tool_choice, dict):
        return None

    choice_type = tool_choice.get("type")
    if choice_type in {"none", "auto", "any"}:
        return {"type": choice_type}
    if choice_type == "tool":
        name = tool_choice.get("name")
        if isinstance(name, str) and name:
            return {"type": "tool", "name": name}
    if choice_type == "function":
        function = tool_choice.get("function")
        if isinstance(function, dict):
            name = function.get("name")
            if isinstance(name, str) and name:
                return {"type": "tool", "name": name}
        name = tool_choice.get("name")
        if isinstance(name, str) and name:
            return {"type": "tool", "name": name}
    return None


def _reasoning_effort_from_responses_payload(data: dict[str, Any]) -> str | None:
    reasoning_effort = data.get("reasoning_effort")
    if isinstance(reasoning_effort, str) and reasoning_effort:
        return reasoning_effort

    reasoning = data.get("reasoning")
    if isinstance(reasoning, dict):
        effort = reasoning.get("effort")
        if isinstance(effort, str) and effort:
            return effort
    return None


def _stop_sequences_from_responses_payload(data: dict[str, Any]) -> list[str] | None:
    stop_sequences = data.get("stop_sequences")
    if isinstance(stop_sequences, list):
        values = [item for item in stop_sequences if isinstance(item, str)]
        return values or None

    stop = data.get("stop")
    if isinstance(stop, str) and stop:
        return [stop]
    if isinstance(stop, list):
        values = [item for item in stop if isinstance(item, str)]
        return values or None
    return None


def _anthropic_request_options_from_responses_payload(
    data: dict[str, Any],
) -> dict[str, Any]:
    options: dict[str, Any] = {}

    tool_choice = _anthropic_tool_choice_from_responses_tool_choice(
        data.get("tool_choice"),
    )
    if tool_choice is not None:
        options["tool_choice"] = tool_choice

    thinking = data.get("thinking")
    if isinstance(thinking, dict):
        options["thinking"] = thinking

    output_config = data.get("output_config")
    if isinstance(output_config, dict):
        options["output_config"] = output_config
    else:
        effort = _reasoning_effort_from_responses_payload(data)
        if effort:
            options["output_config"] = {"effort": effort}

    stop_sequences = _stop_sequences_from_responses_payload(data)
    if stop_sequences is not None:
        options["stop_sequences"] = stop_sequences

    return options


def _dsml_follow_up_prompt(queries: list[str]) -> str:
    query_lines = "\n".join(f"- {query}" for query in queries)
    return (
        "Continue the same answer by using the hosted web_search tool for these "
        "extracted follow-up queries. Do not write DSML or tool-call markup as "
        f"assistant text.\n{query_lines}"
    )


def _tool_choice_disables_tools(request_options: dict[str, Any] | None) -> bool:
    if not request_options:
        return False
    tool_choice = request_options.get("tool_choice")
    return isinstance(tool_choice, dict) and tool_choice.get("type") == "none"


def _message_content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if not isinstance(item, dict):
                continue
            item_type = item.get("type")
            if item_type in {"input_text", "output_text", "text"}:
                text = item.get("text")
                if isinstance(text, str) and text:
                    parts.append(text)
            elif item_type == "tool_result":
                text = item.get("content")
                if isinstance(text, str) and text:
                    parts.append(text)
        return "\n".join(parts)
    return ""


def _tool_use_input(arguments: Any) -> Any:
    if isinstance(arguments, dict):
        return arguments
    if isinstance(arguments, str):
        try:
            parsed = json.loads(arguments)
        except json.JSONDecodeError:
            return {"input": arguments}
        if isinstance(parsed, dict):
            return parsed
        return {"input": parsed}
    return {}


def _queries_from_internal_tool_arguments(arguments: Any) -> list[str]:
    if isinstance(arguments, str):
        try:
            parsed: Any = json.loads(arguments)
        except json.JSONDecodeError:
            return []
    elif isinstance(arguments, dict):
        parsed = arguments
    else:
        return []

    if not isinstance(parsed, dict):
        return []

    queries: list[str] = []
    seen: set[str] = set()

    def add_query(value: Any) -> None:
        if isinstance(value, str):
            query = value.strip()
            if query and query not in seen:
                seen.add(query)
                queries.append(query)
        elif isinstance(value, list):
            for item in value:
                add_query(item)

    add_query(parsed.get("query"))
    add_query(parsed.get("queries"))
    return queries


_DSML_WEB_SEARCH_INVOKE_RE = re.compile(
    r'<｜｜DSML｜｜invoke name="web_search">(?P<body>.*?)(?=<｜｜DSML｜｜invoke name=|<｜｜DSML｜｜tool_calls>|$)',
    re.DOTALL,
)
_DSML_WEB_SEARCH_PARAMETER_RE = re.compile(
    r'<｜｜DSML｜｜parameter name="query" string="true">(?P<query>.*?)</｜｜DSML｜｜parameter>',
    re.DOTALL,
)


def _queries_from_dsml_text(text: str) -> list[str]:
    if not isinstance(text, str) or not text:
        return []

    queries: list[str] = []
    seen: set[str] = set()

    for match in _DSML_WEB_SEARCH_INVOKE_RE.finditer(text):
        body = match.group("body").strip()
        if not body:
            continue
        try:
            parsed = json.loads(body)
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, dict):
            extracted = _queries_from_internal_tool_arguments(parsed)
        else:
            extracted = [
                query.strip()
                for query in _DSML_WEB_SEARCH_PARAMETER_RE.findall(body)
                if isinstance(query, str) and query.strip()
            ]
        for query in extracted:
            if query not in seen:
                seen.add(query)
                queries.append(query)
    return queries


def _tool_type_map_from_responses_tools(tools: list[dict[str, Any]]) -> dict[str, Any]:
    return ChatCompletionAdapterBase().tool_type_map(tools)


def _anthropic_input_schema_from_tool(tool: dict[str, Any]) -> dict[str, Any]:
    if isinstance(tool.get("parameters"), dict):
        return tool["parameters"]
    if isinstance(tool.get("input_schema"), dict):
        return tool["input_schema"]
    return {
        "type": "object",
        "properties": {
            "input": {
                "type": "string",
                "description": "Freeform input for the tool.",
            },
        },
        "required": ["input"],
        "additionalProperties": False,
    }


def _anthropic_tool_from_responses_tool(tool: dict[str, Any]) -> dict[str, Any] | None:
    tool_type = tool.get("type")
    if tool_type == "web_search":
        return _anthropic_tool_from_codex_web_search(tool)
    if tool_type == "function":
        if isinstance(tool.get("function"), dict):
            func = tool["function"]
            name = func.get("name")
            if not isinstance(name, str) or not name:
                return None
            return {
                "name": name,
                "description": func.get("description", ""),
                "input_schema": _anthropic_input_schema_from_tool(func),
            }
        name = tool.get("name")
        if not isinstance(name, str) or not name:
            return None
        return {
            "name": name,
            "description": tool.get("description", ""),
            "input_schema": _anthropic_input_schema_from_tool(tool),
        }
    if tool_type == "namespace":
        namespace = tool.get("name")
        if not isinstance(namespace, str) or not namespace:
            return None
        # Anthropic messages only accept flat tools, so namespace children are
        # expanded into namespaced function tools.
        return None
    if tool_type == "custom":
        name = tool.get("name")
        if not isinstance(name, str) or not name:
            return None
        return {
            "name": name,
            "description": tool.get("description", ""),
            "input_schema": {
                "type": "object",
                "properties": {
                    "input": {
                        "type": "string",
                        "description": "Freeform input for the custom tool.",
                    }
                },
                "required": ["input"],
                "additionalProperties": False,
            },
        }
    return None


def _anthropic_tools_from_responses_tools(
    tools: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    converted: list[dict[str, Any]] = []
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        if tool.get("type") == "namespace":
            namespace = tool.get("name")
            if not isinstance(namespace, str) or not namespace:
                continue
            for child in tool.get("tools", []):
                if not isinstance(child, dict) or child.get("type") != "function":
                    continue
                child_name = child.get("name")
                if not isinstance(child_name, str) or not child_name:
                    continue
                converted.append({
                    "name": ChatCompletionAdapterBase().namespaced_chat_tool_name(
                        namespace,
                        child_name,
                    ),
                    "description": child.get("description", ""),
                    "input_schema": _anthropic_input_schema_from_tool(child),
                })
            continue
        tool_def = _anthropic_tool_from_responses_tool(tool)
        if tool_def is not None:
            converted.append(tool_def)
    return converted


def _system_from_chat_messages(messages: list[dict[str, Any]]) -> str | None:
    parts: list[str] = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        if message.get("role") not in {"system", "developer"}:
            continue
        text = _message_content_to_text(message.get("content"))
        if text:
            parts.append(text)
    if not parts:
        return None
    return "\n\n".join(parts)


def _messages_from_chat_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    anthropic_messages: list[dict[str, Any]] = []
    index = 0
    while index < len(messages):
        message = messages[index]
        if not isinstance(message, dict):
            index += 1
            continue
        role = message.get("role")
        if role in {"system", "developer"}:
            index += 1
            continue
        if role == "tool":
            tool_results, index = _collect_tool_result_blocks(messages, index)
            if tool_results:
                anthropic_messages.append({
                    "role": "user",
                    "content": tool_results,
                })
            continue
        if role == "assistant" and isinstance(message.get("tool_calls"), list):
            blocks: list[dict[str, Any]] = _thinking_blocks_from_message(message)
            text = _message_content_to_text(message.get("content"))
            if text:
                blocks.append({"type": "text", "text": text})
            for tool_call in message.get("tool_calls") or []:
                if not isinstance(tool_call, dict):
                    continue
                function = tool_call.get("function") or {}
                tool_name = function.get("name")
                if not isinstance(tool_name, str) or not tool_name:
                    continue
                blocks.append({
                    "type": "tool_use",
                    "id": str(tool_call.get("id") or tool_call.get("call_id") or ""),
                    "name": tool_name,
                    "input": _tool_use_input(function.get("arguments", "")),
                })
            if blocks:
                anthropic_messages.append({
                    "role": "assistant",
                    "content": blocks,
                })
                tool_results, index = _collect_tool_result_blocks(messages, index + 1)
                if tool_results:
                    anthropic_messages.append({
                        "role": "user",
                        "content": tool_results,
                    })
                continue
            index += 1
            continue
        if role == "assistant":
            text = _message_content_to_text(message.get("content"))
            thinking_blocks = _thinking_blocks_from_message(message)
            if thinking_blocks:
                content: list[dict[str, Any]] = [*thinking_blocks]
                if text:
                    content.append({"type": "text", "text": text})
                anthropic_messages.append({
                    "role": "assistant",
                    "content": content,
                })
            elif text:
                anthropic_messages.append({
                    "role": "assistant",
                    "content": text,
                })
            index += 1
            continue
        text = _message_content_to_text(message.get("content"))
        if text:
            anthropic_messages.append({
                "role": "user" if role == "user" else str(role or "user"),
                "content": text,
            })
        index += 1
    return anthropic_messages


def _thinking_blocks_from_message(message: dict[str, Any]) -> list[dict[str, Any]]:
    reasoning_content = message.get("reasoning_content")
    if isinstance(reasoning_content, str) and reasoning_content:
        return [{"type": "thinking", "thinking": reasoning_content}]
    return []


def _collect_tool_result_blocks(
    messages: list[dict[str, Any]],
    start_index: int,
) -> tuple[list[dict[str, Any]], int]:
    tool_results: list[dict[str, Any]] = []
    index = start_index
    while index < len(messages):
        message = messages[index]
        if not isinstance(message, dict):
            index += 1
            continue
        role = message.get("role")
        if role in {"system", "developer"}:
            index += 1
            continue
        if role != "tool":
            break
        tool_call_id = message.get("tool_call_id") or message.get("call_id")
        if isinstance(tool_call_id, str) and tool_call_id:
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": tool_call_id,
                "content": _message_content_to_text(message.get("content")),
            })
        index += 1
    return tool_results, index


def _combine_usage(first: dict[str, Any], second: dict[str, Any]) -> dict[str, Any]:
    server_first = first.get("server_tool_use") or {}
    server_second = second.get("server_tool_use") or {}
    return {
        "input_tokens": int(first.get("input_tokens", 0)) + int(second.get("input_tokens", 0)),
        "input_tokens_details": {
            "cached_tokens": int(
                (first.get("input_tokens_details") or {}).get("cached_tokens", 0),
            ) + int((second.get("input_tokens_details") or {}).get("cached_tokens", 0)),
        },
        "output_tokens": int(first.get("output_tokens", 0)) + int(second.get("output_tokens", 0)),
        "output_tokens_details": {
            "reasoning_tokens": int(
                (first.get("output_tokens_details") or {}).get("reasoning_tokens", 0),
            ) + int((second.get("output_tokens_details") or {}).get("reasoning_tokens", 0)),
        },
        "server_tool_use": {
            "web_search_requests": int(server_first.get("web_search_requests", 0))
            + int(server_second.get("web_search_requests", 0)),
        },
        "total_tokens": int(first.get("total_tokens", 0)) + int(second.get("total_tokens", 0)),
    }


def _usage_from_anthropic_response(response: dict[str, Any]) -> dict[str, Any]:
    raw_usage = response.get("usage") or {}
    usage = _responses_usage_from_provider(raw_usage if isinstance(raw_usage, dict) else {})
    server_tool_use = raw_usage.get("server_tool_use") if isinstance(raw_usage, dict) else None
    if isinstance(server_tool_use, dict) and server_tool_use:
        usage["server_tool_use"] = {
            "web_search_requests": int(server_tool_use.get("web_search_requests", 0)),
        }
    elif response.get("content"):
        usage["server_tool_use"] = {"web_search_requests": 0}
    return usage


def _looks_like_dsml_tool_call(text: str) -> bool:
    return "<｜｜DSML｜｜tool_calls>" in text or "<｜｜DSML｜｜invoke" in text


def _response_blocks_without_dsml_residual_text(
    blocks: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[str]]:
    filtered_blocks: list[dict[str, Any]] = []
    follow_up_queries: list[str] = []
    seen_queries: set[str] = set()

    for block in blocks:
        if not isinstance(block, dict):
            continue
        if block.get("type") == "text":
            text = block.get("text")
            if isinstance(text, str) and _looks_like_dsml_tool_call(text):
                for query in _queries_from_dsml_text(text):
                    if query not in seen_queries:
                        seen_queries.add(query)
                        follow_up_queries.append(query)
                continue
        filtered_blocks.append(block)

    return filtered_blocks, follow_up_queries


def _tool_call_item(
    block: dict[str, Any],
    tool_type_map: dict[str, Any],
) -> dict[str, Any] | None:
    tool_name = block.get("name")
    call_id = str(block.get("id") or block.get("tool_use_id") or "")
    if not isinstance(tool_name, str) or not tool_name or not call_id:
        return None

    mapping = tool_type_map.get(tool_name)
    arguments = json.dumps(block.get("input") or {}, ensure_ascii=False)
    if mapping == "custom":
        return {
            "type": "custom_tool_call",
            "id": call_id,
            "call_id": call_id,
            "name": tool_name,
            "input": arguments,
        }
    if isinstance(mapping, dict) and mapping.get("type") == "namespace_function":
        return {
            "type": "function_call",
            "id": call_id,
            "call_id": call_id,
            "namespace": mapping.get("namespace", ""),
            "name": mapping.get("name", tool_name),
            "arguments": arguments,
        }
    return {
        "type": "function_call",
        "id": call_id,
        "call_id": call_id,
        "name": tool_name,
        "arguments": arguments,
    }


def _normalize_web_search_tool_result_content(
    content: Any,
) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    results: list[dict[str, Any]] = []
    error: dict[str, Any] | None = None
    if isinstance(content, list):
        for item in content:
            if not isinstance(item, dict):
                continue
            item_type = item.get("type")
            if item_type == "web_search_result":
                results.append(item)
            elif item_type == "web_search_tool_result_error" and error is None:
                error = item
    elif isinstance(content, dict):
        item_type = content.get("type")
        if item_type == "web_search_result":
            results.append(content)
        elif item_type == "web_search_tool_result_error":
            error = content
    return results, error


def _search_summary_from_response_blocks(
    query: str,
    blocks: list[dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]], str | None]:
    search_id = ""
    search_results: list[dict[str, Any]] = []
    text_parts: list[str] = []
    normalized_blocks: list[dict[str, Any]] = []
    for block in blocks:
        if not isinstance(block, dict):
            continue
        block_type = block.get("type")
        if block_type == "server_tool_use" and block.get("name") == "web_search":
            call_id = block.get("id")
            if not isinstance(call_id, str) or not call_id:
                raise LLMRequestError(
                    "DeepSeek Anthropic web search returned a server_tool_use without an id.",
                    status_code=502,
                )
            if not search_id:
                search_id = call_id
            input_block = block.get("input")
            normalized_block = {
                "type": "server_tool_use",
                "id": call_id,
                "name": "web_search",
                "input": input_block if isinstance(input_block, dict) else {},
            }
            normalized_blocks.append(normalized_block)
            continue
        if block_type == "web_search_tool_result":
            tool_use_id = block.get("tool_use_id")
            if not isinstance(tool_use_id, str) or not tool_use_id:
                raise LLMRequestError(
                    "DeepSeek Anthropic web search returned a web_search_tool_result without a tool_use_id.",
                    status_code=502,
                )
            result_content, error_content = _normalize_web_search_tool_result_content(
                block.get("content"),
            )
            search_results.extend(result_content)
            normalized_block: dict[str, Any] = {
                "type": "web_search_tool_result",
                "tool_use_id": tool_use_id,
                "content": result_content,
            }
            if error_content is not None:
                normalized_block["error"] = error_content
            normalized_blocks.append(normalized_block)
            continue
        if block_type == "text":
            text = block.get("text")
            if not isinstance(text, str) or not text:
                continue
            if _looks_like_dsml_tool_call(text):
                raise LLMRequestError(
                    "DeepSeek Anthropic web search returned DSML residual tool call text.",
                    status_code=502,
                )
            text_parts.append(text)
            normalized_block = {
                "type": "text",
                "text": text,
            }
            citations = block.get("citations")
            if isinstance(citations, list) and citations:
                normalized_block["citations"] = citations
            normalized_blocks.append(normalized_block)
            continue
        if block_type == "thinking":
            continue
        if block_type == "tool_use" and block.get("name") == "web_search":
            raise LLMRequestError(
                "DeepSeek Anthropic web search returned web_search as tool_use; expected server_tool_use.",
                status_code=502,
            )
        normalized_blocks.append(block)

    text = "\n".join(text_parts) or None
    summary = {
        "id": search_id,
        "query": query,
        "results": search_results,
        "text": text,
    }
    return summary, normalized_blocks, text


class DeepSeekAnthropicWebSearchExecutor:
    """Execute small Anthropic web-search-only requests for the router loop."""

    def __init__(self, base_url: str, api_key: str | None, model: str) -> None:
        self.base_url = base_url
        self.api_key = api_key
        self.model = model

    def execute(
        self,
        queries: list[str],
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
    ) -> DeepSeekAnthropicSearchExecution:
        cleaned_queries = [
            query.strip()
            for query in queries
            if isinstance(query, str) and query.strip()
        ]
        combined_usage: dict[str, Any] = _responses_usage_from_provider({})
        search_runs: list[dict[str, Any]] = []
        raw_responses: list[dict[str, Any]] = []
        content_blocks: list[dict[str, Any]] = []
        aggregated_text_parts: list[str] = []
        saw_any_results = False
        saw_any_visible_text = False

        for seed_query in cleaned_queries:
            pending_queries = [seed_query]
            seen_queries = {seed_query}
            chain_saw_results = False
            chain_saw_visible_text = False
            chain_attempts = 0

            while pending_queries and chain_attempts < _MAX_PAUSE_TURN_RETRIES:
                query = pending_queries.pop(0)
                response = self._execute_query(
                    query,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                )
                raw_responses.append(response)
                combined_usage = _combine_usage(
                    combined_usage,
                    _usage_from_anthropic_response(response),
                )
                response_blocks = [
                    block for block in (response.get("content") or []) if isinstance(block, dict)
                ]
                response_blocks, follow_up_queries = _response_blocks_without_dsml_residual_text(
                    response_blocks,
                )
                for follow_up_query in follow_up_queries:
                    if follow_up_query not in seen_queries:
                        seen_queries.add(follow_up_query)
                        pending_queries.append(follow_up_query)
                summary, normalized_blocks, text = _search_summary_from_response_blocks(
                    query,
                    response_blocks,
                )
                search_runs.append(summary)
                content_blocks.extend(normalized_blocks)
                if summary["results"]:
                    chain_saw_results = True
                    saw_any_results = True
                if text:
                    aggregated_text_parts.append(text)
                    chain_saw_visible_text = True
                    saw_any_visible_text = True
                chain_attempts += 1

            if pending_queries:
                if chain_saw_results or chain_saw_visible_text:
                    log_debug("DEEPSEEK_ANTHROPIC_WEB_SEARCH_BUDGET_EXHAUSTED", {
                        "seed_query": seed_query,
                        "completed_queries": [search["query"] for search in search_runs],
                        "pending_queries": pending_queries,
                    })
                    break
                raise LLMRequestError(
                    "DeepSeek Anthropic web search produced only empty search results.",
                    status_code=502,
                )

        if cleaned_queries and not saw_any_results and not saw_any_visible_text:
            raise LLMRequestError(
                "DeepSeek Anthropic web search produced only empty search results.",
                status_code=502,
            )

        raw_response = raw_responses[-1] if raw_responses else {}
        return DeepSeekAnthropicSearchExecution(
            queries=[search["query"] for search in search_runs],
            searches=search_runs,
            text="\n".join(aggregated_text_parts) or None,
            usage=combined_usage,
            raw_response=raw_response,
            raw_responses=raw_responses,
            content_blocks=content_blocks,
        )

    def run(
        self,
        queries: list[str],
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
    ) -> DeepSeekAnthropicSearchExecution:
        return self.execute(
            queries,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
        )

    def _execute_query(
        self,
        query: str,
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
    ) -> dict[str, Any]:
        messages: list[dict[str, Any]] = [{"role": "user", "content": query}]
        accumulated_content: list[dict[str, Any]] = []
        last_response: dict[str, Any] = {}

        for _attempt in range(_MAX_PAUSE_TURN_RETRIES):
            request_payload: dict[str, Any] = {
                "model": self.model,
                "messages": list(messages),
                "tools": [
                    {
                        "type": _WEB_SEARCH_TOOL_VERSION,
                        "name": "web_search",
                        "max_uses": _WEB_SEARCH_MAX_USES,
                    }
                ],
                "max_tokens": max_tokens
                if isinstance(max_tokens, int) and max_tokens > 0
                else _WEB_SEARCH_DEFAULT_MAX_TOKENS,
            }
            if isinstance(temperature, (int, float)):
                request_payload["temperature"] = temperature
            if isinstance(top_p, (int, float)):
                request_payload["top_p"] = top_p
            response = make_deepseek_anthropic_messages_request(
                request_payload,
                self.base_url,
                self.api_key,
            )
            last_response = response
            response_content = response.get("content") or []
            if isinstance(response_content, list):
                accumulated_content.extend(
                    block for block in response_content if isinstance(block, dict)
                )
            if response.get("stop_reason") != "pause_turn":
                break
            if not response_content:
                break
            messages.append({
                "role": "assistant",
                "content": response_content,
            })
        if accumulated_content and last_response.get("content") != accumulated_content:
            last_response = dict(last_response)
            last_response["content"] = accumulated_content
        return last_response


class DeepSeekAnthropicWebSearchBridge:
    """Sync bridge for router-owned hosted web_search turns."""

    def __init__(self, base_url: str, api_key: str | None, model: str) -> None:
        self.base_url = base_url
        self.api_key = api_key
        self.model = model

    def run(
        self,
        *,
        messages: list[dict[str, Any]],
        tools_raw: list[dict[str, Any]],
        max_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        request_options: dict[str, Any] | None = None,
    ) -> DeepSeekAnthropicWebSearchResult:
        anthropic_messages = _messages_from_chat_messages(messages)
        system = _system_from_chat_messages(messages)
        tool_type_map = _tool_type_map_from_responses_tools(tools_raw)
        tools = _anthropic_tools_from_responses_tools(tools_raw)
        accumulated_content: list[dict[str, Any]] = []
        combined_usage: dict[str, Any] = _responses_usage_from_provider({})
        last_response: dict[str, Any] = {}
        seen_dsml_queries: set[str] = set()

        for _attempt in range(_MAX_PAUSE_TURN_RETRIES):
            request_payload: dict[str, Any] = {
                "model": self.model,
                "messages": list(anthropic_messages),
                "tools": tools,
                "max_tokens": max_tokens
                if isinstance(max_tokens, int) and max_tokens > 0
                else _WEB_SEARCH_DEFAULT_MAX_TOKENS,
            }
            if system:
                request_payload["system"] = system
            if request_options:
                request_payload.update(request_options)
            if isinstance(temperature, (int, float)):
                request_payload["temperature"] = temperature
            if isinstance(top_p, (int, float)):
                request_payload["top_p"] = top_p
            response = make_deepseek_anthropic_messages_request(
                request_payload,
                self.base_url,
                self.api_key,
            )
            last_response = response
            response_content = response.get("content") or []
            response_blocks = [
                block for block in response_content if isinstance(block, dict)
            ] if isinstance(response_content, list) else []
            had_dsml_residual = any(
                block.get("type") == "text"
                and isinstance(block.get("text"), str)
                and _looks_like_dsml_tool_call(block["text"])
                for block in response_blocks
            )
            filtered_blocks, follow_up_queries = (
                _response_blocks_without_dsml_residual_text(response_blocks)
            )
            if isinstance(response_content, list):
                accumulated_content.extend(filtered_blocks)
            combined_usage = _combine_usage(combined_usage, _usage_from_anthropic_response(response))
            if follow_up_queries:
                if _tool_choice_disables_tools(request_options):
                    raise LLMRequestError(
                        "DeepSeek Anthropic web search returned DSML residual tool call text.",
                        status_code=502,
                    )
                new_queries = [
                    query for query in follow_up_queries
                    if query not in seen_dsml_queries
                ]
                if not new_queries:
                    raise LLMRequestError(
                        "DeepSeek Anthropic web search repeated DSML residual tool calls.",
                        status_code=502,
                    )
                seen_dsml_queries.update(new_queries)
                if filtered_blocks:
                    anthropic_messages.append({
                        "role": "assistant",
                        "content": filtered_blocks,
                    })
                anthropic_messages.append({
                    "role": "user",
                    "content": _dsml_follow_up_prompt(new_queries),
                })
                continue
            if had_dsml_residual:
                raise LLMRequestError(
                    "DeepSeek Anthropic web search returned DSML residual tool call text.",
                    status_code=502,
                )
            if response.get("stop_reason") != "pause_turn":
                break
            if not filtered_blocks:
                break
            anthropic_messages.append({
                "role": "assistant",
                "content": filtered_blocks,
            })
        output_items, output_text, tool_calls_list = _parse_response_blocks(
            accumulated_content,
            tool_type_map,
        )
        result = DeepSeekAnthropicWebSearchResult(
            output_items=output_items,
            output_text=output_text,
            usage=combined_usage,
            raw_response=last_response,
            tool_calls_list=tool_calls_list,
        )
        return result


def _parse_response_blocks(
    content: list[dict[str, Any]],
    tool_type_map: dict[str, Any],
) -> tuple[list[dict[str, Any]], str | None, list[dict[str, Any]]]:
    output_items: list[dict[str, Any]] = []
    tool_calls_list: list[dict[str, Any]] = []
    current_text_items: list[dict[str, Any]] = []
    current_text_parts: list[str] = []
    all_text_parts: list[str] = []
    thinking_parts: list[str] = []
    has_tool_calls = False
    emitted_reasoning = False

    def reasoning_text() -> str:
        return "\n".join(thinking_parts)

    def ensure_reasoning_item() -> None:
        nonlocal emitted_reasoning
        text = reasoning_text()
        if emitted_reasoning or not text:
            return
        output_items.append({
            "type": "reasoning",
            "summary": [{"type": "summary_text", "text": ""}],
            "content": [{"type": "reasoning_text", "text": text}],
        })
        emitted_reasoning = True

    def attach_reasoning(item: dict[str, Any]) -> None:
        text = reasoning_text()
        if text:
            item["reasoning_content"] = text

    def flush_text_items() -> None:
        nonlocal current_text_items, current_text_parts
        if not current_text_items:
            return
        ensure_reasoning_item()
        item = {
            "type": "message",
            "role": "assistant",
            "content": current_text_items,
        }
        attach_reasoning(item)
        output_items.append(item)
        current_text_items = []
        current_text_parts = []

    for block in content:
        block_type = block.get("type")
        if block_type == "server_tool_use" and block.get("name") == "web_search":
            flush_text_items()
            call_id = block.get("id")
            if not isinstance(call_id, str) or not call_id:
                raise LLMRequestError(
                    "DeepSeek Anthropic web search returned a server_tool_use without an id.",
                    status_code=502,
                )
            query = ""
            block_input = block.get("input")
            if isinstance(block_input, dict):
                query_value = block_input.get("query")
                if isinstance(query_value, str):
                    query = query_value
                elif isinstance(query_value, list):
                    query = "\n".join(
                        item for item in query_value if isinstance(item, str) and item
                    )
            action: dict[str, Any] = (
                {"type": "search", "query": query}
                if query
                else {"type": "other"}
            )
            ensure_reasoning_item()
            output_items.append({
                "type": "web_search_call",
                "id": call_id,
                "status": "completed",
                "action": action,
            })
            continue
        if block_type == "tool_use":
            if block.get("name") == "web_search":
                raise LLMRequestError(
                    "DeepSeek Anthropic web search returned web_search as tool_use; "
                    "expected server_tool_use.",
                    status_code=502,
                )
            item = _tool_call_item(block, tool_type_map)
            if item is None:
                continue
            flush_text_items()
            ensure_reasoning_item()
            attach_reasoning(item)
            output_items.append(item)
            tool_calls_list.append(item)
            has_tool_calls = True
            continue
        if block_type == "text":
            text = block.get("text")
            if not isinstance(text, str) or not text:
                continue
            if _looks_like_dsml_tool_call(text):
                raise LLMRequestError(
                    "DeepSeek Anthropic web search returned DSML residual tool call text.",
                    status_code=502,
                )
            content_item: dict[str, Any] = {
                "type": "output_text",
                "text": text,
            }
            citations = block.get("citations")
            if isinstance(citations, list) and citations:
                content_item["annotations"] = citations
            current_text_items.append(content_item)
            current_text_parts.append(text)
            all_text_parts.append(text)
            continue
        if block_type == "thinking":
            thinking = block.get("thinking")
            if isinstance(thinking, str) and thinking:
                thinking_parts.append(thinking)
            continue

    flush_text_items()
    output_text = None if has_tool_calls else "\n".join(all_text_parts) or None
    if not output_text and not has_tool_calls:
        if any(
            block.get("type") == "server_tool_use"
            and block.get("name") == "web_search"
            for block in content
        ):
            raise LLMRequestError(
                "DeepSeek Anthropic web search completed without final text.",
                status_code=502,
            )
        raise LLMRequestError(
            "DeepSeek Anthropic web search returned no usable output.",
            status_code=502,
        )
    return output_items, output_text, tool_calls_list


def make_deepseek_anthropic_messages_request(
    payload: dict[str, Any],
    llm_base_url: str,
    api_key: str | None = None,
) -> dict[str, Any]:
    """Call DeepSeek's Anthropic-compatible Messages endpoint."""
    url = _anthropic_messages_url(llm_base_url)
    headers = {
        "Content-Type": "application/json",
        "anthropic-version": _ANTHROPIC_VERSION,
    }
    if api_key:
        headers["x-api-key"] = api_key

    log_debug("DEEPSEEK_ANTHROPIC_WEB_SEARCH_REQUEST", {
        "url": url,
        "model": payload.get("model"),
        "tool_count": len(payload.get("tools", []) or []),
        "message_count": len(payload.get("messages", []) or []),
        "has_system": bool(payload.get("system")),
        "max_tokens": payload.get("max_tokens"),
    })

    try:
        response = httpx.post(
            url,
            headers=headers,
            json=payload,
            timeout=httpx.Timeout(connect=5.0, read=120.0, write=30.0, pool=10.0),
        )
        response.raise_for_status()
        try:
            result = response.json()
        except ValueError as exc:
            raise LLMRequestError(
                "DeepSeek Anthropic web search returned a non-JSON response.",
                status_code=response.status_code,
                body=getattr(response, "text", None),
            ) from exc
        log_debug("DEEPSEEK_ANTHROPIC_WEB_SEARCH_RESPONSE", result)
        if not isinstance(result, dict):
            raise LLMRequestError(
                "DeepSeek Anthropic web search returned a non-object response.",
                status_code=response.status_code,
                body=result,
            )
        return result
    except httpx.HTTPStatusError as exc:
        body: Any | None = None
        try:
            body = exc.response.json()
        except Exception:
            body = exc.response.text
        message = "DeepSeek Anthropic web search request failed."
        if isinstance(body, dict):
            error = body.get("error")
            if isinstance(error, dict) and isinstance(error.get("message"), str):
                message = error["message"]
            elif isinstance(body.get("message"), str):
                message = body["message"]
        raise LLMRequestError(
            message,
            status_code=exc.response.status_code,
            body=body,
        ) from exc
    except httpx.HTTPError as exc:
        raise LLMRequestError(
            f"DeepSeek Anthropic web search request failed: {exc}",
            status_code=502,
        ) from exc
