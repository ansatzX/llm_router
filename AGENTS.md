# Repository Guidelines

## Project Structure & Module Organization

Core application code lives in `llm_router/`. The main HTTP entrypoint is `llm_router/server.py`; CLI startup is `llm_router/cli.py`. Provider-specific behavior is split by adapter: `llm_router/deepseek/`, `llm_router/mirothinker/`, and `llm_router/openai_chat.py`. Responses state management is isolated in `llm_router/responses_state/`. Tests live in `tests/`, with focused files such as `test_server_responses.py` and `test_deepseek_adapter.py`. Configuration examples are in `router.toml`, `codex.config.example.toml`, and `llm_router.json`.

## Build, Test, and Development Commands

- `uv run llm-router serve` — start the local router on the configured host/port.
- `uv run python -m pytest -q` — run the full test suite.
- `uv run python -m pytest tests/test_server_responses.py -q` — run a focused regression file.
- `uv run ruff check .` — run lint checks.

Use `--debug` when you need structured router logs in `llm_router.jsonl`.

## Coding Style & Naming Conventions

Target Python is 3.10+. Follow the existing style: 4-space indentation, type hints on public helpers, and short Google-style docstrings where useful. Keep modules narrowly scoped by responsibility. Prefer snake_case for functions, variables, and test names; use PascalCase for dataclasses and adapter/state-machine classes. Do not add broad compatibility hacks in `llm_client.py` when provider adapters can own the behavior.

## Source-of-Truth Rule

Behavioral changes in this repository must be grounded in three sources together:

- Codex runtime/source code, usually available locally at a path provided by the author
- provider API behavior, based on official developer documentation
- observed real-world behavior from router logs and live tests

When these disagree, do not guess. Reconcile the router against actual Codex request shapes first, then align provider adapters with the provider’s documented API contract, and finally verify with real requests or regression tests. For compatibility work, treat `llm_router.jsonl`, local Codex source, and provider docs as the primary evidence set.

## Testing Guidelines

This project uses `pytest`. New behavior should come with targeted regression tests in the nearest existing test file; avoid duplicate or purely structural tests. Name tests `test_<behavior>()`. For router bugs, prefer end-to-end request tests over isolated mocks unless the behavior is adapter-local. Run `uv run python -m pytest -q` before finishing.

## Commit & Pull Request Guidelines

Recent history uses short prefixes such as `feat:`, `fix:`, `refactor:`, `test:`, and `update:`. Keep commit messages imperative and scoped, for example: `fix: retry plan-mode clarifying questions via request_user_input`. PRs should explain the user-visible problem, the root cause, the files changed, and the verification performed. Include log snippets or screenshots only when they clarify transport/state-machine behavior.

## Security & Configuration Tips

Never commit real API keys. Use `api_key_env` in `router.toml` and environment variables such as `DEEPSEEK_API_KEY`. Treat `llm_router.jsonl` as debug output that may contain prompts and tool payloads; inspect locally and avoid sharing raw logs casually.
