# Windows UI Decoupling Design

## Goal

Keep `llm_router`'s core runtime focused on Codex-facing Responses routing,
provider adaptation, and local session state while making the Windows tray,
dashboard, and updater an optional, loosely coupled shell.

The Windows shell should be useful for local desktop operation, but it must not
become part of the core `/v1/responses` runtime contract.

## Evidence

- Local Codex source at `/Users/ansatz/data/code/codex` was refreshed with
  `git pull --ff-only` before this design was written.
- `codex features list` was checked. This design does not depend on
  experimental or under-development Codex features.
- Review of `6272bac..1c80f89` found the new Windows surface introduces risks
  around dashboard authentication, subprocess argument ordering, dirty-tree
  auto-updates, and tray actions depending on core server globals.
- Current code lets core server creation import `llm_router.windows.dashboard`
  when dashboard mode is enabled, and `serve --dashboard` registers UI routes
  on the same Flask app that handles router API traffic.

## Boundary Principles

Core owns:

- `/v1/responses`, `/v1/chat/completions`, `/v1/models`, and health endpoints
- route matching and upstream selection
- provider adapters
- Responses session state
- session store file format and cleanup primitives
- debug logging primitives

Windows shell owns:

- tray menu and notifications
- starting, stopping, and restarting the core server subprocess
- dashboard HTML and local management routes
- update checks and user-facing update prompts
- Windows-specific configuration defaults

Core must not import `llm_router.windows.*`.

Windows shell may import core configuration and storage primitives, but it must
not reach into `llm_router.server` globals such as `_config` or `_sessions`.

## Target Architecture

```text
llm-router serve
  -> core Flask app only
  -> router API endpoints
  -> no dashboard registration
  -> no Windows imports

llm-router gui
  -> Windows TrayApp
  -> ServerManager starts child `llm-router serve`
  -> ManagementDashboard runs in GUI process on loopback
  -> ManagementDashboard uses a random local token
  -> dashboard calls typed control services, not core Flask globals
```

### Control Services

Add a small control boundary that can be used by the Windows shell and future
non-Windows management UIs:

- `ServerManager`: process lifecycle for a child core server.
- `SessionControl`: inspect and clear the configured session store without
  importing `llm_router.server`.
- `LogControl`: read bounded debug logs from an explicit path.
- `UpdateControl`: check for updates and apply updates only under safe working
  tree conditions.

These services should be plain Python classes/functions. Flask routes and tray
menus are consumers, not owners, of the control behavior.

### Dashboard Security

The dashboard is a local management UI, not a public API.

Required behavior:

- bind the management dashboard to `127.0.0.1` only
- generate a random token when the GUI process starts
- include the token in dashboard URLs and mutating requests
- reject mutating requests without the token
- reject unsafe cross-origin requests where practical
- never expose raw debug logs unless the dashboard token is valid
- keep log reads bounded by line count and size

The core router server should not expose dashboard routes by default. If a
legacy `serve --dashboard` flag remains temporarily, it should be documented as
deprecated and should use the same local-token protection.

### Update Safety

Auto-update must not hide user work.

Required behavior:

- default `auto_update_enabled` to `false`, or make automatic checks
  notify-only
- never run `git stash` silently
- if the working tree is dirty, refuse automatic update and show a clear
  message
- only run `git pull --ff-only` when the working tree is clean
- keep `uv sync` failure visible as an update warning

Manual update can remain available, but it should obey the same dirty-tree
guardrail.

## Phases

### Phase 1: Stop Core/UI Coupling

- Remove dashboard registration from `llm_router.server.create_app`.
- Remove or deprecate `serve --dashboard`.
- Ensure `cmd_serve` remains core-only.
- Move dashboard serving under the Windows shell path.
- Fix `ServerManager` command construction so root CLI args precede the
  subcommand, for example `llm-router --config X serve`.

### Phase 2: Add Local Management Boundary

- Introduce control services for session store inspection/cleanup, bounded log
  reads, and server status.
- Update `TrayApp` so `Clear Sessions` uses explicit control services instead
  of `llm_router.server._config`.
- Update dashboard routes to call control services rather than server globals.
- Make `Open Dashboard` point at the management dashboard started by the GUI
  process, not at the core router API port unless explicitly configured.

### Phase 3: Harden Dashboard And Updater

- Add local token enforcement for dashboard routes.
- Restrict dashboard binding to loopback.
- Change auto-update to notify-only or disabled-by-default.
- Refuse dirty-tree updates instead of stashing.
- Surface update errors in tray notifications and dashboard status.

### Phase 4: Tests And Docs

- Add tests for `ServerManager._build_command` with and without `config_path`.
- Add tests for dashboard token rejection and acceptance.
- Add tests for dirty-tree updater refusal.
- Add tests that tray clear-session behavior does not depend on server globals.
- Update `docs/architecture.md` to state that Windows UI is an optional shell.
- Update `docs/testing.md` with Windows management-surface test commands.

## Non-Goals

- Do not change `/v1/responses` state-machine behavior.
- Do not change provider adapters.
- Do not add a public remote management API.
- Do not make dashboard routes available on non-loopback interfaces.
- Do not implement a separate package or repository in this phase.
- Do not add broad cross-platform GUI behavior before the Windows shell
  boundary is correct.

## Success Criteria

- Importing `llm_router.server` does not import any `llm_router.windows` module.
- Running `llm-router serve` exposes only core router API routes.
- Running `llm-router gui` can open a working dashboard without registering UI
  routes on the core server.
- Dashboard log and session-clear endpoints reject unauthenticated requests.
- Dirty working trees block update application and preserve user changes.
- Existing core router tests pass.
- New Windows management tests cover the reviewed regression risks.

## Open Questions

- Whether `serve --dashboard` should be removed immediately or kept for one
  release as a deprecated, token-protected compatibility path.
- Whether the management dashboard should always use a separate random port or
  a configured port from `[windows]`.
- Whether manual update should support an explicit "stash and restore" workflow
  later, after the safe notify-only path is implemented.
