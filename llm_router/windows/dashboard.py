"""Flask Blueprint for the web-based dashboard."""

from __future__ import annotations

import time
from pathlib import Path

from flask import Blueprint, jsonify, render_template_string

dashboard_bp = Blueprint("dashboard", __name__)

_start_time = time.time()

_DASHBOARD_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>LLM Router Dashboard</title>
<style>
  :root { --bg: #1a1a2e; --card: #16213e; --accent: #0f3460; --text: #e0e0e0; --green: #00e676; --red: #ff5252; --yellow: #ffd740; --border: #2a2a4a; }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background: var(--bg); color: var(--text); padding: 20px; }
  h1 { font-size: 1.5rem; margin-bottom: 8px; }
  .subtitle { color: #888; font-size: 0.85rem; margin-bottom: 20px; }
  .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 16px; margin-bottom: 24px; }
  .card { background: var(--card); border: 1px solid var(--border); border-radius: 8px; padding: 16px; }
  .card h3 { font-size: 0.75rem; text-transform: uppercase; color: #888; margin-bottom: 8px; }
  .card .value { font-size: 1.8rem; font-weight: 700; }
  .dot { display: inline-block; width: 10px; height: 10px; border-radius: 50%; margin-right: 6px; }
  .dot.green { background: var(--green); }
  .dot.red { background: var(--red); }
  .dot.yellow { background: var(--yellow); }
  .section { background: var(--card); border: 1px solid var(--border); border-radius: 8px; padding: 16px; margin-bottom: 20px; }
  .section h2 { font-size: 1rem; margin-bottom: 12px; }
  .routes-table { width: 100%; border-collapse: collapse; font-size: 0.85rem; }
  .routes-table th, .routes-table td { text-align: left; padding: 6px 10px; border-bottom: 1px solid var(--border); }
  .routes-table th { color: #888; font-weight: 600; }
  #log-box { background: #0d1117; border: 1px solid var(--border); border-radius: 6px; padding: 12px; font-family: "Cascadia Code", "Fira Code", monospace; font-size: 0.78rem; max-height: 400px; overflow-y: auto; white-space: pre-wrap; word-break: break-all; line-height: 1.5; }
  .btn { background: var(--accent); color: var(--text); border: 1px solid var(--border); border-radius: 6px; padding: 8px 16px; cursor: pointer; font-size: 0.85rem; }
  .btn:hover { background: #1a4a8a; }
  .btn.danger { background: #8b0000; }
  .btn.danger:hover { background: #b22222; }
  .actions { display: flex; gap: 10px; margin-top: 12px; }
  .health-list { list-style: none; }
  .health-list li { padding: 4px 0; font-size: 0.9rem; }
</style>
</head>
<body>
<h1>LLM Router Dashboard</h1>
<div class="subtitle" id="subtitle">Loading...</div>

<div class="grid">
  <div class="card">
    <h3>Status</h3>
    <div class="value" id="status-val">--</div>
  </div>
  <div class="card">
    <h3>Uptime</h3>
    <div class="value" id="uptime-val">--</div>
  </div>
  <div class="card">
    <h3>Sessions</h3>
    <div class="value" id="sessions-val">--</div>
  </div>
  <div class="card">
    <h3>Disk Usage</h3>
    <div class="value" id="disk-val">--</div>
  </div>
</div>

<div class="section">
  <h2>Upstream Health</h2>
  <ul class="health-list" id="health-list">
    <li>Checking...</li>
  </ul>
</div>

<div class="section">
  <h2>Routes</h2>
  <table class="routes-table">
    <thead><tr><th>Pattern</th><th>Type</th><th>Upstream</th></tr></thead>
    <tbody id="routes-body"></tbody>
  </table>
</div>

<div class="section">
  <h2>Recent Logs</h2>
  <div id="log-box">Waiting for logs...</div>
  <div class="actions">
    <button class="btn" onclick="fetchLogs()">Refresh Logs</button>
    <button class="btn danger" onclick="clearSessions()">Clear All Sessions</button>
  </div>
</div>

<script>
function fmt(seconds) {
  const d = Math.floor(seconds / 86400);
  const h = Math.floor((seconds % 86400) / 3600);
  const m = Math.floor((seconds % 3600) / 60);
  const s = Math.floor(seconds % 60);
  if (d > 0) return d + "d " + h + "h";
  if (h > 0) return h + "h " + m + "m";
  if (m > 0) return m + "m " + s + "s";
  return s + "s";
}

async function fetchStatus() {
  try {
    const r = await fetch("/dashboard/api/status");
    const d = await r.json();
    const dot = d.running ? "green" : "red";
    document.getElementById("status-val").innerHTML = '<span class="dot ' + dot + '"></span>' + (d.running ? "Running" : "Stopped");
    document.getElementById("uptime-val").textContent = d.running ? fmt(d.uptime) : "--";
    document.getElementById("subtitle").textContent = "v" + d.version + " | PID " + (d.pid || "N/A");
  } catch(e) { document.getElementById("status-val").innerHTML = '<span class="dot red"></span>Error'; }
}

async function fetchSessions() {
  try {
    const r = await fetch("/dashboard/api/sessions");
    const d = await r.json();
    document.getElementById("sessions-val").textContent = d.session_count;
    const kb = d.store_size_bytes / 1024;
    document.getElementById("disk-val").textContent = kb > 1024 ? (kb/1024).toFixed(1) + " MB" : kb.toFixed(1) + " KB";
  } catch(e) {}
}

async function fetchConfig() {
  try {
    const r = await fetch("/dashboard/api/config");
    const d = await r.json();
    const tbody = document.getElementById("routes-body");
    tbody.innerHTML = "";
    (d.routes || []).forEach(function(rt) {
      const tr = document.createElement("tr");
      tr.innerHTML = "<td>" + rt.pattern + "</td><td>" + rt.type + "</td><td>" + rt.upstream + "</td>";
      tbody.appendChild(tr);
    });
    const hlist = document.getElementById("health-list");
    hlist.innerHTML = "";
    (d.upstreams || []).forEach(function(u) {
      const li = document.createElement("li");
      li.innerHTML = '<span class="dot green"></span>' + u.name + " — " + u.base_url;
      hlist.appendChild(li);
    });
  } catch(e) {}
}

async function fetchLogs() {
  try {
    const r = await fetch("/dashboard/api/logs");
    const d = await r.json();
    const box = document.getElementById("log-box");
    box.textContent = (d.logs || []).join("\n");
    box.scrollTop = box.scrollHeight;
  } catch(e) {}
}

async function clearSessions() {
  if (!confirm("Delete ALL session history? This cannot be undone.")) return;
  try {
    const r = await fetch("/dashboard/clear-sessions", { method: "POST" });
    const d = await r.json();
    alert(d.message || "Done");
    fetchSessions();
  } catch(e) { alert("Failed: " + e); }
}

fetchStatus(); fetchSessions(); fetchConfig(); fetchLogs();
setInterval(function() { fetchStatus(); fetchSessions(); }, 5000);
setInterval(fetchLogs, 10000);
</script>
</body>
</html>"""


@dashboard_bp.route("/dashboard")
def dashboard_page():
    return render_template_string(_DASHBOARD_HTML)


@dashboard_bp.route("/dashboard/api/status")
def api_status():
    from llm_router import __version__
    return jsonify({
        "running": True,
        "version": __version__,
        "pid": None,
        "uptime": time.time() - _start_time,
    })


@dashboard_bp.route("/dashboard/api/sessions")
def api_sessions():
    from llm_router.server import _sessions
    if _sessions is None:
        return jsonify({"session_count": 0, "total_items": 0, "store_size_bytes": 0})
    return jsonify(_sessions.stats())


@dashboard_bp.route("/dashboard/api/config")
def api_config():
    from llm_router.server import _config
    if _config is None:
        return jsonify({"upstreams": [], "routes": []})

    upstreams = []
    for name, up in _config.upstreams.items():
        upstreams.append({"name": name, "base_url": up.base_url})

    routes = []
    for r in _config.routes:
        routes.append({
            "pattern": r.pattern,
            "type": r.model_type,
            "upstream": r.upstream,
        })

    return jsonify({
        "server_host": _config.server_host,
        "server_port": _config.server_port,
        "upstreams": upstreams,
        "routes": routes,
    })


@dashboard_bp.route("/dashboard/api/logs")
def api_logs():
    lines = []
    log_file = Path("llm_router.jsonl")
    if log_file.exists():
        try:
            with open(log_file, encoding="utf-8", errors="replace") as f:
                all_lines = f.readlines()
                lines = [line.rstrip() for line in all_lines[-100:]]
        except Exception:
            pass
    return jsonify({"logs": lines})


@dashboard_bp.route("/dashboard/clear-sessions", methods=["POST"])
def clear_sessions():
    from llm_router.server import _sessions
    if _sessions is None:
        return jsonify({"message": "No session store loaded"}), 400
    count = _sessions.clear_all()
    return jsonify({"message": f"Deleted {count} sessions"})
