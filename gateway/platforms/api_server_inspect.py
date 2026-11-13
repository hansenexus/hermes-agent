"""Read-only inspect endpoints on the API server (hansenexus fork patch).

Four GET endpoints for external dashboards and control planes (kommandant's
Hermes addon), all behind the same bearer auth as the rest of ``/v1``:

- ``GET /v1/jobs``          recent and active agent runs, dashboard-shaped
- ``GET /v1/mcp/status``    configured MCP servers with a 2 s reachability probe
- ``GET /v1/config``        allowlist-built, redacted config summary
- ``GET /v1/logs?lines=N``  ``journalctl --user`` tail of the gateway unit
                            (``platforms.api_server.extra.logs_unit`` /
                            ``API_SERVER_LOGS_UNIT``), clamped to 1000 lines

Kept in its own module so the fork's footprint on ``api_server.py`` is one import
and one ``routes.extend(...)`` line, which survives upstream rewrites of that file.
"""

import asyncio
import logging
import os
import time
from typing import Any

try:
    from aiohttp import web
except ImportError:
    web = None  # type: ignore[assignment]

logger = logging.getLogger("gateway.platforms.api_server")

DEFAULT_LOGS_UNIT = "hermes-gateway.service"
MAX_LOG_LINES = 1000
MCP_PROBE_TIMEOUT_S = 2.0
TASK_LABEL_MAX = 200

# Upstream run status vocabulary -> the dashboard contract's states (queued, running, paused,
# succeeded, failed).
JOB_STATUS_MAP = {
    "queued": "queued",
    "running": "running",
    "stopping": "running",
    "pausing": "running",  # still running until it reaches a safe boundary (hermes-agent#2)
    "paused": "paused",
    "waiting_for_approval": "running",
    "completed": "succeeded",
    "succeeded": "succeeded",
    "failed": "failed",
    "error": "failed",
    "cancelled": "failed",
    "interrupted": "failed",
    "stopped": "failed",
}
_TERMINAL_JOB_STATES = ("succeeded", "failed")


def run_preview_fields(user_message: Any) -> dict[str, str]:
    """Extra ``_set_run_status`` fields for a new ``/v1/runs`` run: the task label
    ``GET /v1/jobs`` shows. Multimodal (non-string) input gets no label."""
    if isinstance(user_message, str) and user_message:
        return {"input_preview": user_message[:TASK_LABEL_MAX]}
    return {}


def _load_config() -> dict:
    """The gateway user config; indirection point so tests can substitute one."""
    from gateway.run import _load_gateway_config

    return _load_gateway_config() or {}


def _logs_unit(adapter) -> str:
    extra = getattr(getattr(adapter, "config", None), "extra", None) or {}
    return str(extra.get("logs_unit") or os.getenv("API_SERVER_LOGS_UNIT") or DEFAULT_LOGS_UNIT)


def _mcp_enabled(entry: Any) -> bool:
    """v0.23 spells a switched-off server ``enabled: false``; older configs ``disabled: true``."""
    if not isinstance(entry, dict):
        return True
    if entry.get("enabled") is False:
        return False
    return entry.get("disabled") is not True


def _job_view(run_id: str, meta: dict, now: float) -> dict[str, Any]:
    job: dict[str, Any] = {
        "id": run_id,
        "task": str(meta.get("input_preview") or meta.get("last_event") or "agent run")[:TASK_LABEL_MAX],
        "status": JOB_STATUS_MAP.get(str(meta.get("status", "running")), "running"),
    }
    model = meta.get("model")
    if isinstance(model, str) and model:
        job["model"] = model
    created = meta.get("created_at")
    if created:
        job["createdAt"] = int(float(created) * 1000)
        ended = meta.get("updated_at") if job["status"] in _TERMINAL_JOB_STATES else now
        job["durationMs"] = max(0, int((float(ended or now) - float(created)) * 1000))
    return job


async def handle_jobs(adapter, request: "web.Request") -> "web.Response":
    """GET /v1/jobs - recent and active agent runs, newest first."""
    auth_err = adapter._check_auth(request)
    if auth_err:
        return auth_err
    now = time.time()
    from gateway.platforms.api_server_run_pause import overlay_status

    jobs = [_job_view(run_id, overlay_status(adapter, run_id, meta), now)
            for run_id, meta in list(adapter._run_statuses.items())]
    jobs.sort(key=lambda j: j.get("createdAt", 0), reverse=True)
    return web.json_response(jobs)


async def _probe_mcp(name: str, entry: Any) -> dict[str, Any]:
    url = entry.get("url") if isinstance(entry, dict) else None
    if not url:
        # stdio/command-launched server: no HTTP endpoint to probe.
        return {"name": name, "status": "degraded", "endpoint": "stdio/local"}
    import aiohttp

    try:
        timeout = aiohttp.ClientTimeout(total=MCP_PROBE_TIMEOUT_S)
        async with aiohttp.ClientSession(timeout=timeout) as sess:
            async with sess.get(url) as resp:
                # Any HTTP answer (incl. 401/404/405 on the MCP path) means the
                # server process is up and reachable.
                status = "connected" if resp.status < 500 else "degraded"
    except Exception:
        status = "disconnected"
    return {"name": name, "status": status, "endpoint": url}


async def handle_mcp_status(adapter, request: "web.Request") -> "web.Response":
    """GET /v1/mcp/status - enabled MCP servers with a reachability probe."""
    auth_err = adapter._check_auth(request)
    if auth_err:
        return auth_err
    servers = _load_config().get("mcp_servers") or {}
    results = await asyncio.gather(
        *(_probe_mcp(name, entry) for name, entry in servers.items() if _mcp_enabled(entry)))
    return web.json_response(list(results))


async def handle_config(adapter, request: "web.Request") -> "web.Response":
    """GET /v1/config - redacted config summary.

    Built from an allowlist, never by filtering the raw config, so api keys and
    tokens are excluded by construction.
    """
    auth_err = adapter._check_auth(request)
    if auth_err:
        return auth_err
    cfg = _load_config()
    model = cfg.get("model") or {}
    if not isinstance(model, dict):
        model = {"default": model}
    fallbacks = cfg.get("fallback_providers") or []
    fallback = fallbacks[0] if isinstance(fallbacks, list) and fallbacks else {}
    smart = cfg.get("smart_model_routing") or {}
    smart_cheap = smart.get("cheap_model")
    if isinstance(smart_cheap, dict):
        smart_cheap = smart_cheap.get("model")
    servers = cfg.get("mcp_servers") or {}
    skills = cfg.get("skills") or {}
    dashboard = cfg.get("dashboard") or {}
    out = {
        "primaryModel": model.get("default"),
        "primaryProvider": model.get("provider"),
        "contextLength": model.get("context_length"),
        "fallbackModel": fallback.get("model") if isinstance(fallback, dict) else None,
        "smartRouterModel": smart_cheap,
        "smartRoutingEnabled": bool(smart.get("enabled")),
        "mcpServerCount": sum(1 for entry in servers.values() if _mcp_enabled(entry)),
        "skillsDirCount": len(skills.get("external_dirs") or []),
        "apiPort": adapter._port,
        "dashboardPort": dashboard.get("port"),
        "storage": "sqlite",
        "configVersion": cfg.get("_config_version"),
    }
    return web.json_response({k: v for k, v in out.items() if v is not None})


async def handle_logs(adapter, request: "web.Request") -> "web.Response":
    """GET /v1/logs?lines=N - journalctl tail of the gateway's own unit."""
    auth_err = adapter._check_auth(request)
    if auth_err:
        return auth_err
    try:
        lines = int(request.query.get("lines", "100"))
    except ValueError:
        lines = 100
    lines = max(1, min(lines, MAX_LOG_LINES))
    env = dict(os.environ)
    # journalctl --user needs the user bus; harmless if already set.
    env.setdefault("XDG_RUNTIME_DIR", f"/run/user/{os.getuid()}")
    try:
        proc = await asyncio.create_subprocess_exec(
            "journalctl", "--user", "-u", _logs_unit(adapter),
            "-n", str(lines), "--no-pager", "-o", "cat",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
        )
        out, err = await asyncio.wait_for(proc.communicate(), timeout=10)
    except Exception as exc:
        logger.warning("GET /v1/logs: journalctl did not run: %s", exc)
        return web.json_response({"error": str(exc)}, status=500)
    if proc.returncode != 0:
        detail = (err or b"").decode(errors="replace")[:500]
        return web.json_response({"error": f"journalctl failed: {detail}"}, status=500)
    return web.json_response((out or b"").decode(errors="replace").splitlines())


_HANDLERS = (
    ("/v1/jobs", "handle_jobs"),
    ("/v1/mcp/status", "handle_mcp_status"),
    ("/v1/config", "handle_config"),
    ("/v1/logs", "handle_logs"),
)


def _bind(adapter, name: str):
    # Looked up at call time so test patches on this module take effect.
    async def _handler(request: "web.Request") -> "web.Response":
        return await globals()[name](adapter, request)

    _handler.__name__ = name
    return _handler


def _http_routes(adapter) -> list[tuple[str, str, Any]]:
    """(method, path, handler) rows for ``APIServerAdapter._http_route_table``."""
    return [("GET", path, _bind(adapter, name)) for path, name in _HANDLERS]
