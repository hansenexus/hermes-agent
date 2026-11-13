"""
Tests for the read-only inspect endpoints on the API server (hansenexus fork patch).

Covers:
- Route registration in ``APIServerAdapter._http_route_table``
- GET /v1/jobs (status mapping incl. v0.23 statuses, task label precedence, newest-first sort)
- POST /v1/runs stamps the task label /v1/jobs shows
- GET /v1/mcp/status (stdio entries report degraded, disabled servers skipped, empty config -> [])
- GET /v1/config (allowlist-built redaction, port sourcing)
- GET /v1/logs (lines clamp, configurable unit, journalctl failure surfaces as 500)
- Auth enforcement (401 without bearer when API_SERVER_KEY is set)
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.config import PlatformConfig
from gateway.platforms import api_server_inspect as inspect_mod
from gateway.platforms.api_server import APIServerAdapter, cors_middleware

_MOD = "gateway.platforms.api_server_inspect"
_INSPECT_PATHS = ("/v1/jobs", "/v1/mcp/status", "/v1/config", "/v1/logs")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_adapter(api_key: str = "", extra: dict = None) -> APIServerAdapter:
    merged = dict(extra or {})
    if api_key:
        merged["key"] = api_key
    return APIServerAdapter(PlatformConfig(enabled=True, extra=merged))


def _create_app(adapter: APIServerAdapter) -> web.Application:
    app = web.Application(middlewares=[cors_middleware])
    for method, path, handler in inspect_mod._http_routes(adapter):
        app.router.add_route(method, path, handler)
    return app


@pytest.fixture
def adapter():
    return _make_adapter()


@pytest.fixture
def auth_adapter():
    return _make_adapter(api_key="sk-secret")


def _patch_gateway_config(cfg: dict):
    return patch(f"{_MOD}._load_config", return_value=cfg)


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

def test_inspect_routes_registered_on_route_table(adapter):
    table = {(method, path) for method, path, _ in adapter._http_route_table()}
    for path in _INSPECT_PATHS:
        assert ("GET", path) in table


def test_run_preview_fields():
    assert inspect_mod.run_preview_fields("x" * 500) == {"input_preview": "x" * 200}
    assert inspect_mod.run_preview_fields([{"type": "text", "text": "hi"}]) == {}
    assert inspect_mod.run_preview_fields("") == {}


# ---------------------------------------------------------------------------
# /v1/jobs
# ---------------------------------------------------------------------------

class TestInspectJobs:
    @pytest.mark.asyncio
    async def test_empty(self, adapter):
        async with TestClient(TestServer(_create_app(adapter))) as cli:
            resp = await cli.get("/v1/jobs")
            assert resp.status == 200
            assert await resp.json() == []

    @pytest.mark.asyncio
    async def test_status_mapping_and_sort(self, adapter):
        adapter._set_run_status("run-old", "completed", created_at=100.0, input_preview="old task")
        adapter._set_run_status("run-new", "running", created_at=200.0, model="prov/model-x")
        adapter._set_run_status("run-bad", "failed", created_at=150.0)
        async with TestClient(TestServer(_create_app(adapter))) as cli:
            resp = await cli.get("/v1/jobs")
            assert resp.status == 200
            jobs = await resp.json()
        assert [j["id"] for j in jobs] == ["run-new", "run-bad", "run-old"]
        by_id = {j["id"]: j for j in jobs}
        assert by_id["run-old"]["status"] == "succeeded"
        assert by_id["run-new"]["status"] == "running"
        assert by_id["run-bad"]["status"] == "failed"
        assert by_id["run-old"]["task"] == "old task"
        assert by_id["run-new"]["model"] == "prov/model-x"
        assert by_id["run-old"]["createdAt"] == 100_000
        assert by_id["run-old"]["durationMs"] >= 0

    @pytest.mark.asyncio
    @pytest.mark.parametrize("raw, expected", [
        ("waiting_for_approval", "running"),
        ("stopping", "running"),
        ("interrupted", "failed"),
        ("cancelled", "failed"),
        ("some-future-status", "running"),
    ])
    async def test_v023_status_vocabulary(self, adapter, raw, expected):
        adapter._set_run_status("run-x", raw, created_at=1.0)
        async with TestClient(TestServer(_create_app(adapter))) as cli:
            jobs = await (await cli.get("/v1/jobs")).json()
        assert jobs[0]["status"] == expected

    @pytest.mark.asyncio
    async def test_task_falls_back_to_last_event(self, adapter):
        adapter._set_run_status("run-y", "running", created_at=1.0, last_event="tool.started")
        async with TestClient(TestServer(_create_app(adapter))) as cli:
            jobs = await (await cli.get("/v1/jobs")).json()
        assert jobs[0]["task"] == "tool.started"


# ---------------------------------------------------------------------------
# /v1/mcp/status
# ---------------------------------------------------------------------------

class TestInspectMcpStatus:
    @pytest.mark.asyncio
    async def test_empty_config(self, adapter):
        async with TestClient(TestServer(_create_app(adapter))) as cli:
            with _patch_gateway_config({}):
                resp = await cli.get("/v1/mcp/status")
                assert resp.status == 200
                assert await resp.json() == []

    @pytest.mark.asyncio
    async def test_stdio_server_reports_degraded(self, adapter):
        cfg = {"mcp_servers": {"local-tool": {"command": "some-binary"}}}
        async with TestClient(TestServer(_create_app(adapter))) as cli:
            with _patch_gateway_config(cfg):
                data = await (await cli.get("/v1/mcp/status")).json()
        assert data == [{"name": "local-tool", "status": "degraded", "endpoint": "stdio/local"}]

    @pytest.mark.asyncio
    async def test_disabled_servers_skipped(self, adapter):
        cfg = {"mcp_servers": {
            "on": {"command": "a"},
            "off-new": {"command": "b", "enabled": False},
            "off-legacy": {"command": "c", "disabled": True},
        }}
        async with TestClient(TestServer(_create_app(adapter))) as cli:
            with _patch_gateway_config(cfg):
                data = await (await cli.get("/v1/mcp/status")).json()
        assert [d["name"] for d in data] == ["on"]

    @pytest.mark.asyncio
    async def test_unreachable_url_reports_disconnected(self, adapter):
        # Reserved TEST-NET address: connection fails within the 2s timeout.
        cfg = {"mcp_servers": {"dead": {"url": "http://192.0.2.1:1/mcp"}}}
        async with TestClient(TestServer(_create_app(adapter))) as cli:
            with _patch_gateway_config(cfg):
                data = await (await cli.get("/v1/mcp/status")).json()
        assert data == [{"name": "dead", "status": "disconnected", "endpoint": "http://192.0.2.1:1/mcp"}]


# ---------------------------------------------------------------------------
# /v1/config
# ---------------------------------------------------------------------------

SECRETLY_CONFIGURED = {
    "model": {"default": "prov/model-a", "provider": "openrouter", "context_length": 131072},
    "fallback_providers": [{"model": "prov/model-b", "api_key": "sk-FALLBACK-SECRET"}],
    "smart_model_routing": {"enabled": True, "cheap_model": {"model": "prov/cheap"}},
    "mcp_servers": {"a": {"url": "http://x"}, "b": {"command": "y"}, "c": {"command": "z", "enabled": False}},
    "skills": {"external_dirs": ["/some/dir"]},
    "dashboard": {"port": 9119},
    "api_keys": {"openrouter": "sk-TOP-SECRET"},
    "telegram": {"bot_token": "123:SECRET-TOKEN"},
    "_config_version": 7,
}


class TestInspectConfig:
    @pytest.mark.asyncio
    async def test_allowlist_redaction(self, adapter):
        async with TestClient(TestServer(_create_app(adapter))) as cli:
            with _patch_gateway_config(SECRETLY_CONFIGURED):
                resp = await cli.get("/v1/config")
                assert resp.status == 200
                data = await resp.json()
        assert data["primaryModel"] == "prov/model-a"
        assert data["primaryProvider"] == "openrouter"
        assert data["contextLength"] == 131072
        assert data["fallbackModel"] == "prov/model-b"
        assert data["smartRouterModel"] == "prov/cheap"
        assert data["smartRoutingEnabled"] is True
        assert data["mcpServerCount"] == 2
        assert data["skillsDirCount"] == 1
        assert data["apiPort"] == adapter._port
        assert data["dashboardPort"] == 9119
        assert data["configVersion"] == 7
        # Allowlist-built: no secret material can appear in the payload.
        serialized = json.dumps(data)
        assert "SECRET" not in serialized
        assert "bot_token" not in serialized
        assert "api_key" not in serialized

    @pytest.mark.asyncio
    async def test_empty_config_drops_none_fields(self, adapter):
        async with TestClient(TestServer(_create_app(adapter))) as cli:
            with _patch_gateway_config({}):
                data = await (await cli.get("/v1/config")).json()
        assert data["apiPort"] == adapter._port
        assert data["storage"] == "sqlite"
        assert data["smartRoutingEnabled"] is False
        assert "primaryModel" not in data
        assert "dashboardPort" not in data

    @pytest.mark.asyncio
    async def test_bare_string_model(self, adapter):
        async with TestClient(TestServer(_create_app(adapter))) as cli:
            with _patch_gateway_config({"model": "prov/bare"}):
                data = await (await cli.get("/v1/config")).json()
        assert data["primaryModel"] == "prov/bare"


# ---------------------------------------------------------------------------
# /v1/logs
# ---------------------------------------------------------------------------

def _mock_journal_proc(stdout: bytes = b"", stderr: bytes = b"", returncode: int = 0):
    proc = MagicMock()
    proc.communicate = AsyncMock(return_value=(stdout, stderr))
    proc.returncode = returncode
    return proc


class TestInspectLogs:
    @pytest.mark.asyncio
    async def test_tail_lines(self, adapter):
        exec_mock = AsyncMock(return_value=_mock_journal_proc(b"line1\nline2"))
        async with TestClient(TestServer(_create_app(adapter))) as cli:
            with patch(f"{_MOD}.asyncio.create_subprocess_exec", exec_mock):
                resp = await cli.get("/v1/logs?lines=2")
                assert resp.status == 200
                assert await resp.json() == ["line1", "line2"]
        args = exec_mock.call_args.args
        assert args[:4] == ("journalctl", "--user", "-u", inspect_mod.DEFAULT_LOGS_UNIT)
        assert "2" in args

    @pytest.mark.asyncio
    async def test_lines_clamped_to_1000(self, adapter):
        exec_mock = AsyncMock(return_value=_mock_journal_proc(b""))
        async with TestClient(TestServer(_create_app(adapter))) as cli:
            with patch(f"{_MOD}.asyncio.create_subprocess_exec", exec_mock):
                await cli.get("/v1/logs?lines=999999")
        assert "1000" in exec_mock.call_args.args

    @pytest.mark.asyncio
    async def test_bad_lines_defaults_to_100(self, adapter):
        exec_mock = AsyncMock(return_value=_mock_journal_proc(b""))
        async with TestClient(TestServer(_create_app(adapter))) as cli:
            with patch(f"{_MOD}.asyncio.create_subprocess_exec", exec_mock):
                await cli.get("/v1/logs?lines=abc")
        assert "100" in exec_mock.call_args.args

    @pytest.mark.asyncio
    async def test_configurable_unit(self):
        adapter = _make_adapter(extra={"logs_unit": "my-gateway.service"})
        exec_mock = AsyncMock(return_value=_mock_journal_proc(b""))
        async with TestClient(TestServer(_create_app(adapter))) as cli:
            with patch(f"{_MOD}.asyncio.create_subprocess_exec", exec_mock):
                await cli.get("/v1/logs")
        assert "my-gateway.service" in exec_mock.call_args.args

    @pytest.mark.asyncio
    async def test_unit_from_env(self, adapter, monkeypatch):
        monkeypatch.setenv("API_SERVER_LOGS_UNIT", "env-gateway.service")
        exec_mock = AsyncMock(return_value=_mock_journal_proc(b""))
        async with TestClient(TestServer(_create_app(adapter))) as cli:
            with patch(f"{_MOD}.asyncio.create_subprocess_exec", exec_mock):
                await cli.get("/v1/logs")
        assert "env-gateway.service" in exec_mock.call_args.args

    @pytest.mark.asyncio
    async def test_journalctl_failure_is_500(self, adapter):
        exec_mock = AsyncMock(return_value=_mock_journal_proc(b"", b"boom", 1))
        async with TestClient(TestServer(_create_app(adapter))) as cli:
            with patch(f"{_MOD}.asyncio.create_subprocess_exec", exec_mock):
                resp = await cli.get("/v1/logs")
                assert resp.status == 500
                assert "journalctl failed" in (await resp.json())["error"]


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------

class TestInspectAuth:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("path", _INSPECT_PATHS)
    async def test_401_without_bearer(self, auth_adapter, path):
        async with TestClient(TestServer(_create_app(auth_adapter))) as cli:
            resp = await cli.get(path)
            assert resp.status == 401

    @pytest.mark.asyncio
    async def test_200_with_bearer(self, auth_adapter):
        async with TestClient(TestServer(_create_app(auth_adapter))) as cli:
            resp = await cli.get("/v1/jobs", headers={"Authorization": "Bearer sk-secret"})
            assert resp.status == 200


# ---------------------------------------------------------------------------
# POST /v1/runs -> /v1/jobs task label
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_post_run_stamps_task_label(adapter):
    app = _create_app(adapter)
    app.router.add_post("/v1/runs", adapter._handle_runs)
    async with TestClient(TestServer(app)) as cli:
        with patch.object(adapter, "_create_agent") as mock_create:
            mock_agent = MagicMock()
            mock_agent.run_conversation.return_value = {"final_response": "done"}
            mock_create.return_value = mock_agent
            resp = await cli.post("/v1/runs", json={"input": "summarise the inbox"})
            assert resp.status == 202
            run_id = (await resp.json())["run_id"]
            jobs = await (await cli.get("/v1/jobs")).json()
    assert [j["task"] for j in jobs if j["id"] == run_id] == ["summarise the inbox"]
