"""
Tests for opt-in model passthrough on /v1/chat/completions.

A provider-scoped request model (contains "/") overrides the configured
gateway model for that request only, and only when the platform config
enables ``allow_model_override``. Bare logical names never override.
"""

from unittest.mock import AsyncMock, patch

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.config import PlatformConfig
from gateway.platforms.api_server import APIServerAdapter, cors_middleware

_RESULT = (
    {"final_response": "ok", "completed": True},
    {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
)


def _make_adapter(allow: bool) -> APIServerAdapter:
    extra = {"allow_model_override": allow} if allow else {}
    return APIServerAdapter(PlatformConfig(enabled=True, extra=extra))


def _create_app(adapter: APIServerAdapter) -> web.Application:
    app = web.Application(middlewares=[cors_middleware])
    app["api_server_adapter"] = adapter
    app.router.add_post("/v1/chat/completions", adapter._handle_chat_completions)
    return app


async def _post_chat(adapter: APIServerAdapter, model: str):
    run_mock = AsyncMock(return_value=_RESULT)
    app = _create_app(adapter)
    async with TestClient(TestServer(app)) as cli:
        with patch.object(adapter, "_run_agent", run_mock):
            resp = await cli.post(
                "/v1/chat/completions",
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": "hi"}],
                    "stream": False,
                },
            )
            assert resp.status == 200, await resp.text()
    return run_mock


class TestModelOverride:
    @pytest.mark.asyncio
    async def test_flag_off_never_overrides(self):
        adapter = _make_adapter(allow=False)
        run_mock = await _post_chat(adapter, "openrouter/auto")
        assert run_mock.call_args.kwargs["model_override"] is None

    @pytest.mark.asyncio
    async def test_flag_on_provider_scoped_id_overrides(self):
        adapter = _make_adapter(allow=True)
        run_mock = await _post_chat(adapter, "anthropic/claude-sonnet-4.5")
        assert (
            run_mock.call_args.kwargs["model_override"]
            == "anthropic/claude-sonnet-4.5"
        )

    @pytest.mark.asyncio
    async def test_flag_on_bare_name_keeps_default(self):
        adapter = _make_adapter(allow=True)
        run_mock = await _post_chat(adapter, "hermes-agent")
        assert run_mock.call_args.kwargs["model_override"] is None

    def test_flag_parsing(self):
        assert _make_adapter(allow=True)._allow_model_override is True
        assert _make_adapter(allow=False)._allow_model_override is False
        truthy = APIServerAdapter(
            PlatformConfig(enabled=True, extra={"allow_model_override": "yes"})
        )
        assert truthy._allow_model_override is True
