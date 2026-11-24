"""Claude Code provider wiring in llm_core.

The transport is inference only: Odysseus' native tool schemas are never passed
to Claude Code, and a request runs only with the capability token that the
owner-scoped endpoint resolution puts in the session headers.
"""

import asyncio
import json
from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest

from src import claude_code, llm_core

URL = claude_code.BASE_URL
GOOD = {"Authorization": "Bearer cap-good"}
MESSAGES = [
    {"role": "system", "content": "You are Odysseus. Use <tool_call> to call tools."},
    {"role": "user", "content": "hi"},
]
TOOLS = [{"type": "function", "function": {"name": "run_shell", "parameters": {"type": "object"}}}]


@pytest.fixture
def provider(monkeypatch):
    state = SimpleNamespace(calls=[], events=[
        ("model", "claude-opus-5-5"),
        ("thinking", "planning"),
        ("text", "Hel"),
        ("text", "lo"),
        ("usage", {"input_tokens": 1200, "output_tokens": 7}),
    ])

    async def fake_stream(model, messages, **kwargs):
        state.calls.append({"model": model, "messages": messages, **kwargs})
        for event in state.events:
            if isinstance(event, Exception):
                raise event
            yield event

    monkeypatch.setattr(claude_code, "stream_events", fake_stream)
    monkeypatch.setattr(claude_code, "capability_is_valid", lambda token: token == "cap-good")
    return state


def _stream(headers=GOOD, **kwargs):
    async def run():
        return [
            chunk async for chunk in llm_core._stream_llm_inner(
                URL, "claude-opus-5-5", MESSAGES, headers=headers, **kwargs,
            )
        ]
    return asyncio.run(run())


def _data(chunks):
    out = []
    for chunk in chunks:
        for line in chunk.splitlines():
            if line.startswith("data:") and line[5:].strip() != "[DONE]":
                out.append(json.loads(line[5:]))
    return out


def test_provider_is_detected_and_labelled_from_its_url():
    assert llm_core._detect_provider(URL) == "claude-code"
    assert llm_core._provider_label(URL) == "Claude Code"
    assert llm_core._stream_target_url(URL) == URL


def test_stream_maps_provider_events_to_odysseus_sse(provider):
    chunks = _stream()
    assert _data(chunks) == [
        {"delta": "planning", "thinking": True},
        {"delta": "Hel"},
        {"delta": "lo"},
        {"type": "usage", "data": {"input_tokens": 1200, "output_tokens": 7, "model": "claude-opus-5-5"}},
    ]
    assert chunks[-1] == "data: [DONE]\n\n"


def test_stream_reports_a_different_resolved_model(provider):
    provider.events[0] = ("model", "claude-opus-5-6")
    data = _data(_stream())
    assert data[0] == {"type": "model_actual", "requested_model": "claude-opus-5-5", "model": "claude-opus-5-6"}
    assert data[-1]["data"]["requested_model"] == "claude-opus-5-5"


def test_native_tools_are_never_forwarded(provider):
    _stream(tools=TOOLS, reasoning_effort="high")
    call = provider.calls[0]
    assert "tools" not in call
    assert call["effort"] == "high"
    assert call["messages"][0]["content"].startswith("You are Odysseus.")


@pytest.mark.parametrize("headers", [None, {}, {"Authorization": "Bearer stolen"}, {"x-api-key": "cap-good"}])
def test_stream_refuses_without_this_owners_capability(provider, headers):
    chunks = _stream(headers=headers)
    assert provider.calls == []
    assert chunks[0].startswith("event: error")
    payload = json.loads(chunks[0].split("data:", 1)[1])
    assert payload["status"] == 403
    assert payload["fallback_eligible"] is False
    assert "cap-good" not in chunks[0]


def test_provider_errors_become_stream_errors(provider):
    provider.events[:] = [
        ("text", "partial"),
        claude_code.ClaudeCodeError(429, claude_code.RATE_LIMITED),
    ]
    chunks = _stream()
    error = chunks[-1]
    assert error.startswith("event: error")
    assert json.loads(error.split("data:", 1)[1]) == {"status": 429, "text": claude_code.RATE_LIMITED}


def test_non_retryable_provider_errors_disable_fallback(provider):
    provider.events[:] = [claude_code.ClaudeCodeError(503, claude_code.NOT_INSTALLED, fallback_eligible=False)]
    payload = json.loads(_stream()[-1].split("data:", 1)[1])
    assert payload["fallback_eligible"] is False


def test_non_streaming_callers_get_the_collected_text(provider, monkeypatch):
    @asynccontextmanager
    async def no_slot(*_args, **_kwargs):
        yield

    monkeypatch.setattr(llm_core, "_local_model_slot", no_slot)
    text = asyncio.run(llm_core.llm_call_async(URL, "claude-opus-5-5", MESSAGES, headers=GOOD))
    assert text == "Hello"


class SilentCli:
    """A Claude Code CLI that goes quiet at startup or mid-response."""

    def __init__(self, stall_at):
        self.stall_at = stall_at
        self.calls = []

    async def connect(self):
        self.calls.append("connect")
        if self.stall_at == "connect":
            await asyncio.Event().wait()

    async def get_server_info(self):
        return {"account": {"apiProvider": "firstParty", "subscriptionType": "max"}}

    async def query(self, prompt):
        self.calls.append("query")

    async def receive_response(self):
        await asyncio.Event().wait()
        yield None

    async def interrupt(self):
        self.calls.append("interrupt")

    async def disconnect(self):
        self.calls.append("disconnect")


@pytest.mark.parametrize("stall_at", ["connect", "response"])
def test_a_silent_cli_times_out_like_any_stream_and_is_ended(monkeypatch, stall_at):
    cli = SilentCli(stall_at)
    monkeypatch.setattr(claude_code, "_sdk_client", lambda options: cli)
    monkeypatch.setattr(claude_code, "capability_is_valid", lambda token: token == "cap-good")

    async def run():
        return [
            chunk async for chunk in llm_core.stream_llm(
                URL, "claude-opus-5-5", MESSAGES, headers=GOOD, timeout=0.05,
            )
        ]

    chunks = asyncio.run(asyncio.wait_for(run(), 5))
    assert chunks[-1].startswith("event: error")
    assert json.loads(chunks[-1].split("data:", 1)[1])["status"] == 504
    assert cli.calls[-1] == "disconnect"


def test_model_list_never_probes_http(monkeypatch):
    monkeypatch.setattr(llm_core, "_configured_cached_model_ids", lambda *a, **k: [])
    monkeypatch.setattr(llm_core, "httpx_get_kimi_aware", lambda *a, **k: pytest.fail("no HTTP for Claude Code"))
    assert llm_core.list_model_ids(URL) == []
