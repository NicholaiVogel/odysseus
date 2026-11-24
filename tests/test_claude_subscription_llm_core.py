"""Exercise the public llm_core seam through real local upstream HTTP."""

import asyncio
import json

import pytest
from fastapi import HTTPException

from src import claude_subscription as provider
from src import llm_core
from src.prompt_security import untrusted_context_message
from tests.test_claude_subscription_direct import tool
from tests.test_claude_subscription_http import frame, reply, upstream


@pytest.mark.asyncio
async def test_llm_core_stream_routes_tools_and_completion(monkeypatch):
    monkeypatch.setenv("ODYSSEUS_CLAUDE_SUBSCRIPTION_TOKEN", "explicit-fake")
    monkeypatch.delenv("ODYSSEUS_CLAUDE_SUBSCRIPTION_CREDENTIALS_FILE", raising=False)
    monkeypatch.setattr(provider, "capability_is_valid", lambda token: token == "cap")

    async def respond(reader, writer, requests):
        if json.loads(requests[-1][2]).get("stream"):
            data = frame(
                "message_start",
                message={"model": "actual", "usage": {"input_tokens": 4}},
            )
            data += frame(
                "content_block_start",
                index=0,
                content_block={
                    "type": "tool_use",
                    "name": "mcp__lookup",
                    "id": "a",
                    "input": {},
                },
            )
            data += frame(
                "content_block_delta",
                index=0,
                delta={"type": "input_json_delta", "partial_json": '{"q":"ok"}'},
            )
            data += frame("content_block_stop", index=0) + frame("message_stop")
            await reply(writer, data, content_type="text/event-stream")
        else:
            await reply(
                writer,
                {"model": "actual", "content": [{"type": "text", "text": "answer"}]},
            )

    async with upstream(respond) as (url, requests):
        monkeypatch.setattr(provider, "MESSAGES_URL", url)
        chunks = [
            x
            async for x in llm_core.stream_llm(
                provider.BASE_URL,
                "claude-sonnet-4-6",
                [{"role": "user", "content": "hello"}],
                headers={"Authorization": "Bearer cap"},
                tools=[tool("lookup")],
            )
        ]
        result = await llm_core.llm_call_async(
            provider.BASE_URL,
            "claude-sonnet-4-6",
            [{"role": "user", "content": "hello"}],
            headers={"Authorization": "Bearer cap"},
            return_model_metadata=True,
        )
    assert chunks[-1] == "data: [DONE]\n\n"
    events = [json.loads(c.split("data: ", 1)[1]) for c in chunks[:-1]]
    assert (
        next(e for e in events if e.get("type") == "tool_calls")["calls"][0]["name"]
        == "lookup"
    )
    assert result == ("answer", "actual")
    assert len(requests) == 2
    assert all(r[1]["authorization"] == "Bearer explicit-fake" for r in requests)


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [True, False])
async def test_agent_context_and_tool_history_reach_upstream(monkeypatch, streaming):
    monkeypatch.setenv("ODYSSEUS_CLAUDE_SUBSCRIPTION_TOKEN", "explicit-fake")
    monkeypatch.delenv("ODYSSEUS_CLAUDE_SUBSCRIPTION_CREDENTIALS_FILE", raising=False)
    monkeypatch.setattr(provider, "capability_is_valid", lambda token: token == "cap")
    context = untrusted_context_message("web search", "synthetic result")
    assistant = {
        "role": "assistant",
        "content": None,
        "reasoning_content": "local reasoning, not native thinking",
        "tool_calls": [
            {
                "id": "a",
                "type": "function",
                "function": {"name": "lookup", "arguments": "{}"},
            }
        ],
    }
    messages = [
        context,
        {"role": "user", "content": "summarize"},
        assistant,
        {"role": "tool", "tool_call_id": "a", "content": "found"},
    ]

    async def respond(reader, writer, requests):
        if streaming:
            data = frame("message_start", message={}) + frame("message_stop")
            await reply(writer, data, content_type="text/event-stream")
        else:
            await reply(writer, {"content": [{"type": "text", "text": "answer"}]})

    async with upstream(respond) as (url, requests):
        monkeypatch.setattr(provider, "MESSAGES_URL", url)
        args = (provider.BASE_URL, "claude-sonnet-4-6", messages)
        headers = {"Authorization": "Bearer cap"}
        if streaming:
            chunks = [
                chunk
                async for chunk in llm_core.stream_llm(
                    *args, headers=headers, max_tokens=0
                )
            ]
            assert chunks[-1] == "data: [DONE]\n\n"
        else:
            assert (
                await llm_core.llm_call_async(*args, headers=headers, max_tokens=0)
                == "answer"
            )

    assert len(requests) == 1
    assert json.loads(requests[0][2])["max_tokens"] == 4096
    sent = json.loads(requests[0][2])["messages"]
    assert sent[0]["role"] == "user"
    assert sent[0]["content"][0] == {"type": "text", "text": context["content"]}
    assert sent[1]["content"][0]["id"] == "a"
    assert sent[2]["content"][0]["tool_use_id"] == "a"
    assert "metadata" in context and "metadata" not in sent[0]
    assert "reasoning_content" in assistant and "reasoning_content" not in sent[1]


@pytest.mark.asyncio
async def test_llm_core_rejects_capability_before_source_access(monkeypatch):
    monkeypatch.setattr(provider, "capability_is_valid", lambda token: False)
    monkeypatch.delenv("ODYSSEUS_CLAUDE_SUBSCRIPTION_TOKEN", raising=False)
    monkeypatch.delenv("ODYSSEUS_CLAUDE_SUBSCRIPTION_CREDENTIALS_FILE", raising=False)
    chunks = [
        x
        async for x in llm_core.stream_llm(
            provider.BASE_URL, "model", [{"role": "user", "content": "hi"}]
        )
    ]
    assert len(chunks) == 1
    assert json.loads(chunks[0].split("data: ")[1])["status"] == 403
    with pytest.raises(HTTPException) as exc:
        await llm_core.llm_call_async(
            provider.BASE_URL, "model", [{"role": "user", "content": "hi"}]
        )
    assert exc.value.status_code == 403


def test_sync_llm_uses_explicit_subscription_transport(monkeypatch):
    monkeypatch.setattr(provider, "capability_is_valid", lambda token: False)
    with pytest.raises(HTTPException) as exc:
        llm_core.llm_call(
            provider.BASE_URL, "model", [{"role": "user", "content": "hi"}]
        )
    assert exc.value.status_code == 403


@pytest.mark.asyncio
async def test_public_stream_close_closes_upstream_socket(monkeypatch):
    monkeypatch.setenv("ODYSSEUS_CLAUDE_SUBSCRIPTION_TOKEN", "explicit-fake")
    monkeypatch.delenv("ODYSSEUS_CLAUDE_SUBSCRIPTION_CREDENTIALS_FILE", raising=False)
    monkeypatch.setattr(provider, "capability_is_valid", lambda token: True)
    disconnected = asyncio.Event()

    async def respond(reader, writer, requests):
        writer.write(
            b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nConnection: close\r\n\r\n"
            + frame("message_start", message={"model": "actual"})
        )
        await writer.drain()
        assert await reader.read() == b""
        disconnected.set()

    async with upstream(respond) as (url, requests):
        monkeypatch.setattr(provider, "MESSAGES_URL", url)
        stream = llm_core.stream_llm(
            provider.BASE_URL, "model", [{"role": "user", "content": "hi"}]
        )
        assert '"model_actual"' in await anext(stream)
        await stream.aclose()
        await asyncio.wait_for(disconnected.wait(), 2)


def test_internal_route_has_no_http_model_probe_or_fake_chat_url(monkeypatch):
    monkeypatch.setattr(
        llm_core, "_configured_cached_model_ids", lambda *args, **kwargs: []
    )
    monkeypatch.setattr(
        llm_core,
        "httpx_get_kimi_aware",
        lambda *args, **kwargs: pytest.fail(
            "internal provider must not make HTTP model probes"
        ),
    )
    assert llm_core.list_model_ids(provider.BASE_URL) == []
    assert llm_core._stream_target_url(provider.BASE_URL) == provider.BASE_URL
    assert llm_core._provider_label(provider.BASE_URL) == provider.DISPLAY_NAME
