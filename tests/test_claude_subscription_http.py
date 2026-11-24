"""Real TCP/HTTP integration: deterministic fake upstream, no real credentials."""

import asyncio
import json
from contextlib import asynccontextmanager

import pytest

from tests.test_claude_subscription_direct import tool


@asynccontextmanager
async def upstream(responder):
    requests = []
    tasks = set()

    async def handle(reader, writer):
        tasks.add(asyncio.current_task())
        try:
            head = await reader.readuntil(b"\r\n\r\n")
            lines = head.decode().split("\r\n")
            headers = {
                key.lower(): value
                for key, value in (
                    line.split(": ", 1) for line in lines[1:] if ": " in line
                )
            }
            body = await reader.readexactly(int(headers.get("content-length", 0)))
            requests.append((lines[0], headers, body))
            await responder(reader, writer, requests)
        finally:
            writer.close()
            await writer.wait_closed()
            tasks.discard(asyncio.current_task())

    server = await asyncio.start_server(handle, "127.0.0.1", 0)
    try:
        yield f"http://127.0.0.1:{server.sockets[0].getsockname()[1]}", requests
    finally:
        server.close()
        await server.wait_closed()
        for task in list(tasks):
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


async def reply(writer, body, status=200, content_type="application/json"):
    raw = body if isinstance(body, bytes) else json.dumps(body).encode()
    writer.write(
        f"HTTP/1.1 {status} Response\r\nContent-Type: {content_type}\r\nContent-Length: {len(raw)}\r\nConnection: close\r\n\r\n".encode()
        + raw
    )
    await writer.drain()


def frame(kind, **data):
    return ("data:" + json.dumps({"type": kind, **data}) + "\n\n").encode()


@pytest.mark.asyncio
async def test_nonstream_http_headers_native_tools_and_decoding():
    from src.claude_subscription import Transport
    from src.claude_subscription_credentials import StaticToken

    async def respond(reader, writer, requests):
        await reply(
            writer,
            {
                "model": "claude-sonnet-4-6",
                "content": [
                    {"type": "text", "text": "ok"},
                    {
                        "type": "tool_use",
                        "id": "a",
                        "name": "mcp__lookup",
                        "input": {"q": "x"},
                    },
                ],
                "usage": {"input_tokens": 2, "output_tokens": 3},
            },
        )

    async with upstream(respond) as (url, requests):
        result = await Transport(
            StaticToken("fixture-access-token"), messages_url=url + "/v1/messages"
        ).complete(
            "claude-sonnet-4-6",
            [{"role": "user", "content": "hi"}],
            tools=[tool("lookup")],
        )
    assert result["content"][1]["name"] == "lookup"
    assert result["content"][1]["input"] == {"q": "x"}
    _, headers, body = requests[0]
    assert headers["authorization"] == "Bearer fixture-access-token"
    assert "x-api-key" not in headers
    assert headers["anthropic-version"] == "2023-06-01"
    assert headers["x-app"] == "cli"
    assert headers["user-agent"].startswith("claude-code/")
    assert set(headers["anthropic-beta"].split(",")) == {
        "interleaved-thinking-2025-05-14",
        "fine-grained-tool-streaming-2025-05-14",
        "claude-code-20250219",
        "oauth-2025-04-20",
    }
    assert json.loads(body)["tools"][0]["name"] == "mcp__lookup"
    assert len(requests) == 1


@pytest.mark.asyncio
async def test_stream_parallel_calls_split_json_unicode_and_usage():
    from src.claude_subscription import Transport
    from src.claude_subscription_credentials import StaticToken

    data = frame(
        "message_start",
        message={"model": "claude-sonnet-4-6", "usage": {"input_tokens": 4}},
    )
    data += frame(
        "content_block_start", index=0, content_block={"type": "text", "text": ""}
    )
    data += frame(
        "content_block_delta", index=0, delta={"type": "text_delta", "text": "héllo"}
    )
    for i, name in [(1, "mcp__lookup"), (2, "mcp__mcp__literal")]:
        data += frame(
            "content_block_start",
            index=i,
            content_block={"type": "tool_use", "id": str(i), "name": name, "input": {}},
        )
        for fragment in ['{"q":', '"yes"}']:
            data += frame(
                "content_block_delta",
                index=i,
                delta={"type": "input_json_delta", "partial_json": fragment},
            )
        data += frame("content_block_stop", index=i)
    data += frame("content_block_stop", index=0)
    data += frame("message_delta", usage={"output_tokens": 9}) + frame("message_stop")

    async def respond(reader, writer, requests):
        writer.write(
            f"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: {len(data)}\r\nConnection: close\r\n\r\n".encode()
        )
        # Deliberately split every field and JSON token across TCP writes.
        for offset in range(0, len(data), 7):
            writer.write(data[offset : offset + 7])
            await writer.drain()
            await asyncio.sleep(0)

    async with upstream(respond) as (url, requests):
        events = [
            e
            async for e in Transport(StaticToken("fake"), messages_url=url).stream(
                "claude-sonnet-4-6",
                [{"role": "user", "content": "hi"}],
                tools=[tool("lookup"), tool("mcp__mcp__literal")],
            )
        ]
    assert {"delta": "héllo"} in events
    calls = next(e["calls"] for e in events if e.get("type") == "tool_calls")
    assert calls == [
        {"id": "1", "name": "lookup", "arguments": '{"q":"yes"}'},
        {"id": "2", "name": "mcp__mcp__literal", "arguments": '{"q":"yes"}'},
    ]
    assert events[-1] == {
        "type": "usage",
        "data": {"input_tokens": 4, "output_tokens": 9},
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("wire", ["mcp__unknown", "lookup", "mcp__mcp__lookup"])
async def test_unknown_returned_names_never_dispatch(streaming, wire):
    from src.claude_subscription import SubscriptionError, Transport
    from src.claude_subscription_credentials import StaticToken

    async def respond(reader, writer, requests):
        block = {"type": "tool_use", "id": "a", "name": wire, "input": {}}
        if streaming:
            data = frame("message_start", message={}) + frame(
                "content_block_start", index=0, content_block=block
            )
            data += frame("content_block_stop", index=0) + frame("message_stop")
            await reply(writer, data, content_type="text/event-stream")
        else:
            await reply(writer, {"content": [block]})

    async with upstream(respond) as (url, requests):
        transport = Transport(StaticToken("fake"), messages_url=url)
        with pytest.raises(SubscriptionError) as error:
            if streaming:
                [
                    e
                    async for e in transport.stream(
                        "model",
                        [{"role": "user", "content": "hi"}],
                        tools=[tool("lookup")],
                    )
                ]
            else:
                await transport.complete(
                    "model", [{"role": "user", "content": "hi"}], tools=[tool("lookup")]
                )
        assert error.value.status == 502
    assert len(requests) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [400, 401, 403, 429, 500, 503])
async def test_upstream_errors_redacted_and_not_retried(status):
    from src.claude_subscription import SubscriptionError, Transport
    from src.claude_subscription_credentials import StaticToken

    async def respond(reader, writer, requests):
        await reply(writer, {"error": "sensitive-fake request contents"}, status=status)

    async with upstream(respond) as (url, requests):
        with pytest.raises(SubscriptionError) as error:
            await Transport(StaticToken("sensitive-fake"), messages_url=url).complete(
                "model", [{"role": "user", "content": "hi"}]
            )
    assert "sensitive-fake" not in str(error.value)
    assert error.value.status == status
    assert len(requests) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["timeout", "cancel", "close"])
async def test_transport_timeout_cancel_and_close_release_real_socket(operation):
    from src.claude_subscription import SubscriptionError, Transport
    from src.claude_subscription_credentials import StaticToken

    received = asyncio.Event()
    disconnected = asyncio.Event()

    async def respond(reader, writer, requests):
        writer.write(
            b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nConnection: close\r\n\r\n"
            + frame("message_start", message={"model": "model"})
        )
        await writer.drain()
        received.set()
        assert await reader.read() == b""
        disconnected.set()

    async with upstream(respond) as (url, requests):
        stream = Transport(
            StaticToken("fake"),
            messages_url=url,
            timeout=0.05 if operation == "timeout" else 2,
        ).stream("model", [{"role": "user", "content": "hi"}])
        assert (await anext(stream))["type"] == "model_actual"
        await received.wait()
        if operation == "close":
            await stream.aclose()
        elif operation == "cancel":
            task = asyncio.create_task(anext(stream))
            await asyncio.sleep(0)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            await stream.aclose()
        else:
            with pytest.raises(SubscriptionError) as error:
                await anext(stream)
            assert error.value.status == 504
        await asyncio.wait_for(disconnected.wait(), 2)


@pytest.mark.asyncio
async def test_truncated_stream_does_not_report_success():
    from src.claude_subscription import SubscriptionError, Transport
    from src.claude_subscription_credentials import StaticToken

    async def respond(reader, writer, requests):
        await reply(
            writer, frame("message_start", message={}), content_type="text/event-stream"
        )

    async with upstream(respond) as (url, requests):
        with pytest.raises(SubscriptionError, match="before message_stop"):
            [
                e
                async for e in Transport(StaticToken("fake"), messages_url=url).stream(
                    "model", [{"role": "user", "content": "hi"}]
                )
            ]


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("second_status", [200, 401])
async def test_401_refresh_and_inference_retry_are_bounded(
    tmp_path, streaming, second_status
):
    import time

    from src.claude_subscription import SubscriptionError, Transport
    from src.claude_subscription_credentials import FileCredentials
    from tests.test_claude_subscription_credentials import credential_file

    path = credential_file(tmp_path, expiresAt=int(time.time() * 1000 + 3600000))
    inference = []

    async def respond(reader, writer, requests):
        request = requests[-1]
        if request[0].startswith("POST /token "):
            await reply(
                writer,
                {
                    "access_token": "rotated-fake",
                    "refresh_token": "next-fake",
                    "expires_in": 3600,
                },
            )
            return
        inference.append(request)
        if len(inference) == 1 or second_status == 401:
            await reply(writer, {"error": "old-fake"}, status=401)
        elif streaming:
            await reply(
                writer,
                frame("message_start", message={}) + frame("message_stop"),
                content_type="text/event-stream",
            )
        else:
            await reply(writer, {"content": [{"type": "text", "text": "ok"}]})

    async with upstream(respond) as (url, requests):
        transport = Transport(
            FileCredentials(path, token_url=url + "/token"),
            messages_url=url + "/messages",
        )

        async def run():
            if streaming:
                return [
                    e
                    async for e in transport.stream(
                        "model", [{"role": "user", "content": "hi"}]
                    )
                ]
            return await transport.complete(
                "model", [{"role": "user", "content": "hi"}]
            )

        if second_status == 401:
            with pytest.raises(SubscriptionError) as error:
                await run()
            assert error.value.status == 401
        else:
            assert await run()
    assert len(inference) == 2
    assert len(requests) == 3
    assert [r[1]["authorization"] for r in inference] == [
        "Bearer old-fake",
        "Bearer rotated-fake",
    ]


@pytest.mark.asyncio
async def test_injected_source_failure_is_secret_safe():
    from src.claude_subscription import SubscriptionError, Transport

    class FailingSource:
        async def token(self, *args, **kwargs):
            raise RuntimeError("fixture-secret-token")

    with pytest.raises(SubscriptionError) as exc:
        await Transport(FailingSource()).complete(
            "model", [{"role": "user", "content": "hi"}]
        )
    assert "fixture-secret-token" not in str(exc.value)


@pytest.mark.asyncio
async def test_timeout_bounds_credential_resolution_before_inference():
    from src.claude_subscription import SubscriptionError, Transport

    cancelled = asyncio.Event()

    class SlowSource:
        async def token(self, *args, **kwargs):
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()

    with pytest.raises(SubscriptionError) as exc:
        await asyncio.wait_for(
            Transport(SlowSource(), timeout=0.03).complete(
                "model", [{"role": "user", "content": "hi"}]
            ),
            0.5,
        )
    assert exc.value.status == 504
    assert cancelled.is_set()
