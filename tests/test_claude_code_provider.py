"""Claude Code provider: Claude plan inference through the unmodified ``claude`` CLI.

Odysseus is the only agent on this route. Claude Code's own tools are off, the
host's Claude Code setup is not loaded, requests identify as Odysseus, and no API
key in the environment can be billed in place of the plan. None of these tests
need claude-agent-sdk or a Claude login.
"""

import asyncio
from types import SimpleNamespace

import pytest

from src import claude_code

DATA_URI = "data:image/png;base64,iVBORw0KGgo="
PLAN_ACCOUNT = {"apiProvider": "firstParty", "subscriptionType": "max", "email": "op@example.com"}


def _delta(kind, value):
    key = "text" if kind == "text_delta" else "thinking"
    return SimpleNamespace(event={"type": "content_block_delta", "index": 0, "delta": {"type": kind, key: value}})


def _assistant(text="", *, model="claude-haiku-5-5", error=None):
    return SimpleNamespace(content=[SimpleNamespace(text=text)] if text else [], model=model, error=error)


def _result(*, is_error=False, usage=None, result="", api_error_status=None):
    return SimpleNamespace(
        subtype="success", is_error=is_error, usage=usage or {}, result=result, api_error_status=api_error_status,
    )


class FakeClient:
    """Stands in for ClaudeSDKClient; messages carry the SDK's field names."""

    def __init__(self, messages, *, account=PLAN_ACCOUNT):
        self.messages = messages
        self.account = account
        self.sent = None
        self.connected = self.interrupted = self.disconnected = False

    async def connect(self):
        self.connected = True

    async def get_server_info(self):
        return {"account": self.account, "models": []}

    async def query(self, prompt):
        self.sent = [message async for message in prompt]

    async def receive_response(self):
        for message in self.messages:
            yield message

    async def interrupt(self):
        self.interrupted = True

    async def disconnect(self):
        self.disconnected = True


def _factory(client, seen=None):
    def make(options):
        if seen is not None:
            seen.append(options)
        return client
    return make


async def _collect(client, messages=None, **kwargs):
    events = []
    async for event in claude_code.stream_events(
        "claude-haiku-5-5",
        messages or [{"role": "system", "content": "You are Odysseus."}, {"role": "user", "content": "hi"}],
        client_factory=_factory(client),
        **kwargs,
    ):
        events.append(event)
    return events


def _texts(blocks):
    return [block["text"] for block in blocks if block["type"] == "text"]


def test_detects_only_the_claude_code_scheme():
    assert claude_code.is_claude_code_base(claude_code.BASE_URL)
    assert claude_code.is_claude_code_base("claude-code://local/")
    for url in (
        "https://api.anthropic.com",
        "https://chatgpt.com/backend-api/codex",
        "http://claude-code.local",
        "",
        None,
    ):
        assert not claude_code.is_claude_code_base(url)


def test_fold_puts_system_text_in_the_system_prompt_and_explains_turns():
    system, blocks = claude_code.fold_messages([
        {"role": "system", "content": "You are Odysseus."},
        {"role": "system", "content": "Use tools when needed."},
        {"role": "user", "content": "hi"},
    ])
    assert system.startswith("You are Odysseus.\n\nUse tools when needed.")
    assert claude_code.TURN_FORMAT_NOTE in system
    assert _texts(blocks) == ['<turn role="user">hi</turn>']


def test_fold_is_append_only_so_earlier_turns_stay_cacheable():
    messages = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
        {"role": "user", "content": "again"},
    ]
    _, blocks = claude_code.fold_messages(messages)
    assert _texts(blocks) == [
        '<turn role="user">hi</turn>',
        '<turn role="assistant">hello</turn>',
        '<turn role="user">again</turn>',
    ]
    _, longer = claude_code.fold_messages(
        messages + [{"role": "assistant", "content": "yo"}, {"role": "user", "content": "three"}]
    )
    assert longer[: len(blocks)] == blocks


def test_fold_keeps_images_inside_their_turn():
    _, blocks = claude_code.fold_messages([{"role": "user", "content": [
        {"type": "text", "text": "what is this?"},
        {"type": "image_url", "image_url": {"url": DATA_URI}},
    ]}])
    assert blocks == [
        {"type": "text", "text": '<turn role="user">what is this?'},
        {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "iVBORw0KGgo="}},
        {"type": "text", "text": "</turn>"},
    ]


def test_fold_accepts_responses_style_and_remote_images():
    _, blocks = claude_code.fold_messages([{"role": "user", "content": [
        {"type": "input_image", "image_url": DATA_URI},
        {"type": "image_url", "image_url": {"url": "https://example.com/cat.png"}},
    ]}])
    images = [block for block in blocks if block["type"] == "image"]
    assert images[0]["source"]["type"] == "base64"
    assert images[1]["source"] == {"type": "url", "url": "https://example.com/cat.png"}


def test_fold_labels_parts_it_cannot_forward_instead_of_dropping_them():
    _, blocks = claude_code.fold_messages([{"role": "user", "content": [
        {"type": "text", "text": "listen"},
        {"type": "audio", "audio": {"url": "data:audio/mpeg;base64,AAAA"}},
        {"type": "image_url", "image_url": {"url": "file:///etc/passwd"}},
    ]}])
    text = "".join(_texts(blocks))
    assert "[audio attachment omitted]" in text
    assert "[image attachment could not be forwarded]" in text
    assert "/etc/passwd" not in text


def test_fold_renders_tool_turns_and_native_tool_calls_as_text():
    _, blocks = claude_code.fold_messages([
        {"role": "user", "content": "list the repo"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "c1", "type": "function", "function": {"name": "ls", "arguments": "{\"path\": \".\"}"}},
        ]},
        {"role": "tool", "tool_call_id": "c1", "content": "README.md"},
    ])
    texts = _texts(blocks)
    assert texts[1] == '<turn role="assistant">[tool call ls: {"path": "."}]</turn>'
    assert texts[2] == '<turn role="tool">README.md</turn>'


def test_fold_without_a_conversation_turn_is_an_error():
    with pytest.raises(claude_code.ClaudeCodeError) as excinfo:
        claude_code.fold_messages([{"role": "system", "content": "only system"}])
    assert excinfo.value.status == 400


def test_options_keep_claude_code_a_pure_model():
    opts = claude_code.option_kwargs("claude-haiku-5-5", "You are Odysseus.")
    assert opts["model"] == "claude-haiku-5-5"
    assert opts["system_prompt"] == "You are Odysseus."
    assert opts["tools"] == []
    assert opts["setting_sources"] == []
    assert opts["strict_mcp_config"] is True
    assert opts["mcp_servers"] == {}
    assert opts["max_turns"] == 1
    assert opts["include_partial_messages"] is True
    assert opts["verbatim_prompts"] is True
    # Host Claude Code setup (CLAUDE.md, skills, plugins, hooks, MCP, commands)
    # stays out, user text is never a slash command, and no transcripts are kept.
    assert {"safe-mode", "disable-slash-commands", "no-session-persistence"} <= set(opts["extra_args"])


def test_options_identify_as_odysseus_and_never_bill_an_api_key():
    opts = claude_code.option_kwargs("m", "s")
    assert opts["env"]["CLAUDE_AGENT_SDK_CLIENT_APP"].startswith("odysseus/")
    assert "client_app" not in opts  # not every supported SDK release has the option
    for key in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_BASE_URL"):
        assert opts["env"][key] == ""


def test_options_run_in_a_private_empty_working_directory(tmp_path, monkeypatch):
    monkeypatch.setenv("ODYSSEUS_CLAUDE_CODE_WORKDIR", str(tmp_path / "cc"))
    opts = claude_code.option_kwargs("m", "s")
    assert opts["cwd"] == str(tmp_path / "cc")
    assert (tmp_path / "cc").is_dir()


def test_effort_is_forwarded_only_when_claude_code_accepts_it():
    assert claude_code.option_kwargs("m", "s", effort="high")["effort"] == "high"
    assert "effort" not in claude_code.option_kwargs("m", "s", effort="turbo")
    assert "effort" not in claude_code.option_kwargs("m", "s")


def test_models_come_from_the_cli_model_list_resolved_and_deduplicated():
    info = {"models": [
        {"value": "default", "resolvedModel": "claude-opus-5-5"},
        {"value": "opus", "resolvedModel": "claude-opus-5-5"},
        {"value": "sonnet", "resolvedModel": "claude-sonnet-5-5"},
        {"value": "claude-haiku-4-5-20251001", "resolvedModel": "claude-haiku-4-5-20251001"},
        {"value": "broken"},
        "junk",
    ]}
    assert claude_code.models_from_server_info(info) == [
        "claude-opus-5-5", "claude-sonnet-5-5", "claude-haiku-4-5-20251001", "broken",
    ]


# ── Streaming ───────────────────────────────────────────────────────────────


def test_stream_yields_text_thinking_model_and_plan_usage():
    client = FakeClient([
        SimpleNamespace(subtype="init", data={"tools": []}),
        SimpleNamespace(event={"type": "message_start", "message": {"model": "claude-haiku-5-5"}}),
        _delta("thinking_delta", "hmm"),
        _delta("text_delta", "Hel"),
        _delta("text_delta", "lo"),
        _assistant("Hello"),
        _result(usage={
            "input_tokens": 2, "cache_creation_input_tokens": 300, "cache_read_input_tokens": 700, "output_tokens": 5,
        }),
    ])
    events = asyncio.run(_collect(client))
    assert events == [
        ("model", "claude-haiku-5-5"),
        ("thinking", "hmm"),
        ("text", "Hel"),
        ("text", "lo"),
        ("usage", {"input_tokens": 1002, "output_tokens": 5}),
    ]
    assert client.disconnected and not client.interrupted


def test_stream_sends_one_user_message_with_the_folded_turns():
    client = FakeClient([_result()])
    seen = []

    async def run():
        async for _ in claude_code.stream_events(
            "claude-haiku-5-5",
            [{"role": "system", "content": "You are Odysseus."}, {"role": "user", "content": "hi"}],
            client_factory=_factory(client, seen),
        ):
            pass

    asyncio.run(run())
    assert seen[0]["system_prompt"].startswith("You are Odysseus.")
    assert seen[0]["tools"] == []
    assert len(client.sent) == 1
    assert client.sent[0]["type"] == "user"
    assert client.sent[0]["message"] == {
        "role": "user", "content": [{"type": "text", "text": '<turn role="user">hi</turn>'}],
    }


def test_stream_falls_back_to_the_final_message_when_no_deltas_arrive():
    events = asyncio.run(_collect(FakeClient([_assistant("whole reply"), _result()])))
    assert ("text", "whole reply") in events


@pytest.mark.parametrize("account", [
    {},
    {"apiProvider": "firstParty"},
    {"apiProvider": "bedrock", "subscriptionType": "max"},
])
def test_nothing_is_sent_unless_the_cli_is_signed_in_to_a_claude_plan(account):
    client = FakeClient([_delta("text_delta", "should not happen")], account=account)
    with pytest.raises(claude_code.ClaudeCodeError) as excinfo:
        asyncio.run(_collect(client))
    assert excinfo.value.status == 401
    assert "claude auth login" in excinfo.value.message
    assert client.sent is None
    assert client.disconnected


def test_authentication_failure_asks_for_claude_auth_login():
    client = FakeClient([
        _assistant("Not logged in · Please run /login", model="<synthetic>", error="authentication_failed"),
        _result(is_error=True, result="Not logged in · Please run /login"),
    ])
    with pytest.raises(claude_code.ClaudeCodeError) as excinfo:
        asyncio.run(_collect(client))
    assert excinfo.value.status == 401
    assert "claude auth login" in excinfo.value.message


@pytest.mark.parametrize("messages", [
    [_assistant(error="rate_limit"), _result(is_error=True)],
    [_result(is_error=True, api_error_status=429, result="rate limited")],
])
def test_plan_limits_surface_as_429(messages):
    with pytest.raises(claude_code.ClaudeCodeError) as excinfo:
        asyncio.run(_collect(FakeClient(messages)))
    assert excinfo.value.status == 429


def test_other_result_errors_report_the_cli_message():
    with pytest.raises(claude_code.ClaudeCodeError) as excinfo:
        asyncio.run(_collect(FakeClient([_result(is_error=True, result="boom")])))
    assert excinfo.value.status == 502
    assert "boom" in excinfo.value.message


def test_closing_the_stream_early_interrupts_the_cli():
    client = FakeClient([_delta("text_delta", str(n)) for n in range(50)])

    async def run():
        stream = claude_code.stream_events(
            "claude-haiku-5-5", [{"role": "user", "content": "count"}], client_factory=_factory(client),
        )
        assert await stream.__anext__() == ("text", "0")
        await stream.aclose()

    asyncio.run(run())
    assert client.interrupted
    assert client.disconnected


def test_missing_sdk_is_a_clear_server_error(monkeypatch):
    def missing(_options):
        raise claude_code.ClaudeCodeError(503, claude_code.NOT_INSTALLED, fallback_eligible=False)

    async def run():
        async for _ in claude_code.stream_events("m", [{"role": "user", "content": "hi"}], client_factory=missing):
            pass

    with pytest.raises(claude_code.ClaudeCodeError) as excinfo:
        asyncio.run(run())
    assert excinfo.value.status == 503
    assert "claude-agent-sdk" in excinfo.value.message


def test_sdk_message_fields_match_what_the_provider_reads():
    sdk = pytest.importorskip("claude_agent_sdk")
    fields = {
        sdk.StreamEvent: {"event"},
        sdk.AssistantMessage: {"content", "model", "error"},
        sdk.ResultMessage: {"is_error", "usage", "result", "api_error_status"},
        sdk.TextBlock: {"text"},
    }
    for cls, names in fields.items():
        assert names <= set(cls.__dataclass_fields__), cls.__name__
    options = sdk.ClaudeAgentOptions(**claude_code.option_kwargs("claude-haiku-5-5", "s", effort="low"))
    assert options.tools == [] and options.verbatim_prompts is True
