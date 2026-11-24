"""Direct OAuth wire contracts; all credentials are synthetic fixtures."""

import copy

import pytest


def tool(name):
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": "Odysseus memory",
            "parameters": {"type": "object"},
        },
    }


def test_native_history_names_and_branding_survive_round_trip():
    from src.claude_subscription import build_payload

    messages = [
        {"role": "system", "content": "You are Odysseus. memory session_search"},
        {"role": "developer", "content": "Keep domain tools."},
        {"role": "user", "content": "look"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "a",
                    "type": "function",
                    "function": {"name": "memory", "arguments": '{"q":"x"}'},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "a", "content": "found"},
    ]
    original = copy.deepcopy(messages)
    payload, names = build_payload(
        "claude-sonnet-4-6", messages, tools=[tool("memory")]
    )
    assert payload["tools"][0]["name"] == "mcp__memory"
    assert payload["tools"][0]["description"] == "Odysseus memory"
    assert payload["messages"][1]["content"][0] == {
        "type": "tool_use",
        "id": "a",
        "name": "mcp__memory",
        "input": {"q": "x"},
    }
    assert payload["messages"][2]["content"][0]["tool_use_id"] == "a"
    assert (
        payload["system"][0]["text"]
        == "You are Claude Code, Anthropic's official CLI for Claude."
    )
    assert "Odysseus" in payload["system"][1]["text"]
    assert names == {"mcp__memory": "memory"}
    assert messages == original


@pytest.mark.parametrize(
    "name,wire",
    [
        ("read_file", "mcp__read_file"),
        ("mcp__read_file", "mcp__read_file"),
        ("mcp_server_tool", "mcp__server_tool"),
        ("a" * 59, "mcp__" + "a" * 59),
        ("mcp__mcp__x", "mcp__mcp__x"),
    ],
)
def test_tool_name_encoding(name, wire):
    from src.claude_subscription import build_payload

    payload, names = build_payload(
        "claude-sonnet-4-6", [{"role": "user", "content": "hi"}], tools=[tool(name)]
    )
    assert payload["tools"][0]["name"] == wire
    assert names[wire] == name


@pytest.mark.parametrize(
    "tools,messages",
    [
        ([tool("foo"), tool("mcp__foo")], [{"role": "user", "content": "hi"}]),
        ([tool("mcp_foo"), tool("foo")], [{"role": "user", "content": "hi"}]),
        ([tool("x" * 60)], [{"role": "user", "content": "hi"}]),
        ([tool("bad.name")], [{"role": "user", "content": "hi"}]),
        ([tool("foo")], [{"role": "alien", "content": "hi"}]),
        ([{"type": "web_search"}], [{"role": "user", "content": "hi"}]),
        (
            [tool("foo")],
            [{"role": "user", "content": [{"type": "audio", "data": "x"}]}],
        ),
        (
            [tool("foo")],
            [
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": "x",
                            "function": {"name": "foo", "arguments": "not json"},
                        }
                    ],
                }
            ],
        ),
    ],
)
def test_invalid_or_lossy_requests_fail_explicitly(tools, messages):
    from src.claude_subscription import SubscriptionError, build_payload

    with pytest.raises(SubscriptionError) as error:
        build_payload("claude-sonnet-4-6", messages, tools=tools)
    assert error.value.status == 400
    assert "not json" not in str(error.value)


def test_native_blocks_and_parallel_results_replay():
    from src.claude_subscription import build_payload

    messages = [
        {"role": "user", "content": "go"},
        {
            "role": "assistant",
            "content": [
                {"type": "tool_use", "id": "1", "name": "foo", "input": {}},
                {"type": "tool_use", "id": "2", "name": "bar", "input": {}},
            ],
        },
        {"role": "tool", "tool_call_id": "1", "content": "first"},
        {"role": "tool", "tool_call_id": "2", "content": "second"},
    ]
    payload, names = build_payload(
        "claude-sonnet-4-6",
        messages,
        tools=[tool("foo"), tool("bar")],
        tool_choice_none=True,
    )
    assert [b["name"] for b in payload["messages"][1]["content"]] == [
        "mcp__foo",
        "mcp__bar",
    ]
    assert [b["tool_use_id"] for b in payload["messages"][2]["content"]] == ["1", "2"]
    assert payload["tool_choice"] == {"type": "none"}


@pytest.mark.parametrize(
    "model,mode,effort,thinking",
    [
        ("claude-sonnet-4-6", "off", None, {"type": "disabled"}),
        (
            "claude-sonnet-4-6",
            "on",
            "high",
            {"type": "adaptive", "display": "summarized"},
        ),
        ("claude-sonnet-4-5", "on", "low", {"type": "enabled", "budget_tokens": 4000}),
        ("claude-sonnet-5-5", "off", None, {"type": "between_tools"}),
    ],
)
def test_thinking_settings_are_forwarded_not_ignored(model, mode, effort, thinking):
    from src.claude_subscription import build_payload

    payload, _ = build_payload(
        model,
        [{"role": "user", "content": "hi"}],
        thinking_mode=mode,
        reasoning_effort=effort,
        max_tokens=16384,
    )
    assert payload["thinking"] == thinking
    if thinking["type"] == "adaptive":
        assert payload["output_config"] == {"effort": effort}


@pytest.mark.parametrize(
    "model", ["claude-fable-5-2", "claude-opus-5-5", "claude-mythos-1"]
)
def test_thinking_off_on_mandatory_thinking_models_sends_no_thinking(model):
    from src.claude_subscription import build_payload

    payload, _ = build_payload(
        model, [{"role": "user", "content": "hi"}], thinking_mode="off"
    )
    assert "thinking" not in payload


@pytest.mark.parametrize(
    "tool_calls", [[], [{"id": "a", "function": {"name": "foo", "arguments": "{}"}}]]
)
def test_assistant_thinking_and_text_remain_native_with_tool_calls(tool_calls):
    from src.claude_subscription import build_payload

    content = [
        {"type": "thinking", "thinking": "analysis", "signature": "fixture-signature"},
        {"type": "text", "text": "checking"},
    ]
    followup = (
        {"role": "tool", "tool_call_id": "a", "content": "found"}
        if tool_calls
        else {"role": "user", "content": "continue"}
    )
    payload, _ = build_payload(
        "model",
        [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": content, "tool_calls": tool_calls},
            followup,
        ],
    )
    assert payload["messages"][1]["content"][:2] == content
    assert len(payload["messages"][1]["content"]) == (3 if tool_calls else 2)


@pytest.mark.parametrize("native_result", [True, False])
def test_tool_result_images_use_native_anthropic_sources(native_result):
    from src.claude_subscription import build_payload

    image = {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA=="}}
    caption = {"type": "text", "text": "found"}
    result = (
        {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "a", "content": [image, caption]}
            ],
        }
        if native_result
        else {"role": "tool", "tool_call_id": "a", "content": [image, caption]}
    )
    payload, _ = build_payload(
        "model",
        [
            {"role": "user", "content": "hi"},
            {
                "role": "assistant",
                "tool_calls": [
                    {"id": "a", "function": {"name": "foo", "arguments": "{}"}}
                ],
            },
            result,
        ],
    )
    assert payload["messages"][2]["content"] == [
        {
            "type": "tool_result",
            "tool_use_id": "a",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": "AA==",
                    },
                },
                caption,
            ],
        }
    ]
    assert image["type"] == "image_url"
