"""Local reference: direct Messages OAuth, not the Claude Agent SDK.

Only the explicit claude-subscription://local route uses this transport. See
artifacts/claude-subscription-direct-pr.md for compatibility identity disclosure.
"""

from __future__ import annotations

import asyncio
import copy
import hmac
import json
import re
import secrets
from functools import lru_cache
from typing import Any, Dict, Optional
from urllib.parse import urlparse

PROVIDER = "claude-subscription"
BASE_URL = "claude-subscription://local"
DISPLAY_NAME = "Claude Subscription (direct OAuth reference)"
AUTH_MODE = "claude-subscription-oauth"
IDENTITY = "You are Claude Code, Anthropic's official CLI for Claude."


class SubscriptionError(RuntimeError):
    def __init__(self, status, message, *, fallback_eligible=False):
        super().__init__(message)
        self.status = status
        self.message = message
        self.fallback_eligible = fallback_eligible


def _invalid():
    return SubscriptionError(400, "Unsupported or invalid Claude subscription request.")


def _wire(name):
    if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", name):
        raise _invalid()
    wire = name if name.startswith("mcp__") else "mcp__" + name.removeprefix("mcp_")
    if len(wire) > 64:
        raise _invalid()
    return wire


def build_payload(
    model,
    messages,
    *,
    tools=None,
    temperature=None,
    max_tokens=4096,
    stream=False,
    tool_choice_none=False,
    thinking_mode=None,
    reasoning_effort=None,
):
    """Build native role-based history without mutating caller-owned objects.

    Names are bijective within this request. Ambiguous/overlong names fail rather
    than silently rename a domain tool. Only current tools may be dispatched.
    """
    from src.llm_core import _build_anthropic_payload

    # The chat and agent routes use zero to mean no user output-limit override.
    # Messages requires a positive limit even when Odysseus leaves it unset.
    if (
        isinstance(max_tokens, int)
        and not isinstance(max_tokens, bool)
        and max_tokens == 0
    ):
        max_tokens = 4096
    messages = copy.deepcopy(messages)
    tools = copy.deepcopy(tools or [])
    # Keep the guarded message content, but don't send local provenance or
    # convert unsigned assistant reasoning into native thinking blocks.
    for message in messages:
        if isinstance(message, dict):
            message.pop("metadata", None)
            message.pop("reasoning_content", None)
    _validate_request(model, messages, tools, max_tokens)
    for message in messages:
        if message.get("role") == "developer":
            message["role"] = "system"
        if message.get("tool_calls"):
            from src.llm_core import _as_content_blocks

            blocks = _as_content_blocks(message.get("content"))
            for call in message.pop("tool_calls"):
                function = call["function"]
                args = function.get("arguments", "{}")
                blocks.append(
                    {
                        "type": "tool_use",
                        "id": call["id"],
                        "name": function["name"],
                        "input": json.loads(args) if isinstance(args, str) else args,
                    }
                )
            message["content"] = blocks
    payload = _build_anthropic_payload(
        model, messages, temperature, max_tokens, stream=stream, tools=tools
    )
    payload["system"] = [{"type": "text", "text": IDENTITY}] + payload.get("system", [])
    names = {}
    for t in payload.get("tools", []):
        original = t["name"]
        wire = _wire(original)
        if wire in names:
            raise _invalid()
        names[wire] = original
        t["name"] = wire
    claimed = dict(names)
    for message in payload["messages"]:
        for block in (
            message.get("content", [])
            if isinstance(message.get("content"), list)
            else []
        ):
            if block.get("type") == "tool_use":
                name = block["name"]
                wire = _wire(name)
                if wire in claimed and claimed[wire] != name:
                    raise _invalid()
                claimed[wire] = name
                block["name"] = wire
    # Tool results for parallel calls belong in one user turn.
    merged = []
    for message in payload["messages"]:
        if merged and merged[-1]["role"] == message["role"]:
            from src.llm_core import _as_content_blocks

            merged[-1]["content"] = _as_content_blocks(
                merged[-1]["content"]
            ) + _as_content_blocks(message["content"])
        else:
            merged.append(message)
    payload["messages"] = merged
    if tool_choice_none:
        payload["tool_choice"] = {"type": "none"}
    if payload.get("temperature") is None:
        payload.pop("temperature", None)
    _apply_thinking(payload, model, thinking_mode, reasoning_effort)
    return payload, names


def _validate_blocks(content):
    if content is None or isinstance(content, str):
        return
    if not isinstance(content, list):
        raise _invalid()
    allowed = {
        "text",
        "image",
        "image_url",
        "tool_use",
        "tool_result",
        "thinking",
        "redacted_thinking",
    }
    for block in content:
        if not isinstance(block, dict) or block.get("type") not in allowed:
            raise _invalid()
        if block["type"] == "tool_use" and not isinstance(block.get("input"), dict):
            raise _invalid()


def _validate_request(model, messages, tools, max_tokens):
    if not isinstance(model, str) or not model or not messages:
        raise _invalid()
    if (
        isinstance(max_tokens, bool)
        or not isinstance(max_tokens, int)
        or max_tokens <= 0
    ):
        raise _invalid()
    for tool in tools:
        if (
            not isinstance(tool, dict)
            or tool.get("type") != "function"
            or set(tool) - {"type", "function"}
        ):
            raise _invalid()
        function = tool.get("function")
        if not isinstance(function, dict) or set(function) - {
            "name",
            "description",
            "parameters",
        }:
            raise _invalid()
        _wire(function.get("name"))
    for message in messages:
        if not isinstance(message, dict) or message.get("role") not in {
            "system",
            "developer",
            "user",
            "assistant",
            "tool",
        }:
            raise _invalid()
        if set(message) - {"role", "content", "tool_calls", "tool_call_id", "name"}:
            raise _invalid()
        _validate_blocks(message.get("content"))
        if message["role"] in {"system", "developer"} and not isinstance(
            message.get("content"), str
        ):
            raise _invalid()
        for call in message.get("tool_calls") or []:
            try:
                args = call["function"].get("arguments", "{}")
                if not isinstance(
                    json.loads(args) if isinstance(args, str) else args, dict
                ):
                    raise _invalid()
                if not call.get("id"):
                    raise _invalid()
            except (KeyError, TypeError, ValueError):
                raise _invalid() from None


MESSAGES_URL = "https://api.anthropic.com/v1/messages"
MODELS_URL = "https://api.anthropic.com/v1/models"
BETAS = "interleaved-thinking-2025-05-14,fine-grained-tool-streaming-2025-05-14,claude-code-20250219,oauth-2025-04-20"


@lru_cache(maxsize=1)
def _installed_cli_version():
    """Read the installed version once, including services with a minimal PATH."""
    import shutil
    import subprocess
    from pathlib import Path

    command = shutil.which("claude")
    if not command:
        candidate = Path.home() / ".local" / "bin" / "claude"
        if candidate.is_file():
            command = str(candidate)
    if command:
        try:
            result = subprocess.run(
                [command, "--version"],
                stdin=subprocess.DEVNULL,
                capture_output=True,
                text=True,
                timeout=5,
                check=True,
            )
            match = re.match(r"([0-9]+\.[0-9]+\.[0-9]+)(?:\s|$)", result.stdout.strip())
            if match:
                return match.group(1)
        except (OSError, subprocess.SubprocessError):
            pass
    raise SubscriptionError(
        503,
        "Cannot detect the Claude Code compatibility version. Set "
        "ODYSSEUS_CLAUDE_SUBSCRIPTION_CLI_VERSION to the installed CLI version.",
    )


def request_headers(token):
    import os

    version = (
        os.getenv("ODYSSEUS_CLAUDE_SUBSCRIPTION_CLI_VERSION")
        or _installed_cli_version()
    )
    if not re.fullmatch(r"[0-9]+\.[0-9]+\.[0-9]+", version):
        raise SubscriptionError(400, "Invalid Claude compatibility version setting.")
    return {
        "Authorization": "Bearer " + token,
        "Content-Type": "application/json",
        "anthropic-version": "2023-06-01",
        "anthropic-beta": BETAS,
        "user-agent": f"claude-code/{version} (external, cli)",
        "x-app": "cli",
    }


def _decode_tool(block, names):
    wire = block.get("name")
    if (
        wire not in names
        or not isinstance(block.get("input"), dict)
        or not block.get("id")
    ):
        raise SubscriptionError(502, "Claude returned an unknown or invalid tool call.")
    block["name"] = names[wire]
    return block


class Transport:
    """One cancellable HTTP client per request; no implicit retries or proxy env.

    Only a 401 before any output can refresh/retry, once. Other failures never
    replay inference (including a dropped stream after partial output).
    """

    def __init__(self, source, *, messages_url=None, timeout=300):
        self.source = source
        self.messages_url = messages_url or MESSAGES_URL
        self.timeout = timeout

    def _client(self):
        import httpx

        return httpx.AsyncClient(
            timeout=httpx.Timeout(self.timeout, connect=min(10, self.timeout)),
            follow_redirects=False,
            trust_env=False,
        )

    async def _token(self, client, *, failed_token=None):
        try:
            async with asyncio.timeout(self.timeout):
                return await self.source.token(client, failed_token=failed_token)
        except TimeoutError:
            raise SubscriptionError(
                504, "Claude subscription credential resolution timed out."
            ) from None

    async def list_models(self, *, models_url=None):
        """Discover the account's live catalog without spending inference tokens."""
        import httpx

        try:
            async with asyncio.timeout(self.timeout), self._client() as client:
                token = await self._token(client)
                models = []
                cursors = set()
                params: dict[str, str | int] = {"limit": 1000}
                for _page in range(20):
                    response: httpx.Response
                    for attempt in range(2):
                        response = await client.get(
                            models_url or MODELS_URL,
                            params=params,
                            headers=request_headers(token),
                        )
                        if response.status_code == 401 and attempt == 0:
                            token = await self._token(client, failed_token=token)
                            continue
                        break
                    if response.status_code != 200:
                        raise SubscriptionError(
                            response.status_code
                            if 400 <= response.status_code <= 599
                            else 502,
                            "Claude subscription model discovery failed.",
                        )
                    data = response.json()
                    rows = data.get("data")
                    if not isinstance(rows, list) or any(
                        not isinstance(row, dict)
                        or not isinstance(row.get("id"), str)
                        or not row["id"]
                        for row in rows
                    ):
                        raise ValueError
                    models.extend(row["id"] for row in rows)
                    if not data.get("has_more"):
                        if not models:
                            raise ValueError
                        return list(dict.fromkeys(models))
                    cursor = data.get("last_id")
                    if not isinstance(cursor, str) or not cursor or cursor in cursors:
                        raise ValueError
                    cursors.add(cursor)
                    params["after_id"] = cursor
                raise ValueError
        except SubscriptionError:
            raise
        except (TimeoutError, httpx.TimeoutException):
            raise SubscriptionError(
                504, "Claude subscription model discovery timed out."
            ) from None
        except Exception:
            raise SubscriptionError(
                502, "Claude subscription returned an invalid model catalog."
            ) from None

    async def complete(self, model, messages, **kwargs):
        import httpx

        payload, names = build_payload(model, messages, **kwargs)
        try:
            async with self._client() as client:
                token = await self._token(client)
                for attempt in range(2):
                    response = await client.post(
                        self.messages_url, json=payload, headers=request_headers(token)
                    )
                    if response.status_code == 401 and attempt == 0:
                        token = await self._token(client, failed_token=token)
                        continue
                    if response.status_code != 200:
                        raise SubscriptionError(
                            response.status_code
                            if 400 <= response.status_code <= 599
                            else 502,
                            "Claude subscription upstream rejected the request.",
                        )
                    data = response.json()
                    if not isinstance(data.get("content"), list):
                        raise ValueError
                    for block in data["content"]:
                        if block.get("type") == "tool_use":
                            _decode_tool(block, names)
                    return data
        except SubscriptionError:
            raise
        except httpx.TimeoutException:
            raise SubscriptionError(
                504, "Claude subscription request timed out."
            ) from None
        except Exception:
            raise SubscriptionError(
                502, "Claude subscription response was invalid or interrupted."
            ) from None

    async def stream(self, model, messages, **kwargs):
        import httpx

        payload, names = build_payload(model, messages, stream=True, **kwargs)
        try:
            async with self._client() as client:
                token = await self._token(client)
                for attempt in range(2):
                    async with client.stream(
                        "POST",
                        self.messages_url,
                        json=payload,
                        headers=request_headers(token),
                    ) as response:
                        if response.status_code == 401 and attempt == 0:
                            # Close the rejected response before refreshing.
                            pass
                        elif response.status_code != 200:
                            raise SubscriptionError(
                                response.status_code
                                if 400 <= response.status_code <= 599
                                else 502,
                                "Claude subscription upstream rejected the request.",
                            )
                        else:
                            async for event in _stream_events(response, names):
                                yield event
                            return
                    token = await self._token(client, failed_token=token)
        except SubscriptionError:
            raise
        except httpx.TimeoutException:
            raise SubscriptionError(
                504, "Claude subscription request timed out."
            ) from None
        except Exception:
            raise SubscriptionError(
                502, "Claude subscription response was invalid or interrupted."
            ) from None


async def _sse(response):
    lines = []
    async for line in response.aiter_lines():
        if not line:
            if lines:
                yield json.loads("\n".join(lines))
                lines = []
        elif line.startswith("data:"):
            lines.append(line[5:].lstrip(" "))
    if lines:
        raise SubscriptionError(
            502, "Claude subscription stream ended inside an event."
        )


async def _stream_events(response, names):
    blocks = {}
    closed = set()
    usage = {"input_tokens": 0, "output_tokens": 0}
    started = False
    async for event in _sse(response):
        kind = event.get("type")
        if kind == "error":
            raise SubscriptionError(
                502, "Claude subscription upstream reported a stream error."
            )
        if kind == "message_start":
            if started:
                raise ValueError
            started = True
            message = event["message"]
            usage.update(
                {k: v for k, v in (message.get("usage") or {}).items() if k in usage}
            )
            if message.get("model"):
                yield {"type": "model_actual", "model": message["model"]}
        elif kind == "content_block_start":
            index = event["index"]
            if index in blocks or not started:
                raise ValueError
            block = event["content_block"]
            if block["type"] == "tool_use":
                _decode_tool(block, names)
                block["arguments"] = ""
            blocks[index] = block
            if block["type"] == "text" and block.get("text"):
                yield {"delta": block["text"]}
        elif kind == "content_block_delta":
            index = event["index"]
            if index in closed:
                raise ValueError
            block = blocks[index]
            delta = event["delta"]
            dtype = delta["type"]
            if dtype == "text_delta" and block["type"] == "text":
                yield {"delta": delta["text"]}
            elif dtype == "thinking_delta":
                yield {"delta": delta["thinking"], "thinking": True}
            elif dtype == "input_json_delta" and block["type"] == "tool_use":
                block["arguments"] += delta["partial_json"]
                if block["name"] in {
                    "create_document",
                    "update_document",
                    "edit_document",
                }:
                    yield {
                        "type": "tool_call_delta",
                        "index": index,
                        "name": block["name"],
                        "arg_delta": delta["partial_json"],
                    }
            elif dtype != "signature_delta":
                raise ValueError
        elif kind == "content_block_stop":
            index = event["index"]
            if index not in blocks or index in closed:
                raise ValueError
            closed.add(index)
        elif kind == "message_delta":
            usage.update(
                {k: v for k, v in (event.get("usage") or {}).items() if k in usage}
            )
        elif kind == "message_stop":
            if not started or closed != set(blocks):
                raise ValueError
            calls = []
            for index in sorted(blocks):
                block = blocks[index]
                if block["type"] != "tool_use":
                    continue
                arguments = block["arguments"] or json.dumps(block["input"])
                if not isinstance(json.loads(arguments), dict):
                    raise ValueError
                calls.append(
                    {"id": block["id"], "name": block["name"], "arguments": arguments}
                )
            if calls:
                yield {"type": "tool_calls", "calls": calls}
            from src.llm_core import _normalize_usage_counts

            normalized = _normalize_usage_counts(
                **{k.replace("_tokens", "_value"): v for k, v in usage.items()}
            )
            if normalized is None:
                raise ValueError
            yield {"type": "usage", "data": normalized}
            return
        elif kind != "ping":
            raise ValueError
    raise SubscriptionError(
        502, "Claude subscription stream ended before message_stop."
    )


NOT_CONNECTED = "Claude subscription is not connected for this Odysseus account."
NOT_SIGNED_IN = "Claude subscription requires explicit operator credentials."


def is_claude_subscription_base(url):
    try:
        return urlparse(str(url or "")).scheme == PROVIDER
    except ValueError:
        return False


async def probe():
    """Resolve operator credentials and fetch the live account catalog."""
    from src.claude_subscription_credentials import operator_source

    models = await Transport(operator_source(), timeout=30).list_models()
    return {
        "credentials_configured": True,
        "plan": None,
        "email": None,
        "models": models,
        "upstream_verified": True,
    }


def _database_handles():
    from core.database import ModelEndpoint, ProviderAuthSession, SessionLocal

    return ModelEndpoint, ProviderAuthSession, SessionLocal


def new_capability() -> str:
    return secrets.token_urlsafe(32)


def capability_from_headers(headers: Optional[Dict[str, Any]]) -> str:
    for key, value in (headers or {}).items():
        if (
            str(key).lower() == "authorization"
            and isinstance(value, str)
            and value.startswith("Bearer ")
        ):
            return value[len("Bearer ") :].strip()
    return ""


def capability_is_valid(token: Optional[str]) -> bool:
    """True when ``token`` belongs to a connected, enabled direct subscription endpoint
    that is owned by the account that connected it."""
    if not token:
        return False
    ModelEndpoint, ProviderAuthSession, SessionLocal = _database_handles()
    db = SessionLocal()
    try:
        for auth in (
            db.query(ProviderAuthSession)
            .filter(ProviderAuthSession.provider == PROVIDER)
            .all()
        ):
            stored = auth.access_token or ""
            if stored and hmac.compare_digest(stored.encode(), token.encode()):
                return (
                    db.query(ModelEndpoint.id)
                    .filter(
                        ModelEndpoint.provider_auth_id == auth.id,
                        ModelEndpoint.base_url == BASE_URL,
                        ModelEndpoint.owner == auth.owner,
                        ModelEndpoint.is_enabled == True,  # noqa: E712 - SQLAlchemy comparison
                    )
                    .first()
                    is not None
                )
        return False
    finally:
        db.close()


def resolve_runtime_credentials(
    auth_id: str, owner: Optional[str] = None
) -> Dict[str, Any]:
    """Owner-scoped capability for endpoint resolution (``resolve_endpoint_runtime``)."""
    _ModelEndpoint, ProviderAuthSession, SessionLocal = _database_handles()
    db = SessionLocal()
    try:
        row = (
            db.query(ProviderAuthSession)
            .filter(
                ProviderAuthSession.id == auth_id,
                ProviderAuthSession.provider == PROVIDER,
                ProviderAuthSession.owner == owner,
            )
            .first()
        )
        if row is None or not row.access_token:
            raise SubscriptionError(403, NOT_CONNECTED, fallback_eligible=False)
        return {"provider": PROVIDER, "base_url": BASE_URL, "api_key": row.access_token}
    finally:
        db.close()


def models_for_capability_sync(token, *, timeout=30):
    if not capability_is_valid(token):
        return []
    from src.claude_subscription_credentials import operator_source

    def discover():
        return asyncio.run(Transport(operator_source(), timeout=timeout).list_models())

    # Model probes can run from synchronous helpers inside an active event loop.
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return discover()
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=1) as executor:
        return executor.submit(discover).result()


_LEGACY = (
    "claude-3",
    "claude-opus-4-0",
    "claude-opus-4-1",
    "claude-opus-4-2025",
    "claude-sonnet-4-0",
    "claude-sonnet-4-2025",
    "claude-opus-4-5",
    "claude-sonnet-4-5",
    "claude-haiku-4",
)


def _apply_thinking(payload, model, mode, effort):
    """Hermes built-in thinking policy, without guessing unsupported options."""
    if mode not in (None, "", "default", "off", "on") or effort not in (
        None,
        "",
        "default",
        "low",
        "medium",
        "high",
        "xhigh",
        "max",
    ):
        raise _invalid()
    normalized = model.lower().replace(".", "-")
    legacy = normalized in {"claude-opus-4", "claude-sonnet-4"} or any(
        part in normalized for part in _LEGACY
    )
    old_haiku = "haiku" in normalized and "haiku-5" not in normalized
    adaptive = "claude" in normalized and not legacy and not old_haiku
    if adaptive and not any(part in normalized for part in ("opus-4-6", "sonnet-4-6")):
        payload.pop("temperature", None)
    if mode == "off":
        # Mandatory-thinking families reject an explicit disable; like Hermes,
        # "off" sends no thinking field and the model keeps its default.
        if any(
            part in normalized
            for part in ("claude-fable", "claude-mythos", "claude-opus-5-5")
        ):
            return
        if "claude-sonnet-5-5" in normalized:
            payload["thinking"] = {"type": "between_tools"}
        elif adaptive:
            payload["thinking"] = {"type": "disabled"}
        return
    if mode != "on" and effort in (None, "", "default"):
        return
    effort = effort if effort not in (None, "", "default") else "medium"
    if old_haiku or "claude" not in normalized:
        raise SubscriptionError(400, "Explicit thinking is unsupported for this model.")
    if adaptive:
        if effort == "xhigh" and any(
            part in normalized for part in ("opus-4-6", "sonnet-4-6")
        ):
            effort = "max"
        payload.update(
            thinking={"type": "adaptive", "display": "summarized"},
            output_config={"effort": effort},
        )
    else:
        budgets = {"low": 4000, "medium": 8000, "high": 16000, "xhigh": 32000}
        if effort not in budgets or payload["max_tokens"] <= budgets[effort]:
            raise SubscriptionError(
                400, "Thinking budget must fit below max_tokens on this model."
            )
        payload.update(
            thinking={"type": "enabled", "budget_tokens": budgets[effort]},
            temperature=1,
        )
