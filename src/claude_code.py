"""Claude Code provider: Claude plan inference through the unmodified Claude Code CLI.

Reference implementation. Odysseus drives the ``claude`` binary that ships inside
``claude-agent-sdk`` as a pure model, the same shape as the ChatGPT Subscription
provider: Claude Code's own tools are off, so Odysseus stays the only agent
(planning, tool selection and execution), and Odysseus' text tool protocol
travels inside the prompt.

Credentials: Odysseus holds none. The binary uses the login the operator made
with ``claude auth login`` on the server (``CLAUDE_CONFIG_DIR`` is honoured).
Odysseus never reads, stores or forwards Claude credentials, and never offers a
Claude sign-in of its own. The ``ProviderAuthSession`` row for this provider only
holds an Odysseus-internal capability token that binds the endpoint to the admin
who connected it.

Requests identify themselves as Odysseus through the SDK's client-app setting
(``CLAUDE_AGENT_SDK_CLIENT_APP``, sent in the User-Agent); Claude Code's own
identity and telemetry are left exactly as published.

``claude-agent-sdk`` is optional (it bundles the CLI and is large), so it is only
imported when a request actually runs.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import hmac
import logging
import os
import secrets
import sys
import tempfile
import weakref
from contextlib import asynccontextmanager, suppress
from pathlib import Path
from typing import Any, AsyncGenerator, Callable, Dict, Iterable, List, Optional, Tuple
from urllib.parse import urlparse

from src.constants import APP_VERSION

logger = logging.getLogger(__name__)

PROVIDER = "claude-code"
BASE_URL = "claude-code://local"
DISPLAY_NAME = "Claude Code"
AUTH_MODE = "claude-code-cli"
CLIENT_APP = f"odysseus/{APP_VERSION}"
CLIENT_APP_ENV = "CLAUDE_AGENT_SDK_CLIENT_APP"

# Host settings that would bill an API key, use a proxy or switch to a cloud
# provider instead of the signed-in Claude plan.
_SCRUBBED_ENV = (
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_AUTH_TOKEN",
    "ANTHROPIC_BASE_URL",
    "CLAUDE_CODE_USE_BEDROCK",
    "CLAUDE_CODE_USE_VERTEX",
    "CLAUDE_CODE_USE_FOUNDRY",
)
# Keep the host's Claude Code setup (CLAUDE.md, skills, plugins, hooks, MCP
# servers, custom commands and agents) out of Odysseus requests, never run user
# text as a slash command, and leave no transcripts in the host's ~/.claude.
_ISOLATION_ARGS: Dict[str, Optional[str]] = {
    "safe-mode": None,
    "disable-slash-commands": None,
    "no-session-persistence": None,
}
_EFFORT_LEVELS = frozenset({"low", "medium", "high", "xhigh", "max"})

# Time limits. A wedged CLI must never hold a request, a concurrency slot or a
# worker thread, so every wait on it is bounded. A request's ``timeout`` works
# like an HTTP read timeout: no single wait (start-up, the first event, the
# next event) may exceed it, however long the whole answer takes.
DEFAULT_TIMEOUT = 300.0
PROBE_TIMEOUT = 30.0
# Stop sends an interrupt, then disconnects. A CLI that never acknowledges the
# interrupt must not hold up the disconnect that ends it.
INTERRUPT_TIMEOUT = 5.0

TURN_FORMAT_NOTE = (
    "The conversation so far is given as <turn> blocks in order. Write only the "
    "assistant's reply to the last turn, without <turn> tags."
)

NOT_SIGNED_IN = (
    "Claude Code isn't signed in to a Claude plan on the Odysseus server. "
    "Run `claude auth login` there, then retry."
)
NOT_INSTALLED = (
    "The Claude Code provider needs claude-agent-sdk on the Odysseus server "
    "(pip install claude-agent-sdk)."
)
NOT_CONNECTED = "Claude Code is not connected for this Odysseus account."
NOT_RESPONDING = "Claude Code did not respond in time."
RATE_LIMITED = "Claude plan usage limit reached. Retry after it resets."
_ASSISTANT_ERRORS = {
    "authentication_failed": (401, NOT_SIGNED_IN),
    "billing_error": (402, "Claude declined the request for a billing reason. Check the plan at claude.ai."),
    "rate_limit": (429, RATE_LIMITED),
    "invalid_request": (400, "Claude Code rejected the request as invalid."),
    "server_error": (502, "Claude returned a server error."),
}


class ClaudeCodeError(RuntimeError):
    """A provider failure with the HTTP-style status Odysseus reports for it."""

    def __init__(self, status: int, message: str, *, fallback_eligible: bool = True):
        super().__init__(message)
        self.status = status
        self.message = message
        self.fallback_eligible = fallback_eligible


def is_claude_code_base(url: Optional[str]) -> bool:
    try:
        return urlparse(url or "").scheme == "claude-code"
    except ValueError:
        return False


# ── Messages ────────────────────────────────────────────────────────────────


def _text(value: str) -> Dict[str, Any]:
    return {"type": "text", "text": value}


def _image_block(url: str) -> Optional[Dict[str, Any]]:
    url = (url or "").strip()
    if url.startswith("data:"):
        header, sep, data = url.partition(",")
        if not sep or ";base64" not in header or not data:
            return None
        media_type = header[len("data:"):].split(";", 1)[0] or "image/png"
        return {"type": "image", "source": {"type": "base64", "media_type": media_type, "data": data}}
    if url.startswith(("https://", "http://")):
        return {"type": "image", "source": {"type": "url", "url": url}}
    return None


def _part_image_url(part: Dict[str, Any]) -> str:
    raw = part.get("image_url")
    if isinstance(raw, dict):
        raw = raw.get("url")
    if not raw:
        raw = part.get("image")
    return raw if isinstance(raw, str) else ""


def _content_blocks(content: Any) -> List[Dict[str, Any]]:
    if content is None:
        return []
    if not isinstance(content, list):
        return [_text(str(content))]
    blocks: List[Dict[str, Any]] = []
    for part in content:
        if isinstance(part, str):
            blocks.append(_text(part))
            continue
        if not isinstance(part, dict):
            continue
        kind = part.get("type")
        if kind in ("text", "input_text", "output_text"):
            blocks.append(_text(str(part.get("text") or "")))
        elif kind in ("image_url", "input_image", "image"):
            blocks.append(_image_block(_part_image_url(part)) or _text("[image attachment could not be forwarded]"))
        elif isinstance(part.get("text"), str):
            blocks.append(_text(part["text"]))
        else:
            blocks.append(_text(f"[{kind or 'unsupported'} attachment omitted]"))
    return blocks


def _tool_calls_text(message: Dict[str, Any]) -> str:
    lines = []
    for call in message.get("tool_calls") or []:
        function = (call or {}).get("function") or {}
        lines.append(f"[tool call {function.get('name') or 'unknown'}: {function.get('arguments') or ''}]")
    return "\n".join(lines)


def _merge_text(blocks: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Join adjacent text blocks; the API rejects empty text blocks."""
    merged: List[Dict[str, Any]] = []
    for block in blocks:
        if block["type"] == "text" and merged and merged[-1]["type"] == "text":
            merged[-1] = _text(merged[-1]["text"] + block["text"])
        else:
            merged.append(dict(block))
    return [block for block in merged if block["type"] != "text" or block["text"]]


def fold_messages(messages: Iterable[Dict[str, Any]]) -> Tuple[str, List[Dict[str, Any]]]:
    """OpenAI-style chat messages -> (system prompt, content of one user message).

    Claude Code takes one user message per request, so the conversation is sent
    as ``<turn>`` blocks. Every turn, including the last, is wrapped the same way
    and ends on a block boundary: adding a turn only appends blocks, so the
    previous request stays a cacheable prefix.
    """
    system_parts: List[str] = []
    blocks: List[Dict[str, Any]] = []
    for message in messages or []:
        if not isinstance(message, dict):
            continue
        role = str(message.get("role") or "user")
        if role == "system":
            text = "".join(b["text"] for b in _content_blocks(message.get("content")) if b["type"] == "text")
            if text.strip():
                system_parts.append(text)
            continue
        turn = [_text(f'<turn role="{role}">'), *_content_blocks(message.get("content"))]
        calls = _tool_calls_text(message)
        if calls:
            turn.append(_text(calls))
        turn.append(_text("</turn>"))
        blocks.extend(_merge_text(turn))
    if not blocks:
        raise ClaudeCodeError(400, "There is no message to send to Claude Code.", fallback_eligible=False)
    system_parts.append(TURN_FORMAT_NOTE)
    return "\n\n".join(system_parts), blocks


# ── CLI options ─────────────────────────────────────────────────────────────


def _workdir() -> str:
    """An empty working directory: Claude Code tells the model its cwd."""
    path = Path(os.getenv("ODYSSEUS_CLAUDE_CODE_WORKDIR") or Path(tempfile.gettempdir()) / "odysseus-claude-code")
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    return str(path)


def option_kwargs(model: Optional[str], system_prompt: str, *, effort: Optional[str] = None) -> Dict[str, Any]:
    """``ClaudeAgentOptions`` arguments for one inference-only request."""
    kwargs: Dict[str, Any] = {
        "model": model or None,
        "system_prompt": system_prompt or "",
        "tools": [],
        "setting_sources": [],
        "strict_mcp_config": True,
        "mcp_servers": {},
        "max_turns": 1,
        "include_partial_messages": True,
        "verbatim_prompts": True,
        "cwd": _workdir(),
        # Identify requests as Odysseus. Set directly (not ClaudeAgentOptions.client_app)
        # so SDK releases that predate that option send it too.
        "env": {**{key: "" for key in _SCRUBBED_ENV}, CLIENT_APP_ENV: CLIENT_APP},
        "extra_args": dict(_ISOLATION_ARGS),
    }
    if effort in _EFFORT_LEVELS:
        kwargs["effort"] = effort
    return kwargs


def models_from_server_info(info: Dict[str, Any]) -> List[str]:
    """Concrete model ids from the CLI's model list, in its order, deduplicated."""
    seen: List[str] = []
    for entry in (info or {}).get("models") or []:
        if not isinstance(entry, dict):
            continue
        model = entry.get("resolvedModel") or entry.get("value")
        if isinstance(model, str) and model.strip() and model.strip() not in seen:
            seen.append(model.strip())
    return seen


# ── Requests ────────────────────────────────────────────────────────────────

ClientFactory = Callable[[Dict[str, Any]], Any]
_SLOTS: "weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, asyncio.Semaphore]" = weakref.WeakKeyDictionary()


def _load_sdk():
    try:
        import claude_agent_sdk
    except ImportError as exc:
        raise ClaudeCodeError(503, NOT_INSTALLED, fallback_eligible=False) from exc
    return claude_agent_sdk


def _sdk_client(options: Dict[str, Any]):
    sdk = _load_sdk()
    return sdk.ClaudeSDKClient(options=sdk.ClaudeAgentOptions(**options))


@asynccontextmanager
async def _slot():
    """Bound concurrent CLI processes; each request runs its own."""
    loop = asyncio.get_running_loop()
    semaphore = _SLOTS.get(loop)
    if semaphore is None:
        limit = max(1, int(os.getenv("ODYSSEUS_CLAUDE_CODE_MAX_CONCURRENCY") or 4))
        semaphore = _SLOTS[loop] = asyncio.Semaphore(limit)
    async with semaphore:
        yield


def _plan_login(info: Optional[Dict[str, Any]]) -> bool:
    account = (info or {}).get("account") or {}
    return account.get("apiProvider") == "firstParty" and bool(account.get("subscriptionType"))


def _require_plan_login(info: Optional[Dict[str, Any]]) -> None:
    """Refuse to send anything unless the CLI is signed in to a Claude plan."""
    if not _plan_login(info):
        raise ClaudeCodeError(401, NOT_SIGNED_IN)


async def _within(awaitable: Any, seconds: float) -> Any:
    """Await one step of the CLI with a deadline; a miss is a 504 like a read timeout."""
    try:
        return await asyncio.wait_for(awaitable, seconds)
    except asyncio.TimeoutError as exc:
        raise ClaudeCodeError(504, NOT_RESPONDING) from exc


async def _end(client: Any, *, interrupt: bool) -> None:
    """Stop the CLI and end its process. Never raises."""
    if interrupt:
        with suppress(Exception):
            await asyncio.wait_for(client.interrupt(), INTERRUPT_TIMEOUT)
    with suppress(Exception):
        # No deadline here on purpose: the SDK bounds disconnect itself, and a
        # foreign cancellation inside it would skip its terminate/kill
        # escalation and leave the CLI process behind.
        await client.disconnect()


async def probe(
    *, client_factory: Optional[ClientFactory] = None, timeout: float = PROBE_TIMEOUT,
) -> Dict[str, Any]:
    """Start the CLI without sending a prompt and report its login and models.

    No API request is made: the CLI answers from its local state.
    """
    client = (client_factory or _sdk_client)(option_kwargs(None, ""))
    try:
        await _within(client.connect(), timeout)
        info = await _within(client.get_server_info(), timeout) or {}
    except ClaudeCodeError:
        raise
    except Exception as exc:
        raise _unexpected_error(exc) from exc
    finally:
        await _end(client, interrupt=False)
    account = info.get("account") or {}
    return {
        "signed_in": _plan_login(info),
        "plan": account.get("subscriptionType") or None,
        "email": account.get("email") or None,
        "models": models_from_server_info(info),
    }


async def _one_user_message(content: List[Dict[str, Any]]):
    yield {"type": "user", "message": {"role": "user", "content": content}, "parent_tool_use_id": None}


def _usage(raw: Optional[Dict[str, Any]]) -> Dict[str, int]:
    raw = raw or {}
    prompt = sum(
        int(raw.get(key) or 0)
        for key in ("input_tokens", "cache_creation_input_tokens", "cache_read_input_tokens")
    )
    return {"input_tokens": prompt, "output_tokens": int(raw.get("output_tokens") or 0)}


def _result_error(message: Any) -> ClaudeCodeError:
    status = getattr(message, "api_error_status", None)
    text = str(getattr(message, "result", "") or "").strip()
    if status == 429:
        return ClaudeCodeError(429, RATE_LIMITED)
    if status in (401, 403) or "not logged in" in text.lower():
        return ClaudeCodeError(401, NOT_SIGNED_IN)
    return ClaudeCodeError(502, f"Claude Code request failed: {text[:300]}" if text else "Claude Code request failed.")


def _unexpected_error(exc: Exception) -> ClaudeCodeError:
    sdk = sys.modules.get("claude_agent_sdk")
    not_found = getattr(sdk, "CLINotFoundError", None) if sdk is not None else None
    if not_found is not None and isinstance(exc, not_found):
        return ClaudeCodeError(
            503, "The Claude Code CLI was not found. Reinstall claude-agent-sdk, which bundles it.",
            fallback_eligible=False,
        )
    logger.warning("Claude Code request failed: %s", type(exc).__name__)
    return ClaudeCodeError(502, "Claude Code request failed.", fallback_eligible=False)


async def stream_events(
    model: str,
    messages: Iterable[Dict[str, Any]],
    *,
    effort: Optional[str] = None,
    timeout: Optional[float] = None,
    client_factory: Optional[ClientFactory] = None,
) -> AsyncGenerator[Tuple[str, Any], None]:
    """Run one request through the CLI.

    Yields ``("model", id)``, ``("thinking", text)``, ``("text", text)`` and a
    final ``("usage", {"input_tokens", "output_tokens"})``. Raises
    :class:`ClaudeCodeError`; a CLI that stays silent for ``timeout`` seconds
    (start-up included) is a 504. Closing the generator early (Stop) interrupts
    the CLI so it stops generating, then ends the process.
    """
    limit = timeout if timeout and timeout > 0 else DEFAULT_TIMEOUT
    system_prompt, content = fold_messages(messages)
    client = (client_factory or _sdk_client)(option_kwargs(model, system_prompt, effort=effort))
    completed = False
    reported_model = ""
    streamed_text = False
    try:
        async with _slot():
            await _within(client.connect(), limit)
            _require_plan_login(await _within(client.get_server_info(), limit))
            await _within(client.query(_one_user_message(content)), limit)
            responses = client.receive_response().__aiter__()
            while True:
                try:
                    message = await _within(responses.__anext__(), limit)
                except StopAsyncIteration:
                    break
                event = getattr(message, "event", None)
                if isinstance(event, dict):  # StreamEvent: a raw API stream event
                    if event.get("type") == "message_start":
                        model_id = str((event.get("message") or {}).get("model") or "")
                        if model_id and model_id != reported_model:
                            reported_model = model_id
                            yield ("model", model_id)
                    elif event.get("type") == "content_block_delta":
                        delta = event.get("delta") or {}
                        if delta.get("type") == "text_delta" and delta.get("text"):
                            streamed_text = True
                            yield ("text", delta["text"])
                        elif delta.get("type") == "thinking_delta" and delta.get("thinking"):
                            yield ("thinking", delta["thinking"])
                elif hasattr(message, "is_error"):  # ResultMessage
                    if message.is_error:
                        raise _result_error(message)
                    yield ("usage", _usage(getattr(message, "usage", None)))
                elif hasattr(message, "content") and hasattr(message, "model"):  # AssistantMessage
                    error = getattr(message, "error", None)
                    if error:
                        status, text = _ASSISTANT_ERRORS.get(str(error), (502, "Claude Code request failed."))
                        raise ClaudeCodeError(status, text)
                    model_id = str(message.model or "")
                    if model_id and model_id != "<synthetic>" and model_id != reported_model:
                        reported_model = model_id
                        yield ("model", model_id)
                    if not streamed_text:
                        for block in message.content or []:
                            text = getattr(block, "text", None)
                            if isinstance(text, str) and text:
                                yield ("text", text)
            completed = True
    except ClaudeCodeError:
        raise
    except Exception as exc:
        raise _unexpected_error(exc) from exc
    finally:
        await _end(client, interrupt=not completed)


# ── Authorization ───────────────────────────────────────────────────────────
#
# The CLI login belongs to whoever signed in on the server, so a request must
# prove it comes from the Odysseus account that connected it. The connect route
# (admin only) mints a random capability, stores it encrypted in this
# provider's ProviderAuthSession, and owner-scoped endpoint resolution puts it
# in that owner's session headers as a bearer. It is an Odysseus-internal token,
# never a Claude credential.


def _database_handles():
    from core.database import ModelEndpoint, ProviderAuthSession, SessionLocal

    return ModelEndpoint, ProviderAuthSession, SessionLocal


def new_capability() -> str:
    return secrets.token_urlsafe(32)


def capability_from_headers(headers: Optional[Dict[str, Any]]) -> str:
    for key, value in (headers or {}).items():
        if str(key).lower() == "authorization" and isinstance(value, str) and value.startswith("Bearer "):
            return value[len("Bearer "):].strip()
    return ""


def capability_is_valid(token: Optional[str]) -> bool:
    """True when ``token`` belongs to a connected, enabled Claude Code endpoint
    that is owned by the account that connected it."""
    if not token:
        return False
    ModelEndpoint, ProviderAuthSession, SessionLocal = _database_handles()
    db = SessionLocal()
    try:
        for auth in db.query(ProviderAuthSession).filter(ProviderAuthSession.provider == PROVIDER).all():
            stored = auth.access_token or ""
            if stored and hmac.compare_digest(stored.encode(), token.encode()):
                return db.query(ModelEndpoint.id).filter(
                    ModelEndpoint.provider_auth_id == auth.id,
                    ModelEndpoint.owner == auth.owner,
                    ModelEndpoint.is_enabled == True,  # noqa: E712 - SQLAlchemy comparison
                ).first() is not None
        return False
    finally:
        db.close()


def resolve_runtime_credentials(auth_id: str, owner: Optional[str] = None) -> Dict[str, Any]:
    """Owner-scoped capability for endpoint resolution (``resolve_endpoint_runtime``)."""
    _ModelEndpoint, ProviderAuthSession, SessionLocal = _database_handles()
    db = SessionLocal()
    try:
        row = db.query(ProviderAuthSession).filter(
            ProviderAuthSession.id == auth_id,
            ProviderAuthSession.provider == PROVIDER,
            ProviderAuthSession.owner == owner,
        ).first()
        if row is None or not row.access_token:
            raise ClaudeCodeError(403, NOT_CONNECTED, fallback_eligible=False)
        return {"provider": PROVIDER, "base_url": BASE_URL, "api_key": row.access_token}
    finally:
        db.close()


def models_for_capability_sync(token: Optional[str], *, timeout: float = PROBE_TIMEOUT) -> List[str]:
    """Model refresh for an already connected endpoint, callable from sync code.

    Only the connected owner's capability can make the server start the CLI.
    """
    if not capability_is_valid(token):
        return []
    pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    try:
        # probe() enforces the deadline inside the worker's own event loop, so
        # a wedged CLI is ended there instead of being left running with its
        # thread. The outer wait is only a backstop for the SDK's own cleanup.
        future = pool.submit(asyncio.run, probe(timeout=timeout))
        info = future.result(timeout=2 * timeout + 60)
    except Exception as exc:
        logger.warning("Claude Code model refresh failed: %s", type(exc).__name__)
        return []
    finally:
        pool.shutdown(wait=False, cancel_futures=True)
    return list(info.get("models") or []) if info.get("signed_in") else []
