# Direct Claude subscription OAuth — local reference

This is a **local comparison implementation**, not an upstream-supported provider,
not an assertion that Anthropic authorizes third-party subscription use, and not
the Claude Agent SDK. Do not publish this patch without a separate decision.
It calls Anthropic's Messages endpoint directly and presents the compatibility
identity described below. Upstream's existing API-key provider remains separate.

## Opt-in and ownership

Configure exactly one of these operator-owned environment settings on the
Odysseus server, then restart that server yourself:

- `ODYSSEUS_CLAUDE_SUBSCRIPTION_TOKEN`: explicit OAuth/setup token. It is used
  in memory, never refreshed, and never written to the database or a file.
- `ODYSSEUS_CLAUDE_SUBSCRIPTION_CREDENTIALS_FILE`: explicit path to a dedicated
  JSON file in the format below. This opt-in permits reading the file and
  committing rotated credentials back to it. Use an operator-only directory
  and mode `0600`. Do not share it with a concurrently running Claude CLI.

```json
{
  "claudeAiOauth": {
    "accessToken": "OPERATOR_SUPPLIED_OAUTH_TOKEN",
    "refreshToken": "OPERATOR_SUPPLIED_REFRESH_TOKEN",
    "expiresAt": 0,
    "scopes": ["user:inference"]
  }
}
```

`expiresAt` is an epoch timestamp in milliseconds. Zero/absent means expiry is
unknown; a rejected access token can still trigger refresh when a refresh token
is present. Nonzero expiry refreshes with a 60-second skew. Existing metadata in
this JSON is preserved on successful rotation. Never paste real tokens into
chat, documentation, command arguments, or source control.

There is **no ambient discovery** of `~/.claude`, Keychain, Hermes credentials,
`.env`, `CLAUDE_CODE_OAUTH_TOKEN`, or `ANTHROPIC_API_KEY` by this provider. The
application's existing configuration loader is unchanged. There is no OAuth
login/exchange UI, credential pool, account rotation, proxy-env support, or CLI
subprocess. `Transport(source)` is injectable for operators/tests; the normal
application path constructs only the explicit operator source.

Using an authenticated admin session:

1. `GET /api/claude-subscription/status` resolves the configured credentials and
   fetches the live catalog from Anthropic's `/v1/models`, with pagination and a
   bounded timeout. `upstream_verified: true` means catalog access succeeded,
   not that every listed model has been exercised for inference.
2. `POST /api/claude-subscription/connect` creates an enabled, owner-private
   endpoint at `claude-subscription://local`, with `supports_tools=True` and
   manual model refresh. Its catalog comes from that live discovery, not the
   application's static Anthropic model list. Reconnecting or refreshing the
   endpoint updates its cached catalog. Discovery failures are reported rather
   than hidden behind a stale fallback.
3. Select that endpoint through the existing model picker. No new UI is added.

A fixed `ProviderAuthSession` primary key binds this operator source to one
Odysseus owner even when first-connect requests race in separate workers.
Reconnecting reuses only that owner's linked endpoint; another account gets a
conflict and cannot take over an endpoint. The encrypted auth row contains an
**internal random capability, not Anthropic credentials**. Endpoint resolution
checks both endpoint and auth-row owner; inference validates the capability
against an enabled, correctly owned endpoint before opening the operator source.
Capabilities are request-local, not persisted in chat sessions. Deleting an
endpoint does not transfer the auth-row binding to another account.

## Exactly what is sent

Inference target: `https://api.anthropic.com/v1/messages`.

| Header | Value |
| --- | --- |
| `Authorization` | `Bearer <explicit or refreshed OAuth token>` |
| `Content-Type` | `application/json` |
| `anthropic-version` | `2023-06-01` |
| `anthropic-beta` | `interleaved-thinking-2025-05-14,fine-grained-tool-streaming-2025-05-14,claude-code-20250219,oauth-2025-04-20` |
| `User-Agent` | `claude-code/<detected or configured version> (external, cli)` |
| `x-app` | `cli` |

`ODYSSEUS_CLAUDE_SUBSCRIPTION_CLI_VERSION` overrides the compatibility version
with a `major.minor.patch` value. Otherwise the installed CLI's `--version` is
read once, with a five-second timeout and no login or inference subprocess.
Services with minimal PATH also check `~/.local/bin/claude`. When detection is
unavailable, set the version explicitly; no obsolete hardcoded fallback is used.
No `x-api-key` is sent. HTTPX adds its normal transport headers (Host, content
length, accepted encodings, connection defaults). Redirects and ambient proxy
settings are disabled so a bearer is not forwarded to another origin.

The first system block is exactly:

> You are Claude Code, Anthropic's official CLI for Claude.

That is a compatibility identity, not an accurate description of Odysseus or
an authorization claim. The rest of Odysseus's system/developer text, branding,
tool descriptions, domain-specific names, and parameter values remain unchanged.
This intentionally does **not** port Hermes's product/prose substitutions,
`memory`/`session_search` aliases, or billing-classifier workarounds.

Tool names follow Hermes's required wire prefix: already-`mcp__` names remain
unchanged; `mcp_foo` becomes `mcp__foo`; otherwise `foo` becomes `mcp__foo`.
The same map applies to outgoing schemas and **every historical tool_use**, both
OpenAI-style tool_calls and native blocks. Tool results keep their pairing IDs.
An exact reverse map for the current advertised tools decodes responses; unknown
returned names fail closed, never by blindly stripping a prefix. Ambiguous
names/collisions and encoded names longer than 64 characters fail locally rather
than silently renaming an Odysseus tool.

Refresh target: `https://platform.claude.com/v1/oauth/token`. It sends form data
`grant_type=refresh_token`, the explicit file's refresh token, and client ID
`9d1c250a-e61b-44d9-88ed-5944d1962f5e`, with `User-Agent: axios/1.7.9`, matching
the inspected Hermes built-in file-credential refresh path. No inference bearer
or compatibility system prompt is sent to the token endpoint. Unlike Hermes,
there is no fallback token host: ambiguous refresh delivery must not replay a
single-use refresh token elsewhere.

## Behavior and failure boundaries

- Native roles: system/developer instructions, user/assistant turns, images,
  native tool_use/tool_result blocks, OpenAI tool_calls/results, parallel tool
  results merged into a single user turn. Input history is not mutated.
- Streaming SSE decodes split network frames and split tool-input JSON, emits
  Odysseus text/thinking/tool_call_delta/tool_calls/model_actual/usage events,
  and requires `message_stop`. Truncation is an error, never a successful DONE.
- Nonstreaming `Transport.complete` returns native content with decoded tool
  names. Existing `llm_call`/`llm_call_async` string-returning contracts remain
  unchanged; agent native tool calls use the existing streaming contract.
- Thinking maps to manual budgets on legacy models and adaptive thinking/effort
  on modern models, following the inspected built-in policy. Legacy Haiku
  explicit thinking, unknown modes/efforts, insufficient manual budgets, and
  unsupported explicit thinking-off on mandatory-thinking models fail clearly.
- Unsupported message fields, tool types/options, and content kinds fail
  explicitly. This is not a generic passthrough for every Messages API option.
- One bounded inference retry after an initial HTTP 401, only when the source
  yields a refreshed/adopted token. Static tokens do not refresh. No retries
  after emitted output, 429/5xx, malformed responses, or dropped connections.
  Provider errors are fallback-ineligible to avoid silently switching billing.
- Refresh uses a bounded, cancellable, nonblocking POSIX file lock around
  reread → decision → POST → atomic private commit. Different processes adopt
  the winner's rotation rather than spend the same token twice. File-backed
  sources are POSIX-only; static-token transport does not require `fcntl`.
- A private durable rotation-intent fingerprint is written **before** refresh.
  Ambiguous delivery, cancellation, terminal rejection, or failed persistence
  requires operator reauthentication instead of replaying a possibly spent
  token. The sidecars are `<file>.odysseus.lock` and
  `<file>.odysseus-rotation.json`; neither stores plaintext tokens. Supply a new
  credential pair after reauthentication; do not erase the safety verdict and
  retry the old pair. External CLI writers do not honor this lock.
- Credential resolution and HTTP waits are bounded by request timeouts.
  Cancellation/early stream close unwinds the HTTP response and client sockets.
  Upstream bodies, source exceptions, and token values are not reflected in
  public errors. HTTP bodies/payloads are not logged by this provider.
- Subscription classification suppresses per-token cost estimates. It does
  **not** guarantee included quota, entitlement, exemption from extra usage,
  or the absence of charges imposed by Anthropic.

## Verification scope

See local `artifacts/claude-subscription-direct-pr.md` and the JUnit/JSON evidence
beside it. Verification uses disposable SQLite databases, fake operator files,
real loopback TCP HTTP upstreams, independent refresh subprocesses, and ASGI
connect/status routes feeding actual `llm_core` inference. No real credentials,
inference, refresh, or existing application instances were used. Full app/UI
startup and live subscription compatibility remain unverified.
