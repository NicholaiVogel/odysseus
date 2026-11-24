"""Direct reference connect, capability binding and owner-scope contracts.

The explicit operator source serves one Odysseus account. No real source is
opened; tests provision disposable database rows and synthetic credentials.
"""

import asyncio
import json
import threading
import types
from concurrent.futures import ThreadPoolExecutor

import pytest
from fastapi import HTTPException
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

import routes.chat_helpers as chat_helpers
import routes.claude_subscription_routes as ccr
import routes.model_routes as model_routes
import src.endpoint_resolver as endpoint_resolver
from core.database import Base, ModelEndpoint, ProviderAuthSession
from core.database import Session as DbSession
from src import claude_subscription
from src.chat_helpers import model_supports_vision
from src.model_pricing import is_subscription_endpoint

MODELS = ["claude-opus-5-5", "claude-sonnet-5-5", "claude-haiku-5-5"]
SIGNED_IN = {
    "credentials_configured": True,
    "plan": "max",
    "email": "op@example.com",
    "models": MODELS,
}


@pytest.fixture
def db(monkeypatch):
    # One shared connection: the routes run their DB work in worker threads.
    engine = create_engine(
        "sqlite:///:memory:",
        poolclass=StaticPool,
        connect_args={"check_same_thread": False},
    )
    Base.metadata.create_all(bind=engine)
    TestSessionLocal = sessionmaker(bind=engine, autoflush=False)
    monkeypatch.setattr(ccr, "SessionLocal", TestSessionLocal)
    monkeypatch.setattr(chat_helpers, "SessionLocal", TestSessionLocal)
    monkeypatch.setattr(
        claude_subscription,
        "_database_handles",
        lambda: (ModelEndpoint, ProviderAuthSession, TestSessionLocal),
    )
    monkeypatch.setattr(ccr, "_invalidate_models_cache", lambda: None)
    return TestSessionLocal


def _rows(TestSessionLocal):
    session = TestSessionLocal()
    try:
        return (
            session.query(ProviderAuthSession)
            .filter_by(provider=claude_subscription.PROVIDER)
            .all(),
            session.query(ModelEndpoint)
            .filter_by(base_url=claude_subscription.BASE_URL)
            .all(),
        )
    finally:
        session.close()


def _request(user="alice"):
    return types.SimpleNamespace(
        state=types.SimpleNamespace(current_user=user), headers={}, cookies={}
    )


@pytest.fixture
def as_admin(monkeypatch):
    monkeypatch.setattr(ccr, "require_admin", lambda request: None)
    monkeypatch.setattr(ccr, "_owner", lambda request: request.state.current_user)


def _probe_returns(monkeypatch, info):
    async def probe(**_kwargs):
        return dict(info)

    monkeypatch.setattr(claude_subscription, "probe", probe)


# ── Provisioning ────────────────────────────────────────────────────────────


def test_connect_creates_an_owner_private_native_tool_endpoint(db):
    result = ccr._provision_endpoint("alice", MODELS)
    auths, endpoints = _rows(db)
    assert len(auths) == 1 and len(endpoints) == 1
    auth, ep = auths[0], endpoints[0]
    assert auth.owner == "alice" and auth.auth_mode == claude_subscription.AUTH_MODE
    assert len(auth.access_token) >= 32
    assert ep.owner == "alice"
    assert ep.provider_auth_id == auth.id
    assert ep.supports_tools is True
    assert ep.endpoint_kind == "api" and ep.model_refresh_mode == "manual"
    assert ep.api_key is None
    assert json.loads(ep.cached_models) == MODELS
    assert auth.access_token not in json.dumps(result)


def test_reconnecting_keeps_one_endpoint_and_the_same_capability(db):
    first = ccr._provision_endpoint("alice", MODELS)
    token = _rows(db)[0][0].access_token
    second = ccr._provision_endpoint("alice", MODELS[:1])
    auths, endpoints = _rows(db)
    assert (len(auths), len(endpoints)) == (1, 1)
    assert second["id"] == first["id"]
    assert auths[0].access_token == token
    assert json.loads(endpoints[0].cached_models) == MODELS[:1]


def test_one_claude_login_serves_one_odysseus_account(db):
    ccr._provision_endpoint("alice", MODELS)
    with pytest.raises(ccr.SubscriptionAlreadyConnected):
        ccr._provision_endpoint("bob", MODELS)
    auths, endpoints = _rows(db)
    assert [a.owner for a in auths] == ["alice"] and [e.owner for e in endpoints] == [
        "alice"
    ]


@pytest.mark.parametrize("owners", [("alice", "bob"), ("alice", "alice")])
def test_concurrent_first_connects_leave_one_owner_and_one_endpoint(
    monkeypatch, tmp_path, owners
):
    # A file database gives each thread its own connection, like separate
    # requests or server workers.
    engine = create_engine(
        f"sqlite:///{tmp_path / 'race.db'}", connect_args={"check_same_thread": False}
    )
    Base.metadata.create_all(bind=engine)
    Sessions = sessionmaker(bind=engine, autoflush=False)
    monkeypatch.setattr(ccr, "SessionLocal", Sessions)
    monkeypatch.setattr(
        claude_subscription,
        "_database_handles",
        lambda: (ModelEndpoint, ProviderAuthSession, Sessions),
    )
    monkeypatch.setattr(ccr, "_invalidate_models_cache", lambda: None)

    # Both connects read the empty table before either writes.
    barrier = threading.Barrier(2)
    waited = threading.local()
    read = ccr._provider_auths

    def read_then_wait(db):
        rows = read(db)
        if not getattr(waited, "done", False):
            waited.done = True
            barrier.wait(timeout=5)
        return rows

    monkeypatch.setattr(ccr, "_provider_auths", read_then_wait)

    def connect(owner):
        try:
            ccr._provision_endpoint(owner, MODELS)
            return owner
        except ccr.SubscriptionAlreadyConnected:
            return None

    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            connected = [owner for owner in pool.map(connect, owners) if owner]
        auths, endpoints = _rows(Sessions)
    finally:
        engine.dispose()
    assert len(set(connected)) == 1
    assert [a.owner for a in auths] == connected[:1]
    assert [e.owner for e in endpoints] == connected[:1]
    assert [e.provider_auth_id for e in endpoints] == [a.id for a in auths]


@pytest.mark.parametrize(
    "foreign",
    [
        # An unrelated endpoint whose id collides with a fixed id.
        {"id": "claude-subscription", "provider_auth_id": "bob-auth"},
        # Another account's endpoint still linked to this provider's auth row id.
        {"id": "bob-ep", "provider_auth_id": "claude-subscription"},
    ],
)
def test_connect_never_takes_over_another_accounts_endpoint(db, foreign):
    session = db()
    try:
        session.add(
            ModelEndpoint(
                name="Bob's endpoint",
                base_url="https://api.example.com/v1",
                api_key="bob-key",
                owner="bob",
                is_enabled=True,
                **foreign,
            )
        )
        session.commit()
    finally:
        session.close()
    ccr._provision_endpoint("alice", MODELS)
    session = db()
    try:
        bobs = session.query(ModelEndpoint).filter_by(id=foreign["id"]).one()
        assert (bobs.owner, bobs.base_url, bobs.api_key, bobs.provider_auth_id) == (
            "bob",
            "https://api.example.com/v1",
            "bob-key",
            foreign["provider_auth_id"],
        )
        auth = (
            session.query(ProviderAuthSession)
            .filter_by(provider=claude_subscription.PROVIDER)
            .one()
        )
        mine = session.query(ModelEndpoint).filter_by(owner="alice").one()
        assert (mine.provider_auth_id, mine.base_url) == (
            auth.id,
            claude_subscription.BASE_URL,
        )
        # Only alice's own endpoint keeps her capability valid.
        mine.is_enabled = False
        session.commit()
        assert not claude_subscription.capability_is_valid(auth.access_token)
    finally:
        session.close()


def test_connect_route_requires_admin(db, monkeypatch):
    def deny(request):
        raise HTTPException(403, "Admin only")

    monkeypatch.setattr(ccr, "require_admin", deny)
    _probe_returns(monkeypatch, SIGNED_IN)
    with pytest.raises(HTTPException) as excinfo:
        asyncio.run(ccr.connect(_request()))
    assert excinfo.value.status_code == 403
    assert _rows(db) == ([], [])


def test_connect_route_refuses_when_the_server_is_not_credentials_configured(
    db, as_admin, monkeypatch
):
    _probe_returns(
        monkeypatch,
        {
            "credentials_configured": False,
            "plan": None,
            "email": None,
            "models": MODELS,
        },
    )
    with pytest.raises(HTTPException) as excinfo:
        asyncio.run(ccr.connect(_request()))
    assert excinfo.value.status_code == 409
    assert "operator" in excinfo.value.detail
    assert _rows(db) == ([], [])


def test_connect_route_reports_another_account_as_a_conflict(db, as_admin, monkeypatch):
    _probe_returns(monkeypatch, SIGNED_IN)
    asyncio.run(ccr.connect(_request("alice")))
    with pytest.raises(HTTPException) as excinfo:
        asyncio.run(ccr.connect(_request("bob")))
    assert excinfo.value.status_code == 409


def test_status_route_reports_login_and_connection(db, as_admin, monkeypatch):
    _probe_returns(monkeypatch, SIGNED_IN)
    before = asyncio.run(ccr.status(_request()))
    assert before["credentials_configured"] is True and before["connected"] is False
    asyncio.run(ccr.connect(_request()))
    after = asyncio.run(ccr.status(_request()))
    assert after["connected"] is True and after["models"] == MODELS
    other = asyncio.run(ccr.status(_request("bob")))
    assert other["connected"] is False and other["connected_by_other"] is True


# ── Capability ──────────────────────────────────────────────────────────────


def test_capability_is_valid_only_for_a_connected_enabled_endpoint(db):
    ccr._provision_endpoint("alice", MODELS)
    token = _rows(db)[0][0].access_token
    assert claude_subscription.capability_is_valid(token)
    assert not claude_subscription.capability_is_valid("guess")
    assert not claude_subscription.capability_is_valid("")
    session = db()
    try:
        session.query(ModelEndpoint).filter_by(
            base_url=claude_subscription.BASE_URL
        ).update({"is_enabled": False})
        session.commit()
    finally:
        session.close()
    assert not claude_subscription.capability_is_valid(token)


def test_capability_resolves_only_for_its_owner(db):
    ccr._provision_endpoint("alice", MODELS)
    session = db()
    try:
        ep = (
            session.query(ModelEndpoint)
            .filter_by(base_url=claude_subscription.BASE_URL)
            .one()
        )
        token = session.query(ProviderAuthSession).one().access_token
        assert endpoint_resolver.resolve_endpoint_runtime(ep, owner="alice") == (
            claude_subscription.BASE_URL,
            token,
        )
        with pytest.raises(claude_subscription.SubscriptionError):
            endpoint_resolver.resolve_endpoint_runtime(ep, owner="bob")
    finally:
        session.close()


def test_capability_is_request_local_and_never_persisted(db, monkeypatch):
    session = db()
    try:
        session.add(
            ModelEndpoint(
                id="ep1",
                name="Claude Code",
                base_url=claude_subscription.BASE_URL,
                provider_auth_id="auth1",
                owner="alice",
                is_enabled=True,
                api_key=None,
            )
        )
        session.add(
            DbSession(
                id="sess1",
                name="chat",
                endpoint_url=claude_subscription.BASE_URL,
                model="claude-opus-5-5",
                owner="alice",
                headers={},
            )
        )
        session.commit()
    finally:
        session.close()
    monkeypatch.setattr(
        endpoint_resolver,
        "resolve_endpoint_runtime",
        lambda ep, owner=None: (claude_subscription.BASE_URL, "cap"),
    )
    sess = types.SimpleNamespace(
        id="sess1",
        endpoint_url=claude_subscription.BASE_URL,
        model="claude-opus-5-5",
        owner="alice",
        headers={"Authorization": "Bearer stale"},
    )
    chat_helpers.resolve_session_auth(sess, "sess1", owner="alice")
    assert sess.headers == {"Authorization": "Bearer cap"}
    session = db()
    try:
        assert session.query(DbSession).filter_by(id="sess1").one().headers in (
            {},
            None,
        )
    finally:
        session.close()


# ── Model routes, pricing, vision ───────────────────────────────────────────


def test_endpoint_is_visible_only_to_its_owner():
    ep = types.SimpleNamespace(base_url=claude_subscription.BASE_URL, owner="alice")
    req = lambda user: types.SimpleNamespace(
        state=types.SimpleNamespace(current_user=user)
    )  # noqa: E731
    assert model_routes._chatgpt_endpoint_visible(ep, req("alice"))
    assert not model_routes._chatgpt_endpoint_visible(ep, req("bob"))


def test_model_discovery_requires_connected_capability(monkeypatch):
    from src.claude_subscription_credentials import StaticToken

    assert model_routes._is_discovery_only_provider("claude-subscription")
    monkeypatch.setattr(
        claude_subscription, "capability_is_valid", lambda token: token == "cap"
    )
    monkeypatch.setattr(
        "src.claude_subscription_credentials.operator_source",
        lambda: StaticToken("fixture"),
    )

    async def discovered(self, **kwargs):
        return ["claude-current"]

    monkeypatch.setattr(claude_subscription.Transport, "list_models", discovered)
    assert model_routes._probe_endpoint(claude_subscription.BASE_URL, "cap") == [
        "claude-current"
    ]
    assert model_routes._probe_endpoint(claude_subscription.BASE_URL, "guess") == []
    assert model_routes._probe_endpoint(claude_subscription.BASE_URL, None) == []


def test_plan_usage_is_not_priced_per_token():
    assert is_subscription_endpoint(claude_subscription.BASE_URL)
    # The non-secret route classification the chat stream reports to the UI.
    assert (
        endpoint_resolver.endpoint_cost_tracked(claude_subscription.BASE_URL, "api")
        is False
    )


def test_every_claude_subscription_model_is_treated_as_vision_capable(monkeypatch):
    import src.chat_helpers as helpers

    monkeypatch.setattr(
        helpers,
        "lmstudio_supports_vision",
        lambda *a, **k: pytest.fail("no HTTP probe"),
    )
    assert model_supports_vision("claude-fable-5-2", claude_subscription.BASE_URL)


def test_tampered_foreign_endpoint_cannot_resolve_owners_auth(db):
    ccr._provision_endpoint("alice", MODELS)
    session = db()
    try:
        auth = session.query(ProviderAuthSession).one()
        ep = ModelEndpoint(
            id="foreign",
            name="foreign",
            base_url=claude_subscription.BASE_URL,
            provider_auth_id=auth.id,
            owner="bob",
            is_enabled=True,
        )
        session.add(ep)
        session.commit()
        with pytest.raises(claude_subscription.SubscriptionError):
            endpoint_resolver.resolve_endpoint_runtime(ep, owner="alice")
    finally:
        session.close()


@pytest.mark.asyncio
async def test_asgi_connect_resolve_infer_and_owner_isolation(db, monkeypatch):
    import httpx
    from fastapi import FastAPI

    from src.llm_core import stream_llm
    from tests.test_claude_subscription_direct import tool
    from tests.test_claude_subscription_http import frame, reply, upstream

    monkeypatch.setenv("ODYSSEUS_CLAUDE_SUBSCRIPTION_TOKEN", "operator-fixture")
    monkeypatch.delenv("ODYSSEUS_CLAUDE_SUBSCRIPTION_CREDENTIALS_FILE", raising=False)
    monkeypatch.setattr("core.middleware.auth_disabled", lambda: False)
    app = FastAPI()
    app.state.auth_manager = types.SimpleNamespace(
        is_configured=True, is_admin=lambda user: user in {"alice", "bob"}
    )

    @app.middleware("http")
    async def fixture_identity(request, call_next):
        request.state.current_user = request.headers.get("x-fixture-user")
        return await call_next(request)

    app.include_router(ccr.setup_claude_subscription_routes())

    async def respond(reader, writer, requests):
        assert requests[-1][1]["authorization"] == "Bearer operator-fixture"
        if requests[-1][0].startswith("GET "):
            await reply(
                writer, {"data": [{"id": "claude-sonnet-current"}], "has_more": False}
            )
            return
        assert json.loads(requests[-1][2])["tools"][0]["name"] == "mcp__lookup"
        data = frame("message_start", message={"model": "actual"})
        data += frame(
            "content_block_start",
            index=0,
            content_block={
                "type": "tool_use",
                "id": "a",
                "name": "mcp__lookup",
                "input": {"q": "value"},
            },
        )
        data += frame("content_block_stop", index=0) + frame("message_stop")
        await reply(writer, data, content_type="text/event-stream")

    async with (
        upstream(respond) as (url, requests),
        httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client,
    ):
        monkeypatch.setattr(claude_subscription, "MESSAGES_URL", url)
        monkeypatch.setattr(claude_subscription, "MODELS_URL", url + "/v1/models")
        assert (
            await client.post("/api/claude-subscription/connect")
        ).status_code == 403
        connected = await client.post(
            "/api/claude-subscription/connect", headers={"x-fixture-user": "alice"}
        )
        assert connected.status_code == 200
        assert "access_token" not in connected.text
        assert (
            await client.post(
                "/api/claude-subscription/connect", headers={"x-fixture-user": "bob"}
            )
        ).status_code == 409
        state = (
            await client.get(
                "/api/claude-subscription/status", headers={"x-fixture-user": "alice"}
            )
        ).json()
        assert state["credentials_configured"] and state["upstream_verified"] is True
        assert state["models"] == ["claude-sonnet-current"]
        assert len(requests) == 3
        assert all(
            request[0].startswith("GET ") for request in requests
        )  # No inference spent.
        session = db()
        try:
            ep = session.query(ModelEndpoint).filter_by(id=connected.json()["id"]).one()
            assert ep.supports_tools is True
            base, cap = endpoint_resolver.resolve_endpoint_runtime(ep, owner="alice")
            with pytest.raises(claude_subscription.SubscriptionError):
                endpoint_resolver.resolve_endpoint_runtime(ep, owner="bob")
        finally:
            session.close()
        events = [
            c
            async for c in stream_llm(
                base,
                "claude-sonnet-4-6",
                [{"role": "user", "content": "go"}],
                tools=[tool("lookup")],
                headers={"Authorization": "Bearer " + cap},
                thinking_mode="off",
            )
        ]
        calls = [
            json.loads(c.split("data: ")[1]) for c in events if '"tool_calls"' in c
        ]
        assert calls[0]["calls"] == [
            {"id": "a", "name": "lookup", "arguments": '{"q": "value"}'}
        ]
        assert events[-1] == "data: [DONE]\n\n"
        assert len(requests) == 4
