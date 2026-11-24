"""Claude Code connect route, capability authorization and owner scoping.

The CLI login on the server belongs to whoever ran ``claude auth login`` there.
Odysseus binds it to exactly one Odysseus account: the admin who connected it.
That account's endpoint is owner-private, its capability resolves only for that
owner, and the capability never lands in the sessions table.
"""

import asyncio
import json
import types

import pytest
from fastapi import HTTPException
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

import routes.chat_helpers as chat_helpers
import routes.claude_code_routes as ccr
import routes.model_routes as model_routes
import src.endpoint_resolver as endpoint_resolver
from core.database import Base, ModelEndpoint, ProviderAuthSession, Session as DbSession
from src import claude_code
from src.chat_helpers import model_supports_vision
from src.model_pricing import is_subscription_endpoint

MODELS = ["claude-opus-5-5", "claude-sonnet-5-5", "claude-haiku-5-5"]
SIGNED_IN = {"signed_in": True, "plan": "max", "email": "op@example.com", "models": MODELS}


@pytest.fixture
def db(monkeypatch):
    # One shared connection: the routes run their DB work in worker threads.
    engine = create_engine(
        "sqlite:///:memory:", poolclass=StaticPool, connect_args={"check_same_thread": False},
    )
    Base.metadata.create_all(bind=engine)
    TestSessionLocal = sessionmaker(bind=engine, autoflush=False)
    monkeypatch.setattr(ccr, "SessionLocal", TestSessionLocal)
    monkeypatch.setattr(chat_helpers, "SessionLocal", TestSessionLocal)
    monkeypatch.setattr(
        claude_code, "_database_handles", lambda: (ModelEndpoint, ProviderAuthSession, TestSessionLocal),
    )
    monkeypatch.setattr(ccr, "_invalidate_models_cache", lambda: None)
    return TestSessionLocal


def _rows(TestSessionLocal):
    session = TestSessionLocal()
    try:
        return (
            session.query(ProviderAuthSession).filter_by(provider=claude_code.PROVIDER).all(),
            session.query(ModelEndpoint).filter_by(base_url=claude_code.BASE_URL).all(),
        )
    finally:
        session.close()


def _request(user="alice"):
    return types.SimpleNamespace(state=types.SimpleNamespace(current_user=user), headers={}, cookies={})


@pytest.fixture
def as_admin(monkeypatch):
    monkeypatch.setattr(ccr, "require_admin", lambda request: None)
    monkeypatch.setattr(ccr, "_owner", lambda request: request.state.current_user)


def _probe_returns(monkeypatch, info):
    async def probe(**_kwargs):
        return dict(info)
    monkeypatch.setattr(claude_code, "probe", probe)


# ── Provisioning ────────────────────────────────────────────────────────────


def test_connect_creates_an_owner_private_inference_only_endpoint(db):
    result = ccr._provision_endpoint("alice", MODELS)
    auths, endpoints = _rows(db)
    assert len(auths) == 1 and len(endpoints) == 1
    auth, ep = auths[0], endpoints[0]
    assert auth.owner == "alice" and auth.auth_mode == claude_code.AUTH_MODE
    assert len(auth.access_token) >= 32
    assert ep.owner == "alice"
    assert ep.provider_auth_id == auth.id
    assert ep.supports_tools is False
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
    with pytest.raises(ccr.ClaudeCodeAlreadyConnected):
        ccr._provision_endpoint("bob", MODELS)
    auths, endpoints = _rows(db)
    assert [a.owner for a in auths] == ["alice"] and [e.owner for e in endpoints] == ["alice"]


def test_connect_route_requires_admin(db, monkeypatch):
    def deny(request):
        raise HTTPException(403, "Admin only")
    monkeypatch.setattr(ccr, "require_admin", deny)
    _probe_returns(monkeypatch, SIGNED_IN)
    with pytest.raises(HTTPException) as excinfo:
        asyncio.run(ccr.connect(_request()))
    assert excinfo.value.status_code == 403
    assert _rows(db) == ([], [])


def test_connect_route_refuses_when_the_server_is_not_signed_in(db, as_admin, monkeypatch):
    _probe_returns(monkeypatch, {"signed_in": False, "plan": None, "email": None, "models": MODELS})
    with pytest.raises(HTTPException) as excinfo:
        asyncio.run(ccr.connect(_request()))
    assert excinfo.value.status_code == 409
    assert "claude auth login" in excinfo.value.detail
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
    assert before["signed_in"] is True and before["connected"] is False
    asyncio.run(ccr.connect(_request()))
    after = asyncio.run(ccr.status(_request()))
    assert after["connected"] is True and after["models"] == MODELS
    other = asyncio.run(ccr.status(_request("bob")))
    assert other["connected"] is False and other["connected_by_other"] is True


# ── Capability ──────────────────────────────────────────────────────────────


def test_capability_is_valid_only_for_a_connected_enabled_endpoint(db):
    ccr._provision_endpoint("alice", MODELS)
    token = _rows(db)[0][0].access_token
    assert claude_code.capability_is_valid(token)
    assert not claude_code.capability_is_valid("guess")
    assert not claude_code.capability_is_valid("")
    session = db()
    try:
        session.query(ModelEndpoint).filter_by(base_url=claude_code.BASE_URL).update({"is_enabled": False})
        session.commit()
    finally:
        session.close()
    assert not claude_code.capability_is_valid(token)


def test_capability_resolves_only_for_its_owner(db):
    ccr._provision_endpoint("alice", MODELS)
    session = db()
    try:
        ep = session.query(ModelEndpoint).filter_by(base_url=claude_code.BASE_URL).one()
        token = session.query(ProviderAuthSession).one().access_token
        assert endpoint_resolver.resolve_endpoint_runtime(ep, owner="alice") == (claude_code.BASE_URL, token)
        with pytest.raises(claude_code.ClaudeCodeError):
            endpoint_resolver.resolve_endpoint_runtime(ep, owner="bob")
    finally:
        session.close()


def test_capability_is_request_local_and_never_persisted(db, monkeypatch):
    session = db()
    try:
        session.add(ModelEndpoint(
            id="ep1", name="Claude Code", base_url=claude_code.BASE_URL,
            provider_auth_id="auth1", owner="alice", is_enabled=True, api_key=None,
        ))
        session.add(DbSession(
            id="sess1", name="chat", endpoint_url=claude_code.BASE_URL,
            model="claude-opus-5-5", owner="alice", headers={},
        ))
        session.commit()
    finally:
        session.close()
    monkeypatch.setattr(
        endpoint_resolver, "resolve_endpoint_runtime", lambda ep, owner=None: (claude_code.BASE_URL, "cap"),
    )
    sess = types.SimpleNamespace(
        id="sess1", endpoint_url=claude_code.BASE_URL, model="claude-opus-5-5", owner="alice",
        headers={"Authorization": "Bearer stale"},
    )
    chat_helpers.resolve_session_auth(sess, "sess1", owner="alice")
    assert sess.headers == {"Authorization": "Bearer cap"}
    session = db()
    try:
        assert session.query(DbSession).filter_by(id="sess1").one().headers in ({}, None)
    finally:
        session.close()


# ── Model routes, pricing, vision ───────────────────────────────────────────


def test_endpoint_is_visible_only_to_its_owner():
    ep = types.SimpleNamespace(base_url=claude_code.BASE_URL, owner="alice")
    req = lambda user: types.SimpleNamespace(state=types.SimpleNamespace(current_user=user))  # noqa: E731
    assert model_routes._chatgpt_endpoint_visible(ep, req("alice"))
    assert not model_routes._chatgpt_endpoint_visible(ep, req("bob"))


def test_model_discovery_runs_the_cli_only_for_the_connected_owner(monkeypatch):
    assert model_routes._is_discovery_only_provider("claude-code")
    monkeypatch.setattr(claude_code, "capability_is_valid", lambda token: token == "cap")
    _probe_returns(monkeypatch, SIGNED_IN)
    assert model_routes._probe_endpoint(claude_code.BASE_URL, "cap") == MODELS
    assert model_routes._probe_endpoint(claude_code.BASE_URL, "guess") == []
    assert model_routes._probe_endpoint(claude_code.BASE_URL, None) == []


def test_plan_usage_is_not_priced_per_token():
    assert is_subscription_endpoint(claude_code.BASE_URL)
    # The non-secret route classification the chat stream reports to the UI.
    assert endpoint_resolver.endpoint_cost_tracked(claude_code.BASE_URL, "api") is False


def test_every_claude_code_model_is_treated_as_vision_capable(monkeypatch):
    import src.chat_helpers as helpers
    monkeypatch.setattr(helpers, "lmstudio_supports_vision", lambda *a, **k: pytest.fail("no HTTP probe"))
    assert model_supports_vision("claude-fable-5-2", claude_code.BASE_URL)
