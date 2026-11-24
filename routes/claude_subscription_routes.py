"""Direct OAuth reference routes: bind the explicit operator source to one admin.

Odysseus offers no Claude sign-in of its own. The operator opts in with a namespaced token or file setting;
these routes check configuration locally (no inference or refresh request) and create the owner-private endpoint that uses it. One
Claude login serves one Odysseus account, so a second account cannot connect it.
"""

import asyncio
import json
import logging
import uuid
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Request
from sqlalchemy.exc import IntegrityError

from core.database import ModelEndpoint, ProviderAuthSession, SessionLocal, utcnow_naive
from core.middleware import require_admin
from src import claude_subscription
from src.auth_helpers import effective_user, get_current_user

logger = logging.getLogger(__name__)

ALREADY_CONNECTED = (
    "Claude subscription on this server is already connected to another Odysseus account. "
    "One Claude login serves one account."
)
# The connection's auth row has a fixed id, so the database itself rejects a
# second, concurrent first connect (from another request or server worker)
# instead of letting two accounts each create one.
AUTH_ID = "claude-subscription"


class SubscriptionAlreadyConnected(RuntimeError):
    pass


def _owner(request: Request) -> Optional[str]:
    return effective_user(request) or get_current_user(request) or None


def _invalidate_models_cache() -> None:
    try:
        from routes.model_routes import _invalidate_models_cache as invalidate

        invalidate()
    except Exception:
        pass


def _new_id(db, model_cls) -> str:
    for _ in range(8):
        candidate = str(uuid.uuid4())[:8]
        if db.query(model_cls).filter(model_cls.id == candidate).first() is None:
            return candidate
    return uuid.uuid4().hex[:12]


def _provider_auths(db) -> List[Any]:
    return (
        db.query(ProviderAuthSession)
        .filter(ProviderAuthSession.provider == claude_subscription.PROVIDER)
        .order_by(ProviderAuthSession.created_at)
        .all()
    )


def _connection_state(owner: Optional[str]) -> Dict[str, Any]:
    db = SessionLocal()
    try:
        auths = _provider_auths(db)
        mine = [auth for auth in auths if (auth.owner or None) == (owner or None)]
        endpoint = None
        if mine:
            endpoint = (
                db.query(ModelEndpoint)
                .filter(
                    ModelEndpoint.provider_auth_id == mine[0].id,
                    ModelEndpoint.owner == owner,
                    ModelEndpoint.base_url == claude_subscription.BASE_URL,
                    ModelEndpoint.is_enabled == True,
                )  # noqa: E712
                .first()
            )
        return {
            "connected": endpoint is not None,
            "endpoint_id": endpoint.id if endpoint is not None else None,
            "connected_by_other": any(
                (auth.owner or None) != (owner or None) for auth in auths
            ),
        }
    finally:
        db.close()


def _write_endpoint(owner: Optional[str], models: List[str]) -> Dict[str, Any]:
    db = SessionLocal()
    try:
        auths = _provider_auths(db)
        if any((auth.owner or None) != (owner or None) for auth in auths):
            raise SubscriptionAlreadyConnected(ALREADY_CONNECTED)
        auth = auths[0] if auths else None
        if auth is None:
            auth = ProviderAuthSession(
                id=AUTH_ID,
                provider=claude_subscription.PROVIDER,
                owner=owner,
                label=claude_subscription.DISPLAY_NAME,
                base_url=claude_subscription.BASE_URL,
                access_token=claude_subscription.new_capability(),
                auth_mode=claude_subscription.AUTH_MODE,
            )
            db.add(auth)
        elif not auth.access_token:
            auth.access_token = claude_subscription.new_capability()
        auth.last_refresh = utcnow_naive()

        # Reuse only this account's own endpoint for this connection. Never take
        # over another endpoint, whatever its id or link says.
        ep = (
            db.query(ModelEndpoint)
            .filter(
                ModelEndpoint.provider_auth_id == auth.id, ModelEndpoint.owner == owner
            )
            .order_by(ModelEndpoint.created_at)
            .first()
        )
        if ep is None:
            ep = ModelEndpoint(
                id=_new_id(db, ModelEndpoint),
                name=claude_subscription.DISPLAY_NAME,
                base_url=claude_subscription.BASE_URL,
                model_type="llm",
                endpoint_kind="api",
                owner=owner,
            )
            db.add(ep)
        ep.base_url = claude_subscription.BASE_URL
        ep.api_key = None
        ep.provider_auth_id = auth.id
        ep.is_enabled = True
        # Native tools remain in Odysseus; the transport only encodes names.
        ep.supports_tools = True
        ep.model_type = "llm"
        ep.endpoint_kind = "api"
        ep.model_refresh_mode = "manual"
        ep.cached_models = json.dumps(models)
        db.commit()
        result = {
            "id": ep.id,
            "name": ep.name,
            "base_url": ep.base_url,
            "models": list(models),
            "provider_auth_id": auth.id,
        }
    finally:
        db.close()
    return result


def _provision_endpoint(owner: Optional[str], models: List[str]) -> Dict[str, Any]:
    """Create or refresh this owner's Claude subscription auth row and endpoint."""
    try:
        result = _write_endpoint(owner, models)
    except IntegrityError:
        # A concurrent first connect wrote the fixed-id rows first. Read them
        # again: the same owner refreshes them, another owner gets the conflict.
        result = _write_endpoint(owner, models)
    _invalidate_models_cache()
    return result


async def _probe_or_http_error() -> Dict[str, Any]:
    try:
        return await claude_subscription.probe()
    except claude_subscription.SubscriptionError as exc:
        raise HTTPException(
            exc.status if exc.status in (503, 504) else 502, exc.message
        )


async def status(request: Request) -> Dict[str, Any]:
    """Login and connection state for the admin UI. Never returns the capability."""
    require_admin(request)
    owner = _owner(request)
    state = await asyncio.to_thread(_connection_state, owner)
    try:
        info = await claude_subscription.probe()
    except claude_subscription.SubscriptionError as exc:
        info = {
            "credentials_configured": False,
            "plan": None,
            "email": None,
            "models": [],
            "error": exc.message,
        }
    return {**info, **state}


async def connect(request: Request) -> Dict[str, Any]:
    """Create (or refresh) the calling admin's Claude subscription endpoint."""
    require_admin(request)
    owner = _owner(request)
    info = await _probe_or_http_error()
    if not info.get("credentials_configured"):
        raise HTTPException(409, claude_subscription.NOT_SIGNED_IN)
    models = info.get("models") or []
    if not models:
        raise HTTPException(
            502, "Claude subscription reported no models for this login."
        )
    try:
        result = await asyncio.to_thread(_provision_endpoint, owner, models)
    except SubscriptionAlreadyConnected as exc:
        raise HTTPException(409, str(exc))
    return {**result, "plan": info.get("plan"), "email": info.get("email")}


def setup_claude_subscription_routes() -> APIRouter:
    router = APIRouter(prefix="/api/claude-subscription", tags=["claude-subscription"])
    router.add_api_route("/status", status, methods=["GET"])
    router.add_api_route("/connect", connect, methods=["POST"])
    return router
