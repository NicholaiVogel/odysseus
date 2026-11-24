"""Claude Code provider routes: connect the server's Claude Code login to one admin.

Odysseus offers no Claude sign-in of its own. The operator signs in on the
server with ``claude auth login``; these routes only check that login through the
CLI (no API request) and create the owner-private endpoint that uses it. One
Claude login serves one Odysseus account, so a second account cannot connect it.
"""

import asyncio
import json
import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Request
from sqlalchemy.exc import IntegrityError

from core.database import ModelEndpoint, ProviderAuthSession, SessionLocal, utcnow_naive
from core.middleware import require_admin
from src import claude_code
from src.auth_helpers import effective_user, get_current_user

logger = logging.getLogger(__name__)

ALREADY_CONNECTED = (
    "Claude Code on this server is already connected to another Odysseus account. "
    "One Claude login serves one account."
)
# The connection's auth row and endpoint have fixed ids, so the database itself
# rejects a second, concurrent first connect (from another request or server
# worker) instead of letting two accounts each create one.
AUTH_ID = "claude-code"
ENDPOINT_ID = "claude-code"


class ClaudeCodeAlreadyConnected(RuntimeError):
    pass


def _owner(request: Request) -> Optional[str]:
    return effective_user(request) or get_current_user(request) or None


def _invalidate_models_cache() -> None:
    try:
        from routes.model_routes import _invalidate_models_cache as invalidate

        invalidate()
    except Exception:
        pass


def _provider_auths(db) -> List[Any]:
    return (
        db.query(ProviderAuthSession)
        .filter(ProviderAuthSession.provider == claude_code.PROVIDER)
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
                .filter(ModelEndpoint.provider_auth_id == mine[0].id, ModelEndpoint.is_enabled == True)  # noqa: E712
                .first()
            )
        return {
            "connected": endpoint is not None,
            "endpoint_id": endpoint.id if endpoint is not None else None,
            "connected_by_other": any((auth.owner or None) != (owner or None) for auth in auths),
        }
    finally:
        db.close()


def _write_endpoint(owner: Optional[str], models: List[str]) -> Dict[str, Any]:
    db = SessionLocal()
    try:
        auths = _provider_auths(db)
        if any((auth.owner or None) != (owner or None) for auth in auths):
            raise ClaudeCodeAlreadyConnected(ALREADY_CONNECTED)
        auth = auths[0] if auths else None
        if auth is None:
            auth = ProviderAuthSession(
                id=AUTH_ID,
                provider=claude_code.PROVIDER,
                owner=owner,
                label=claude_code.DISPLAY_NAME,
                base_url=claude_code.BASE_URL,
                access_token=claude_code.new_capability(),
                auth_mode=claude_code.AUTH_MODE,
            )
            db.add(auth)
        elif not auth.access_token:
            auth.access_token = claude_code.new_capability()
        auth.last_refresh = utcnow_naive()

        ep = (
            db.query(ModelEndpoint)
            .filter(ModelEndpoint.provider_auth_id == auth.id)
            .order_by(ModelEndpoint.created_at)
            .first()
        ) or db.query(ModelEndpoint).filter(ModelEndpoint.id == ENDPOINT_ID).first()
        if ep is None:
            ep = ModelEndpoint(
                id=ENDPOINT_ID,
                name=claude_code.DISPLAY_NAME,
                base_url=claude_code.BASE_URL,
                model_type="llm",
                endpoint_kind="api",
                owner=owner,
            )
            db.add(ep)
        ep.owner = owner
        ep.base_url = claude_code.BASE_URL
        ep.api_key = None
        ep.provider_auth_id = auth.id
        ep.is_enabled = True
        # Inference only: Odysseus is the only agent and sends its own text tool
        # protocol, so no native tool schemas are ever offered on this route.
        ep.supports_tools = False
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
    """Create or refresh this owner's Claude Code auth row and endpoint."""
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
        return await claude_code.probe()
    except claude_code.ClaudeCodeError as exc:
        raise HTTPException(exc.status if exc.status in (503, 504) else 502, exc.message)


async def status(request: Request) -> Dict[str, Any]:
    """Login and connection state for the admin UI. Never returns the capability."""
    require_admin(request)
    owner = _owner(request)
    state = await asyncio.to_thread(_connection_state, owner)
    try:
        info = await claude_code.probe()
    except claude_code.ClaudeCodeError as exc:
        info = {"signed_in": False, "plan": None, "email": None, "models": [], "error": exc.message}
    return {**info, **state}


async def connect(request: Request) -> Dict[str, Any]:
    """Create (or refresh) the calling admin's Claude Code endpoint."""
    require_admin(request)
    owner = _owner(request)
    info = await _probe_or_http_error()
    if not info.get("signed_in"):
        raise HTTPException(409, claude_code.NOT_SIGNED_IN)
    models = info.get("models") or []
    if not models:
        raise HTTPException(502, "Claude Code reported no models for this login.")
    try:
        result = await asyncio.to_thread(_provision_endpoint, owner, models)
    except ClaudeCodeAlreadyConnected as exc:
        raise HTTPException(409, str(exc))
    return {**result, "plan": info.get("plan"), "email": info.get("email")}


def setup_claude_code_routes() -> APIRouter:
    router = APIRouter(prefix="/api/claude-code", tags=["claude-code"])
    router.add_api_route("/status", status, methods=["GET"])
    router.add_api_route("/connect", connect, methods=["POST"])
    return router
