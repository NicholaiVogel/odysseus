"""Opt-in sources for the local direct OAuth reference; no ambient discovery."""

import asyncio
import hashlib
import json
import math
import os
import tempfile
import time
from contextlib import asynccontextmanager
from pathlib import Path

from src.claude_subscription import SubscriptionError


class StaticToken:
    """Operator-supplied setup token; never refreshes or writes a file."""

    def __init__(self, token):
        if not isinstance(token, str) or not token or any(c.isspace() for c in token):
            raise SubscriptionError(
                401, "Claude subscription credential is missing or invalid."
            )
        self._token = token

    async def token(self, client, *, failed_token=None):
        if failed_token is not None:
            raise SubscriptionError(
                401, "Claude subscription requires a fresh operator token."
            )
        return self._token


TOKEN_URL = "https://platform.claude.com/v1/oauth/token"
CLIENT_ID = "9d1c250a-e61b-44d9-88ed-5944d1962f5e"


def _credential_error():
    return SubscriptionError(
        401,
        "Claude subscription credentials unavailable; operator reauthentication required.",
    )


def _private_write(path, payload):
    """Private from creation, fsync data + directory, atomic same-filesystem replace."""
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


class FileCredentials:
    """Explicit operator file only. No Keychain, home, SDK, pool or .env reads.

    The source's lock covers reread → refresh → durable write across processes.
    A durable intent fingerprint is committed BEFORE spending a single-use token:
    crash, cancellation or an ambiguous response then requires operator reauth,
    rather than risk replay. A new credential pair naturally clears that verdict.
    Only the operator should write this file; CLI processes do not share our lock.
    """

    def __init__(self, path, *, token_url=TOKEN_URL, lock_timeout=15):
        self.path = Path(path).expanduser().resolve()
        self.token_url = token_url
        self.lock_timeout = lock_timeout

    def read(self):
        try:
            if self.path.stat().st_size > 1024 * 1024:
                raise ValueError
            data = json.loads(self.path.read_text(encoding="utf-8-sig"))
            record = data["claudeAiOauth"]
            token = record["accessToken"]
            StaticToken(token)
            expiry = record.get("expiresAt", 0)
            if (
                isinstance(expiry, bool)
                or not isinstance(expiry, (int, float))
                or not math.isfinite(expiry)
            ):
                raise ValueError
            return data, record
        except (OSError, ValueError, TypeError, KeyError):
            raise _credential_error() from None

    @asynccontextmanager
    async def _lock(self):
        try:
            import fcntl
        except ImportError:
            raise SubscriptionError(
                503, "File-backed OAuth refresh requires POSIX file locking."
            ) from None
        fd = os.open(
            str(self.path) + ".odysseus.lock",
            os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW,
            0o600,
        )
        try:
            async with asyncio.timeout(self.lock_timeout):
                while True:
                    try:
                        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                        break
                    except BlockingIOError:
                        await asyncio.sleep(0.01)
            yield
        finally:
            os.close(fd)

    async def token(self, client, *, failed_token=None):
        try:
            async with self._lock():
                data, record = self.read()
                token = record["accessToken"]
                refresh = record.get("refreshToken")
                fingerprint = hashlib.sha256(str(refresh or token).encode()).hexdigest()
                intent = self.path.with_name(self.path.name + ".odysseus-rotation.json")
                if (
                    intent.exists()
                    and json.loads(intent.read_text()).get("spent") == fingerprint
                ):
                    raise _credential_error()
                valid = (
                    not record.get("expiresAt")
                    or record["expiresAt"] > time.time() * 1000 + 60000
                )
                if valid and (failed_token is None or token != failed_token):
                    return token
                if not isinstance(refresh, str) or not refresh:
                    raise _credential_error()
                _private_write(intent, {"spent": fingerprint})
                response = await client.post(
                    self.token_url,
                    data={
                        "grant_type": "refresh_token",
                        "refresh_token": refresh,
                        "client_id": CLIENT_ID,
                    },
                    headers={"User-Agent": "axios/1.7.9"},
                    timeout=10,
                )
                if response.status_code != 200:
                    raise _credential_error()
                rotated = response.json()
                access = rotated["access_token"]
                StaticToken(access)
                expires = rotated.get("expires_in", 3600)
                if (
                    isinstance(expires, bool)
                    or not isinstance(expires, (int, float))
                    or not math.isfinite(expires)
                    or expires <= 0
                ):
                    raise _credential_error()
                next_refresh = rotated.get("refresh_token", refresh)
                if not isinstance(next_refresh, str) or not next_refresh:
                    raise _credential_error()
                record.update(
                    accessToken=access,
                    refreshToken=next_refresh,
                    expiresAt=int(time.time() * 1000 + expires * 1000),
                )
                _private_write(self.path, data)
                # Clear intent only after the new pair is durable (also handles
                # providers that reissue an access token without rotating refresh).
                _private_write(intent, {"spent": None})
                return access
        except SubscriptionError:
            raise
        except TimeoutError:
            raise SubscriptionError(
                504, "Claude credential refresh lock timed out."
            ) from None
        except Exception:
            raise _credential_error() from None


def operator_source():
    """Presence of exactly one namespaced operator setting is explicit opt-in."""
    token = os.getenv("ODYSSEUS_CLAUDE_SUBSCRIPTION_TOKEN", "")
    path = os.getenv("ODYSSEUS_CLAUDE_SUBSCRIPTION_CREDENTIALS_FILE", "")
    if bool(token) == bool(path):
        raise SubscriptionError(
            503,
            "Set exactly one explicit Claude subscription token or credentials file setting.",
        )
    return StaticToken(token) if token else FileCredentials(path)
