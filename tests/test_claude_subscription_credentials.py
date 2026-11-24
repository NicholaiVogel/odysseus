"""Credential fixtures are private disposable files, never operator accounts."""

import asyncio
import json
import os

import httpx
import pytest

from tests.test_claude_subscription_http import reply, upstream


def credential_file(tmp_path, **updates):
    path = tmp_path / "fake-credentials.json"
    record = {
        "accessToken": "old-fake",
        "refreshToken": "refresh-fake",
        "expiresAt": 1,
        "scopes": ["user:inference"],
        "subscriptionType": "max",
    }
    record.update(updates)
    path.write_text(json.dumps({"claudeAiOauth": record, "other": "preserved"}))
    return path


@pytest.mark.asyncio
async def test_expiry_refresh_race_two_sources_commit_one_rotation(tmp_path):
    from src.claude_subscription_credentials import FileCredentials

    path = credential_file(tmp_path)

    async def respond(reader, writer, requests):
        assert b"grant_type=refresh_token" in requests[-1][2]
        assert requests[-1][1]["user-agent"] == "axios/1.7.9"
        await reply(
            writer,
            {
                "access_token": "rotated-fake",
                "refresh_token": "next-fake",
                "expires_in": 3600,
            },
        )

    async with (
        upstream(respond) as (url, requests),
        httpx.AsyncClient(trust_env=False) as client,
    ):
        sources = [
            FileCredentials(path, token_url=url),
            FileCredentials(path, token_url=url),
        ]
        tokens = await asyncio.gather(*(source.token(client) for source in sources))
        # Stale 401 from an in-flight request must adopt the winner, not refresh again.
        assert await sources[0].token(client, failed_token="old-fake") == "rotated-fake"
    assert tokens == ["rotated-fake", "rotated-fake"]
    assert len(requests) == 1
    saved = json.loads(path.read_text())
    assert saved["claudeAiOauth"]["refreshToken"] == "next-fake"
    assert saved["claudeAiOauth"]["scopes"] == ["user:inference"]
    assert saved["claudeAiOauth"]["subscriptionType"] == "max"
    assert saved["other"] == "preserved"
    assert os.stat(path).st_mode & 0o777 == 0o600


@pytest.mark.asyncio
async def test_failed_refresh_is_redacted_and_not_replayed(tmp_path):
    from src.claude_subscription import SubscriptionError
    from src.claude_subscription_credentials import FileCredentials

    path = credential_file(tmp_path)

    async def respond(reader, writer, requests):
        await reply(
            writer,
            {"error": "invalid_grant", "error_description": "refresh-fake old-fake"},
            status=400,
        )

    async with (
        upstream(respond) as (url, requests),
        httpx.AsyncClient(trust_env=False) as client,
    ):
        for _ in range(2):
            with pytest.raises(SubscriptionError) as exc:
                await FileCredentials(path, token_url=url).token(client)
            assert "refresh-fake" not in str(exc.value)
            assert "old-fake" not in str(exc.value)
    assert len(requests) == 1


@pytest.mark.asyncio
async def test_opt_in_does_not_read_ambient_credentials(monkeypatch, tmp_path):
    from src.claude_subscription import SubscriptionError
    from src.claude_subscription_credentials import operator_source

    monkeypatch.delenv("ODYSSEUS_CLAUDE_SUBSCRIPTION_TOKEN", raising=False)
    monkeypatch.delenv("ODYSSEUS_CLAUDE_SUBSCRIPTION_CREDENTIALS_FILE", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "unrelated-api-key")
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "unrelated-oauth")
    with pytest.raises(SubscriptionError):
        operator_source()
    assert list(tmp_path.iterdir()) == []
    monkeypatch.setenv("ODYSSEUS_CLAUDE_SUBSCRIPTION_TOKEN", "explicit-fake")
    async with httpx.AsyncClient() as client:
        assert await operator_source().token(client) == "explicit-fake"


@pytest.mark.asyncio
async def test_refresh_success_with_failed_commit_requires_reauth(
    tmp_path, monkeypatch
):
    from src import claude_subscription_credentials as credentials
    from src.claude_subscription import SubscriptionError

    path = credential_file(tmp_path)
    write = credentials._private_write

    def fail_commit(target, payload):
        if target == path:
            raise OSError("secret fixture must not escape")
        return write(target, payload)

    monkeypatch.setattr(credentials, "_private_write", fail_commit)

    async def respond(reader, writer, requests):
        await reply(
            writer,
            {
                "access_token": "rotated-fake",
                "refresh_token": "next-fake",
                "expires_in": 3600,
            },
        )

    async with upstream(respond) as (url, requests), httpx.AsyncClient() as client:
        for _ in range(2):
            with pytest.raises(SubscriptionError) as exc:
                await credentials.FileCredentials(path, token_url=url).token(client)
            assert "secret fixture" not in str(exc.value)
    assert len(requests) == 1
    assert json.loads(path.read_text())["claudeAiOauth"]["accessToken"] == "old-fake"


@pytest.mark.asyncio
async def test_refresh_serializes_across_independent_processes(tmp_path):
    import sys

    path = credential_file(tmp_path)
    script = tmp_path / "refresh_probe.py"
    script.write_text("""
import asyncio, httpx, sys
from src.claude_subscription_credentials import FileCredentials
async def main():
    async with httpx.AsyncClient(trust_env=False) as client:
        token = await FileCredentials(sys.argv[1], token_url=sys.argv[2]).token(client)
        assert token == 'rotated-fake'
    print('ok')
asyncio.run(main())
""")

    async def respond(reader, writer, requests):
        await reply(
            writer,
            {
                "access_token": "rotated-fake",
                "refresh_token": "next-fake",
                "expires_in": 3600,
            },
        )

    async with upstream(respond) as (url, requests):
        from pathlib import Path

        env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[1]))
        procs = [
            await asyncio.create_subprocess_exec(
                sys.executable,
                str(script),
                str(path),
                url,
                env=env,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            for _ in range(2)
        ]
        results = await asyncio.wait_for(
            asyncio.gather(*(p.communicate() for p in procs)), 15
        )
    assert [p.returncode for p in procs] == [0, 0], results
    assert [stdout.strip() for stdout, _ in results] == [b"ok", b"ok"]
    assert len(requests) == 1
