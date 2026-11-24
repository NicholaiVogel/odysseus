"""Live model catalog contracts over local HTTP, never real credentials."""

import pytest

from src import claude_subscription as provider
from src.claude_subscription_credentials import StaticToken
from tests.test_claude_subscription_http import reply, upstream


@pytest.mark.asyncio
async def test_catalog_reads_every_page_and_deduplicates():
    async def respond(reader, writer, requests):
        if len(requests) == 1:
            await reply(
                writer,
                {
                    "data": [{"id": "claude-sonnet-current"}],
                    "has_more": True,
                    "last_id": "claude-sonnet-current",
                },
            )
        else:
            await reply(
                writer,
                {
                    "data": [
                        {"id": "claude-sonnet-current"},
                        {"id": "claude-opus-current"},
                    ],
                    "has_more": False,
                },
            )

    async with upstream(respond) as (url, requests):
        result = await provider.Transport(StaticToken("fixture")).list_models(
            models_url=url + "/v1/models"
        )
    assert result == ["claude-sonnet-current", "claude-opus-current"]
    assert len(requests) == 2
    assert "after_id=claude-sonnet-current" in requests[1][0]
    assert all(request[1]["authorization"] == "Bearer fixture" for request in requests)
    assert all(
        request[0].startswith("GET ") and request[2] == b"" for request in requests
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "body",
    [
        {"data": [], "has_more": False},
        {"data": [{"id": "current"}], "has_more": True, "last_id": "same"},
    ],
)
async def test_catalog_does_not_return_stale_fallback_on_bad_pagination(body):
    async def respond(reader, writer, requests):
        await reply(writer, body)

    async with upstream(respond) as (url, requests):
        with pytest.raises(provider.SubscriptionError) as error:
            await provider.Transport(StaticToken("fixture")).list_models(models_url=url)
    assert error.value.status == 502
    assert len(requests) <= 2


@pytest.mark.asyncio
async def test_connect_probe_uses_discovered_catalog(monkeypatch):
    monkeypatch.setenv("ODYSSEUS_CLAUDE_SUBSCRIPTION_TOKEN", "fixture")
    monkeypatch.delenv("ODYSSEUS_CLAUDE_SUBSCRIPTION_CREDENTIALS_FILE", raising=False)

    async def respond(reader, writer, requests):
        await reply(
            writer,
            {"data": [{"id": "claude-current-from-upstream"}], "has_more": False},
        )

    async with upstream(respond) as (url, requests):
        monkeypatch.setattr(provider, "MODELS_URL", url)
        info = await provider.probe()
    assert info["models"] == ["claude-current-from-upstream"]
    assert info["upstream_verified"] is True
    assert len(requests) == 1
