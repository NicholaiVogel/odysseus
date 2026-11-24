"""Compatibility version detection: no credentials or live network."""

import subprocess
from types import SimpleNamespace

import pytest

from src import claude_subscription as provider


@pytest.fixture(autouse=True)
def clear_version_cache():
    provider._installed_cli_version.cache_clear()
    yield
    provider._installed_cli_version.cache_clear()


def test_headers_detect_and_cache_installed_version(monkeypatch):
    monkeypatch.delenv("ODYSSEUS_CLAUDE_SUBSCRIPTION_CLI_VERSION", raising=False)
    monkeypatch.setattr("shutil.which", lambda command: "/fixture/claude")
    calls = []

    def version(command, **kwargs):
        calls.append(command)
        assert kwargs["timeout"] == 5
        return SimpleNamespace(stdout="2.1.327 (Claude Code)\n")

    monkeypatch.setattr(subprocess, "run", version)
    for _ in range(2):
        assert (
            provider.request_headers("fixture")["user-agent"]
            == "claude-code/2.1.327 (external, cli)"
        )
    assert calls == [["/fixture/claude", "--version"]]


def test_explicit_version_needs_no_cli(monkeypatch):
    monkeypatch.setenv("ODYSSEUS_CLAUDE_SUBSCRIPTION_CLI_VERSION", "2.1.327")
    monkeypatch.setattr(
        provider,
        "_installed_cli_version",
        lambda: pytest.fail("explicit version should not probe CLI"),
    )
    assert "2.1.327" in provider.request_headers("fixture")["user-agent"]


def test_missing_cli_reports_configuration_error_not_stale_fallback(monkeypatch):
    monkeypatch.delenv("ODYSSEUS_CLAUDE_SUBSCRIPTION_CLI_VERSION", raising=False)
    monkeypatch.setattr("shutil.which", lambda command: None)
    monkeypatch.setattr("pathlib.Path.is_file", lambda path: False)
    with pytest.raises(provider.SubscriptionError) as exc:
        provider.request_headers("fixture")
    assert exc.value.status == 503
    assert "ODYSSEUS_CLAUDE_SUBSCRIPTION_CLI_VERSION" in exc.value.message
