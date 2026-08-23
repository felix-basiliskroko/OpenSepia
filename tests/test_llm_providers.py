"""Tests for LLM provider adapters and per-agent provider resolution."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from integrations.llm_providers import (
    ClaudeProvider,
    CodexProvider,
    MiniMaxProvider,
    get_llm_provider,
    resolve_agent_provider,
)


AGENTS_CONFIG = {
    "global": {"default_provider": "claude"},
    "agents": {
        "po": {"name": "Product Owner"},
        "dev1": {"name": "Developer 1", "provider": "claude"},
        "dev2": {"name": "Developer 2", "provider": "codex"},
        "tester": {"name": "QA Engineer", "provider": "minimax"},
    },
}


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

def test_get_llm_provider_by_name(clean_llm_env):
    assert isinstance(get_llm_provider("claude"), ClaudeProvider)
    assert isinstance(get_llm_provider("codex"), CodexProvider)
    assert isinstance(get_llm_provider("minimax"), MiniMaxProvider)


def test_get_llm_provider_is_case_insensitive(clean_llm_env):
    assert get_llm_provider("  CoDeX ").name == "codex"


def test_get_llm_provider_unknown_falls_back_to_claude(clean_llm_env):
    assert get_llm_provider("gemini").name == "claude"


def test_get_llm_provider_none_falls_back_to_claude(clean_llm_env):
    assert get_llm_provider(None).name == "claude"


# ---------------------------------------------------------------------------
# Per-agent resolution
# ---------------------------------------------------------------------------

def test_resolve_uses_agent_provider_key(clean_llm_env):
    assert resolve_agent_provider("dev2", AGENTS_CONFIG).name == "codex"
    assert resolve_agent_provider("tester", AGENTS_CONFIG).name == "minimax"


def test_resolve_falls_back_to_global_default(clean_llm_env):
    # po has no explicit provider
    assert resolve_agent_provider("po", AGENTS_CONFIG).name == "claude"


def test_resolve_falls_back_to_claude_without_global_default(clean_llm_env):
    config = {"global": {}, "agents": {"po": {}}}
    assert resolve_agent_provider("po", config).name == "claude"


def test_resolve_global_default_applies_to_unset_agents(clean_llm_env):
    config = {"global": {"default_provider": "minimax"}, "agents": {"po": {}}}
    assert resolve_agent_provider("po", config).name == "minimax"


def test_resolve_env_override_beats_yaml(clean_llm_env, monkeypatch):
    monkeypatch.setenv("AGENT_PROVIDER_DEV2", "claude")
    assert resolve_agent_provider("dev2", AGENTS_CONFIG).name == "claude"


def test_resolve_env_override_is_per_agent(clean_llm_env, monkeypatch):
    monkeypatch.setenv("AGENT_PROVIDER_DEV1", "codex")
    assert resolve_agent_provider("dev1", AGENTS_CONFIG).name == "codex"
    assert resolve_agent_provider("po", AGENTS_CONFIG).name == "claude"


def test_resolve_unknown_agent_uses_default(clean_llm_env):
    assert resolve_agent_provider("nobody", AGENTS_CONFIG).name == "claude"


# ---------------------------------------------------------------------------
# Claude provider
# ---------------------------------------------------------------------------

def test_claude_cmd_is_non_interactive(clean_llm_env):
    cmd = ClaudeProvider().build_cmd(None)
    assert cmd[0] == "claude"
    assert "--print" in cmd
    assert "--allowedTools" in cmd


def test_claude_cmd_pins_model_when_set(clean_llm_env, monkeypatch):
    monkeypatch.setenv("CLAUDE_MODEL", "opus")
    cmd = ClaudeProvider().build_cmd(None)
    assert cmd[cmd.index("--model") + 1] == "opus"


def test_claude_env_scrubs_third_party_redirect(clean_llm_env, monkeypatch):
    """A MiniMax endpoint in the shell must not hijack Claude agents."""
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://api.minimax.io/anthropic")
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "minimax-key")
    monkeypatch.setenv("ANTHROPIC_MODEL", "MiniMax-M3[1m]")
    env = ClaudeProvider().build_env()
    assert "ANTHROPIC_BASE_URL" not in env
    assert "ANTHROPIC_AUTH_TOKEN" not in env
    assert "ANTHROPIC_MODEL" not in env


def test_claude_env_keeps_anthropic_api_key(clean_llm_env, monkeypatch):
    """Anthropic-direct billing is legitimate and must survive scrubbing."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-real")
    assert ClaudeProvider().build_env()["ANTHROPIC_API_KEY"] == "sk-ant-real"


def test_claude_env_unsets_nested_session_guard(clean_llm_env, monkeypatch):
    monkeypatch.setenv("CLAUDECODE", "1")
    assert "CLAUDECODE" not in ClaudeProvider().build_env()


def test_claude_reads_stdout_not_a_file(clean_llm_env):
    assert ClaudeProvider().writes_output_file is False


# ---------------------------------------------------------------------------
# MiniMax provider
# ---------------------------------------------------------------------------

def test_minimax_reuses_claude_binary(clean_llm_env):
    assert MiniMaxProvider().build_cmd(None)[0] == "claude"


def test_minimax_requires_api_key(clean_llm_env):
    provider = MiniMaxProvider()
    assert provider.enabled is False
    assert "MINIMAX_API_KEY" in provider.unavailable_reason()


def test_minimax_env_maps_endpoint_and_token(clean_llm_env, monkeypatch):
    monkeypatch.setenv("MINIMAX_API_KEY", "mm-coding-plan-key")
    env = MiniMaxProvider().build_env()
    assert env["ANTHROPIC_BASE_URL"] == "https://api.minimax.io/anthropic"
    assert env["ANTHROPIC_AUTH_TOKEN"] == "mm-coding-plan-key"
    assert env["CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC"] == "1"


def test_minimax_env_maps_every_model_tier(clean_llm_env, monkeypatch):
    monkeypatch.setenv("MINIMAX_API_KEY", "mm-key")
    monkeypatch.setenv("MINIMAX_MODEL", "MiniMax-M2")
    env = MiniMaxProvider().build_env()
    for var in ("ANTHROPIC_MODEL", "ANTHROPIC_SMALL_FAST_MODEL",
                "ANTHROPIC_DEFAULT_SONNET_MODEL", "ANTHROPIC_DEFAULT_OPUS_MODEL",
                "ANTHROPIC_DEFAULT_HAIKU_MODEL"):
        assert env[var] == "MiniMax-M2"


def test_minimax_env_drops_anthropic_api_key(clean_llm_env, monkeypatch):
    """ANTHROPIC_API_KEY would outrank the MiniMax token."""
    monkeypatch.setenv("MINIMAX_API_KEY", "mm-key")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-real")
    assert "ANTHROPIC_API_KEY" not in MiniMaxProvider().build_env()


def test_minimax_honours_china_base_url(clean_llm_env, monkeypatch):
    monkeypatch.setenv("MINIMAX_API_KEY", "mm-key")
    monkeypatch.setenv("MINIMAX_BASE_URL", "https://api.minimaxi.com/anthropic")
    env = MiniMaxProvider().build_env()
    assert env["ANTHROPIC_BASE_URL"] == "https://api.minimaxi.com/anthropic"


# ---------------------------------------------------------------------------
# Codex provider
# ---------------------------------------------------------------------------

def test_codex_cmd_is_non_interactive_exec(clean_llm_env):
    cmd = CodexProvider().build_cmd(Path("/tmp/out.txt"))
    assert cmd[:2] == ["codex", "exec"]
    # "-" sentinel: read the whole prompt from stdin
    assert cmd[-1] == "-"


def test_codex_cmd_captures_final_message_to_file(clean_llm_env):
    out = Path("/tmp/out.txt")
    cmd = CodexProvider().build_cmd(out)
    assert cmd[cmd.index("--output-last-message") + 1] == str(out)
    assert CodexProvider().writes_output_file is True


def test_codex_cmd_default_sandbox_allows_workspace_writes(clean_llm_env):
    cmd = CodexProvider().build_cmd(None)
    assert cmd[cmd.index("--sandbox") + 1] == "workspace-write"


def test_codex_cmd_sandbox_from_env(clean_llm_env, monkeypatch):
    monkeypatch.setenv("CODEX_SANDBOX", "danger-full-access")
    cmd = CodexProvider().build_cmd(None)
    assert cmd[cmd.index("--sandbox") + 1] == "danger-full-access"


def test_codex_cmd_pins_model_when_set(clean_llm_env, monkeypatch):
    monkeypatch.setenv("CODEX_MODEL", "gpt-5-codex")
    cmd = CodexProvider().build_cmd(None)
    assert cmd[cmd.index("--model") + 1] == "gpt-5-codex"


def test_codex_env_drops_api_key_to_force_subscription(clean_llm_env, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai")
    assert "OPENAI_API_KEY" not in CodexProvider().build_env()


def test_codex_env_keeps_api_key_when_opted_in(clean_llm_env, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai")
    monkeypatch.setenv("CODEX_PREFER_API_KEY", "true")
    assert CodexProvider().build_env()["OPENAI_API_KEY"] == "sk-openai"


def test_codex_unavailable_reason_mentions_login(clean_llm_env, monkeypatch):
    monkeypatch.setattr(CodexProvider, "AUTH_FILE", Path("/nonexistent/auth.json"))
    provider = CodexProvider()
    assert provider.enabled is False
    assert "codex login" in provider.unavailable_reason()


# ---------------------------------------------------------------------------
# Shared driver
# ---------------------------------------------------------------------------

def test_run_returns_error_string_when_provider_unavailable(clean_llm_env):
    """Orchestrator must keep going, so failures come back as ERROR: text."""
    provider = MiniMaxProvider()  # no MINIMAX_API_KEY
    result = provider.run("prompt", cwd=Path("."))
    assert result.startswith("ERROR:")
    assert "MINIMAX_API_KEY" in result


def test_run_sends_prompt_on_stdin_and_returns_stdout(clean_llm_env, monkeypatch):
    captured = {}

    class FakeResult:
        returncode = 0
        stdout = "agent response"
        stderr = ""

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        captured["input"] = kwargs["input"]
        captured["cwd"] = kwargs["cwd"]
        return FakeResult()

    monkeypatch.setattr("integrations.llm_providers.subprocess.run", fake_run)
    monkeypatch.setattr(ClaudeProvider, "enabled", True)

    out = ClaudeProvider().run("my prompt", cwd=Path("/work"))
    assert out == "agent response"
    assert captured["input"] == "my prompt"
    assert captured["cwd"] == "/work"
    assert captured["cmd"][0] == "claude"


def test_run_prefers_output_file_over_stdout(clean_llm_env, monkeypatch):
    """Codex stdout carries the event log; the final message is in the file."""

    class FakeResult:
        returncode = 0
        stdout = "[event log noise]"
        stderr = ""

    def fake_run(cmd, **kwargs):
        out_path = Path(cmd[cmd.index("--output-last-message") + 1])
        out_path.write_text("final message")
        return FakeResult()

    monkeypatch.setattr("integrations.llm_providers.subprocess.run", fake_run)
    monkeypatch.setattr(CodexProvider, "enabled", True)

    assert CodexProvider().run("prompt", cwd=Path(".")) == "final message"


def test_run_reports_timeout_as_error(clean_llm_env, monkeypatch):
    import subprocess as sp

    def fake_run(cmd, **kwargs):
        raise sp.TimeoutExpired(cmd, 900)

    monkeypatch.setattr("integrations.llm_providers.subprocess.run", fake_run)
    monkeypatch.setattr(ClaudeProvider, "enabled", True)
    assert ClaudeProvider().run("prompt", cwd=Path(".")) == "ERROR: Timeout"


def test_run_reports_nonzero_exit_without_output_as_error(clean_llm_env, monkeypatch):
    class FakeResult:
        returncode = 1
        stdout = ""
        stderr = "boom"

    monkeypatch.setattr("integrations.llm_providers.subprocess.run",
                        lambda cmd, **kw: FakeResult())
    monkeypatch.setattr(ClaudeProvider, "enabled", True)
    assert ClaudeProvider().run("prompt", cwd=Path(".")) == "ERROR: boom"


def test_run_keeps_output_from_nonzero_exit(clean_llm_env, monkeypatch):
    """Partial work is still worth parsing; don't discard it."""

    class FakeResult:
        returncode = 1
        stdout = "---FILES---\npath: board/sprint.md\n"
        stderr = "warning"

    monkeypatch.setattr("integrations.llm_providers.subprocess.run",
                        lambda cmd, **kw: FakeResult())
    monkeypatch.setattr(ClaudeProvider, "enabled", True)
    assert "---FILES---" in ClaudeProvider().run("prompt", cwd=Path("."))


def test_timeout_seconds_from_env(clean_llm_env, monkeypatch):
    monkeypatch.setenv("AGENT_TIMEOUT_SECONDS", "1800")
    assert ClaudeProvider().timeout_seconds == 1800


def test_timeout_seconds_invalid_falls_back_to_default(clean_llm_env, monkeypatch):
    monkeypatch.setenv("AGENT_TIMEOUT_SECONDS", "not_a_number")
    assert ClaudeProvider().timeout_seconds == 900
