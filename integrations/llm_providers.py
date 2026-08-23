#!/usr/bin/env python3
"""
AI Dev Team — LLM Provider ABC
Abstract interface for the coding CLIs that back an agent.

Each agent may run on a different provider, so a team can mix
subscriptions (Claude Max, ChatGPT Plus/Pro, MiniMax Coding Plan)
instead of spending one rate limit on all nine agents.

Contract: every provider takes the prompt on stdin and returns the
agent's final message as plain text — the same shape the orchestrator
already expected from `claude --print`.
"""

import os
import logging
import subprocess
import tempfile
from abc import ABC, abstractmethod
from pathlib import Path

logger = logging.getLogger(__name__)


# =============================================================================
# Constants
# =============================================================================
DEFAULT_PROVIDER = "claude"
DEFAULT_TIMEOUT_SECONDS = 900

# Tools every agent needs to do its job
AGENT_TOOLS = "Bash,Edit,Write,Read,Glob,Grep"

# Vars that would silently redirect the `claude` binary at a third-party
# endpoint. Scrubbed for the native Claude provider so that MiniMax config
# living in the same shell cannot hijack Claude agents.
# ANTHROPIC_API_KEY is deliberately NOT scrubbed — it is legitimate
# Anthropic-direct billing and does not change the endpoint.
ANTHROPIC_REDIRECT_VARS = (
    "ANTHROPIC_BASE_URL",
    "ANTHROPIC_AUTH_TOKEN",
    "ANTHROPIC_MODEL",
    "ANTHROPIC_SMALL_FAST_MODEL",
    "ANTHROPIC_DEFAULT_SONNET_MODEL",
    "ANTHROPIC_DEFAULT_OPUS_MODEL",
    "ANTHROPIC_DEFAULT_HAIKU_MODEL",
)


# =============================================================================
# ABC
# =============================================================================
class LLMProvider(ABC):
    """A coding CLI that can run an agent turn non-interactively."""

    name: str = "base"
    binary: str = ""

    # ---- Subclass hooks ----------------------------------------------------
    @abstractmethod
    def build_cmd(self, out_file: Path | None) -> list[str]:
        """argv for one non-interactive turn. Prompt arrives on stdin."""

    def build_env(self) -> dict[str, str]:
        """Environment for the child process."""
        env = os.environ.copy()
        # Otherwise nested-session guards refuse to run under an agent
        env.pop("CLAUDECODE", None)
        return env

    @property
    def writes_output_file(self) -> bool:
        """True if the final message lands in a file instead of stdout."""
        return False

    @property
    def timeout_seconds(self) -> int:
        return _env_int("AGENT_TIMEOUT_SECONDS", DEFAULT_TIMEOUT_SECONDS)

    @property
    def enabled(self) -> bool:
        """True if this provider is usable (binary present, creds set)."""
        return _which(self.binary) is not None

    def unavailable_reason(self) -> str:
        return f"'{self.binary}' CLI not found in PATH"

    # ---- Shared driver ----------------------------------------------------
    def run(self, prompt: str, cwd: Path, verbose: bool = False) -> str:
        """
        Execute one agent turn. Returns the response text, or a string
        starting with "ERROR:" so the orchestrator can keep going.
        """
        if not self.enabled:
            logger.error("Provider %s unavailable: %s", self.name, self.unavailable_reason())
            return f"ERROR: {self.unavailable_reason()}"

        out_file: Path | None = None
        tmp_dir: tempfile.TemporaryDirectory | None = None
        if self.writes_output_file:
            tmp_dir = tempfile.TemporaryDirectory(prefix="ai-team-")
            out_file = Path(tmp_dir.name) / "last_message.txt"

        try:
            cmd = self.build_cmd(out_file)
            if verbose:
                print(f"    Calling {self.name}: {' '.join(cmd)}")

            result = subprocess.run(
                cmd,
                input=prompt,
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
                cwd=str(cwd),
                env=self.build_env(),
            )

            response = ""
            if out_file is not None and out_file.exists():
                response = out_file.read_text(errors="replace").strip()
            if not response:
                response = result.stdout

            if result.returncode != 0 and not response.strip():
                logger.error("%s error: %s", self.name, result.stderr)
                return f"ERROR: {result.stderr}"
            if result.returncode != 0:
                logger.warning("%s exited %d but produced output", self.name, result.returncode)

            return response

        except subprocess.TimeoutExpired:
            logger.error("%s timeout (%d s)", self.name, self.timeout_seconds)
            return "ERROR: Timeout"
        except FileNotFoundError:
            logger.error("%s: %s", self.name, self.unavailable_reason())
            return f"ERROR: {self.unavailable_reason()}"
        except Exception as e:
            logger.error("%s error: %s", self.name, e)
            return f"ERROR: {e}"
        finally:
            if tmp_dir is not None:
                tmp_dir.cleanup()


# =============================================================================
# Claude Code CLI — Anthropic Pro/Max subscription (default)
# =============================================================================
class ClaudeProvider(LLMProvider):
    name = "claude"
    binary = "claude"

    def build_cmd(self, out_file: Path | None) -> list[str]:
        cmd = ["claude", "--print", "--allowedTools", AGENT_TOOLS]
        model = os.getenv("CLAUDE_MODEL", "").strip()
        if model:
            cmd += ["--model", model]
        return cmd

    def build_env(self) -> dict[str, str]:
        env = super().build_env()
        # Keep third-party routing out of native Claude agents
        for var in ANTHROPIC_REDIRECT_VARS:
            env.pop(var, None)
        return env


# =============================================================================
# MiniMax Coding Plan — Anthropic-compatible endpoint, reuses `claude`
# =============================================================================
class MiniMaxProvider(ClaudeProvider):
    """
    MiniMax exposes an Anthropic-compatible API, so the same `claude`
    binary drives it — only the endpoint, token and model name change.
    Credentials use MINIMAX_* names so they cannot leak into Claude agents.
    """

    name = "minimax"
    binary = "claude"

    DEFAULT_BASE_URL = "https://api.minimax.io/anthropic"
    DEFAULT_MODEL = "MiniMax-M3[1m]"

    @property
    def api_key(self) -> str:
        return os.getenv("MINIMAX_API_KEY", "").strip()

    @property
    def base_url(self) -> str:
        return os.getenv("MINIMAX_BASE_URL", "").strip() or self.DEFAULT_BASE_URL

    @property
    def model(self) -> str:
        return os.getenv("MINIMAX_MODEL", "").strip() or self.DEFAULT_MODEL

    @property
    def enabled(self) -> bool:
        return bool(self.api_key) and _which(self.binary) is not None

    def unavailable_reason(self) -> str:
        if not self.api_key:
            return "MINIMAX_API_KEY is not set (MiniMax Coding Plan key)"
        return super().unavailable_reason()

    def build_cmd(self, out_file: Path | None) -> list[str]:
        # Model comes from env, not --model, so every tier maps to MiniMax
        return ["claude", "--print", "--allowedTools", AGENT_TOOLS]

    def build_env(self) -> dict[str, str]:
        env = super().build_env()          # scrubs inherited redirect vars
        env.pop("ANTHROPIC_API_KEY", None)  # would outrank the MiniMax token
        env["ANTHROPIC_BASE_URL"] = self.base_url
        env["ANTHROPIC_AUTH_TOKEN"] = self.api_key
        # Map every tier so subagent/summary calls stay on MiniMax
        for var in ("ANTHROPIC_MODEL", "ANTHROPIC_SMALL_FAST_MODEL",
                    "ANTHROPIC_DEFAULT_SONNET_MODEL",
                    "ANTHROPIC_DEFAULT_OPUS_MODEL",
                    "ANTHROPIC_DEFAULT_HAIKU_MODEL"):
            env[var] = self.model
        # Third-party keys cannot serve Anthropic telemetry endpoints
        env["CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC"] = "1"
        return env


# =============================================================================
# OpenAI Codex CLI — ChatGPT Plus/Pro subscription (not API billing)
# =============================================================================
class CodexProvider(LLMProvider):
    """
    Uses `codex exec` with the credentials saved by `codex login`, i.e. the
    ChatGPT subscription. The final message is captured via
    --output-last-message rather than stdout, which carries the event log.
    """

    name = "codex"
    binary = "codex"

    DEFAULT_SANDBOX = "workspace-write"
    AUTH_FILE = Path.home() / ".codex" / "auth.json"

    @property
    def sandbox(self) -> str:
        return os.getenv("CODEX_SANDBOX", "").strip() or self.DEFAULT_SANDBOX

    @property
    def writes_output_file(self) -> bool:
        return True

    @property
    def enabled(self) -> bool:
        return _which(self.binary) is not None and self.AUTH_FILE.exists()

    def unavailable_reason(self) -> str:
        if _which(self.binary) is None:
            return "'codex' CLI not found in PATH (npm install -g @openai/codex)"
        return "codex is not logged in — run: codex login"

    def build_cmd(self, out_file: Path | None) -> list[str]:
        cmd = [
            "codex", "exec",
            "--sandbox", self.sandbox,
            "--skip-git-repo-check",
            "--color", "never",
        ]
        model = os.getenv("CODEX_MODEL", "").strip()
        if model:
            cmd += ["--model", model]
        if out_file is not None:
            cmd += ["--output-last-message", str(out_file)]
        # "-" = read the whole prompt from stdin
        cmd.append("-")
        return cmd

    def build_env(self) -> dict[str, str]:
        env = super().build_env()
        # Force subscription auth: a stray API key would switch to usage billing
        if os.getenv("CODEX_PREFER_API_KEY", "").strip().lower() not in ("1", "true", "yes"):
            env.pop("OPENAI_API_KEY", None)
        return env


# =============================================================================
# Registry
# =============================================================================
PROVIDERS: dict[str, type[LLMProvider]] = {
    "claude":  ClaudeProvider,
    "minimax": MiniMaxProvider,
    "codex":   CodexProvider,
}


def get_llm_provider(name: str | None) -> LLMProvider:
    """Instantiate a provider by name, falling back to Claude."""
    key = (name or DEFAULT_PROVIDER).strip().lower()
    cls = PROVIDERS.get(key)
    if cls is None:
        logger.warning("Unknown LLM provider '%s', falling back to '%s'. Known: %s",
                       name, DEFAULT_PROVIDER, ", ".join(sorted(PROVIDERS)))
        cls = PROVIDERS[DEFAULT_PROVIDER]
    return cls()


def resolve_agent_provider(agent_id: str, agents_config: dict) -> LLMProvider:
    """
    Resolve which provider runs a given agent.

    Precedence:
    1. env AGENT_PROVIDER_<AGENT_ID>   (e.g. AGENT_PROVIDER_DEV2=codex)
    2. agents.yaml  agents.<id>.provider
    3. agents.yaml  global.default_provider
    4. "claude"
    """
    override = os.getenv(f"AGENT_PROVIDER_{agent_id.upper()}", "").strip()
    if override:
        return get_llm_provider(override)

    agent = agents_config.get("agents", {}).get(agent_id, {}) or {}
    if agent.get("provider"):
        return get_llm_provider(str(agent["provider"]))

    default = agents_config.get("global", {}).get("default_provider")
    return get_llm_provider(str(default) if default else None)


# =============================================================================
# Helpers
# =============================================================================
def _which(binary: str) -> str | None:
    from shutil import which
    return which(binary) if binary else None


def _env_int(name: str, default: int) -> int:
    try:
        value = int(os.getenv(name, "").strip())
        return value if value > 0 else default
    except (TypeError, ValueError):
        return default
