"""Typed configuration loaded from environment / .env file."""

from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Agent configuration.

    Values come from environment variables or a local ``.env`` file. CLI flags
    can override individual fields at runtime (see ``cli.py``).
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- LLM endpoint (OpenAI-compatible) ---
    openai_base_url: str = Field(
        default="http://localhost:8000/v1",
        description="Base URL of the OpenAI-compatible endpoint (must end in /v1).",
    )
    openai_api_key: str = Field(
        default="not-needed",
        description="API key; any non-empty string if the endpoint ignores it.",
    )
    model: str = Field(
        default="gpt-oss",
        alias="OBSIDIAN_MODEL",
        description="Model name served by the endpoint.",
    )
    temperature: float = Field(
        default=0.0,
        alias="AGENT_TEMPERATURE",
        description="Sampling temperature for the agent LLM.",
    )

    # --- Vault ---
    vault_path: Path = Field(
        default=Path("."),
        alias="OBSIDIAN_VAULT_PATH",
        description="Absolute path to the Obsidian vault root.",
    )

    # --- Agent loop ---
    max_iterations: int = Field(
        default=12,
        alias="AGENT_MAX_ITERATIONS",
        description="Recursion limit for the agent->tools->agent loop.",
    )

    def resolved_vault(self) -> Path:
        """Return the vault path expanded and resolved to an absolute path."""
        return self.vault_path.expanduser().resolve()


def load_settings(**overrides: object) -> Settings:
    """Load settings, applying any non-None CLI overrides on top of env/.env."""
    clean = {k: v for k, v in overrides.items() if v is not None}
    return Settings(**clean)
