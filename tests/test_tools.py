"""Tests for the read-only vault tools against a fixture vault."""

from __future__ import annotations

from pathlib import Path

import pytest

from obsidian_agent.tools import build_tools
from obsidian_agent.vault import Vault, VaultError

VAULT_DIR = Path(__file__).parent / "fixtures" / "vault"


@pytest.fixture
def tools() -> dict:
    """Return the agent tools keyed by name, bound to the fixture vault."""
    return {t.name: t for t in build_tools(Vault(VAULT_DIR))}


def call(tool, **kwargs) -> str:
    """Invoke a LangChain tool with keyword args and return its string output."""
    return tool.invoke(kwargs)


def test_list_dir_root(tools):
    out = call(tools["list_dir"], subpath="")
    assert "Operations/" in out
    assert "Compliance Checklist.md" in out


def test_list_dir_ignores_dotfolders(tools):
    out = call(tools["list_dir"], subpath="")
    assert ".obsidian" not in out


def test_glob_notes(tools):
    out = call(tools["glob_notes"], pattern="**/Vendor*.md")
    assert "Operations/Vendor-Onboarding.md" in out


def test_search_notes_substring(tools):
    out = call(tools["search_notes"], query="compliance check")
    assert "Vendor-Onboarding.md" in out
    assert ":" in out  # path:line: text format


def test_search_notes_no_match(tools):
    out = call(tools["search_notes"], query="nonexistent-term-xyz")
    assert out.startswith("(no matches")


def test_read_note_by_name(tools):
    out = call(tools["read_note"], path="Vendor-Onboarding")
    assert out.startswith("# Operations/Vendor-Onboarding.md")
    assert "master service agreement" in out
    assert "   1 |" in out  # line numbering


def test_read_note_not_found(tools):
    out = call(tools["read_note"], path="Does Not Exist")
    assert out.startswith("ERROR")


def test_get_backlinks(tools):
    out = call(tools["get_backlinks"], note="Compliance Checklist")
    assert "Operations/Vendor-Onboarding.md" in out


def test_search_by_tag(tools):
    out = call(tools["search_by_tag"], tag="compliance")
    assert "Compliance Checklist.md" in out


def test_search_by_tag_with_hash(tools):
    out = call(tools["search_by_tag"], tag="#operations")
    assert "Vendor-Onboarding.md" in out


def test_read_note_path_escape_blocked(tools):
    # find_note resolves safely; an escaping path simply isn't found.
    out = call(tools["read_note"], path="../../../../etc/passwd")
    assert out.startswith("ERROR")


def test_vault_safe_resolve_rejects_escape():
    vault = Vault(VAULT_DIR)
    with pytest.raises(VaultError):
        vault.safe_resolve("../secrets.md")
