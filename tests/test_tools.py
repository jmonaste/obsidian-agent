"""Tests for the read-only vault tools against a fixture vault."""

from __future__ import annotations

from pathlib import Path

import pytest

from obsidian_agent.tools import build_tools
from obsidian_agent.vault import Vault, VaultError, parse_tags

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


def test_search_notes_footer_reports_counts(tools):
    out = call(tools["search_notes"], query="compliance")
    assert "across" in out and "notes)" in out


def _repeated_vault(tmp_path) -> Vault:
    # Two notes; the first has many matches, the second has one.
    (tmp_path / "A.md").write_text("\n".join(["needle"] * 10), encoding="utf-8")
    (tmp_path / "B.md").write_text("a needle here", encoding="utf-8")
    return Vault(tmp_path)


def test_search_notes_caps_per_file(tmp_path):
    tools = {t.name: t for t in build_tools(_repeated_vault(tmp_path))}
    out = call(tools["search_notes"], query="needle", max_per_file=3)
    # At most 3 hits from A.md even though it has 10 matching lines.
    assert out.count("A.md:") == 3


def test_search_notes_spans_multiple_files(tmp_path):
    tools = {t.name: t for t in build_tools(_repeated_vault(tmp_path))}
    out = call(tools["search_notes"], query="needle", max_per_file=3)
    # The per-file cap means B.md still surfaces (no alphabetical starvation).
    assert "B.md:" in out
    assert "across 2 notes" in out


def test_read_note_by_name(tools):
    out = call(tools["read_note"], path="Vendor-Onboarding")
    assert out.startswith("# Operations/Vendor-Onboarding.md")
    assert "master service agreement" in out
    assert "   1 |" in out  # line numbering


def test_read_note_not_found(tools):
    out = call(tools["read_note"], path="Does Not Exist")
    assert out.startswith("ERROR")


def test_read_note_default_full_for_small_note(tools):
    # The fixture note fits in one window, so there is no paging footer.
    out = call(tools["read_note"], path="Vendor-Onboarding")
    assert "[more:" not in out


def _long_vault(tmp_path) -> Vault:
    body = "\n".join(f"line {i}" for i in range(1, 31))
    (tmp_path / "Long.md").write_text(f"# Long\n{body}\n", encoding="utf-8")
    return Vault(tmp_path)


def test_read_note_paging_window(tmp_path):
    tools = {t.name: t for t in build_tools(_long_vault(tmp_path))}
    out = call(tools["read_note"], path="Long", limit=5)
    assert out.startswith("# Long.md")
    assert "   1 |" in out
    assert "   5 |" in out
    assert "   6 |" not in out
    assert "[more: showing lines 1-5 of" in out
    assert "offset=5" in out


def test_read_note_offset_continues_numbering(tmp_path):
    tools = {t.name: t for t in build_tools(_long_vault(tmp_path))}
    out = call(tools["read_note"], path="Long", offset=5, limit=5)
    assert out.startswith("# Long.md")  # header invariant preserved
    assert "   6 |" in out
    assert "  10 |" in out
    assert "   5 |" not in out


def test_read_note_reports_total_size(tmp_path):
    tools = {t.name: t for t in build_tools(_long_vault(tmp_path))}
    out = call(tools["read_note"], path="Long", limit=3)
    # 31 lines total ("# Long" header line + 30 body lines).
    assert "of 31 (" in out
    assert "chars)" in out


def test_get_backlinks(tools):
    out = call(tools["get_backlinks"], note="Compliance Checklist")
    assert "Operations/Vendor-Onboarding.md" in out


def test_listing_tools_cap_and_footer(tmp_path):
    for i in range(120):
        (tmp_path / f"note-{i:03d}.md").write_text("body", encoding="utf-8")
    tools = {t.name: t for t in build_tools(Vault(tmp_path))}
    out = call(tools["glob_notes"], pattern="**/*.md")
    assert out.rstrip().endswith("(showing 100 of 120)")
    assert len(out.splitlines()) == 101  # 100 paths + footer line


def test_search_by_tag(tools):
    out = call(tools["search_by_tag"], tag="compliance")
    assert "Compliance Checklist.md" in out


def test_search_by_tag_with_hash(tools):
    out = call(tools["search_by_tag"], tag="#operations")
    assert "Vendor-Onboarding.md" in out


def test_search_by_tag_frontmatter(tools):
    # Budget-Policy.md declares tags only via YAML frontmatter, no inline #tag.
    out = call(tools["search_by_tag"], tag="finance")
    assert "Finance/Budget-Policy.md" in out


def test_parse_tags_reads_frontmatter_list():
    text = "---\ntags: [finance, budget]\n---\n# Budget\nbody\n"
    assert parse_tags(text) == {"finance", "budget"}


def test_parse_tags_frontmatter_string_form():
    text = "---\ntags: finance, budget\n---\nbody\n"
    assert {"finance", "budget"} <= parse_tags(text)


def test_parse_tags_frontmatter_block_list():
    text = "---\ntags:\n  - finance\n  - budget\n---\nbody\n"
    assert {"finance", "budget"} <= parse_tags(text)


def test_parse_tags_still_reads_inline():
    text = "# Note\n\n#operations #vendor\nbody\n"
    assert parse_tags(text) == {"operations", "vendor"}


def test_read_note_path_escape_blocked(tools):
    # find_note resolves safely; an escaping path simply isn't found.
    out = call(tools["read_note"], path="../../../../etc/passwd")
    assert out.startswith("ERROR")


def test_vault_safe_resolve_rejects_escape():
    vault = Vault(VAULT_DIR)
    with pytest.raises(VaultError):
        vault.safe_resolve("../secrets.md")
