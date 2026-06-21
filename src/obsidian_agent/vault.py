"""Vault access helpers: path sandboxing, iteration, and link/tag parsing.

Everything here is read-only and constrained to the configured vault root. Tools
in ``tools.py`` build on these primitives so that no path outside the vault can
ever be read.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

# [[wikilink]] or [[wikilink#heading]] or [[wikilink|alias]]
_WIKILINK_RE = re.compile(r"\[\[([^\]\|#]+)(?:#[^\]\|]+)?(?:\|[^\]]+)?\]\]")
# #tag (not inside a word, not a markdown heading). Allows nested a/b and hyphens.
_TAG_RE = re.compile(r"(?:^|(?<=\s))#([A-Za-z0-9_][A-Za-z0-9_/\-]*)")


class VaultError(Exception):
    """Raised when a requested path escapes or is missing from the vault."""


@dataclass(frozen=True)
class SearchHit:
    """A single content match inside a note."""

    path: str  # vault-relative posix path
    line: int  # 1-based line number
    text: str  # the matching line (stripped)


class Vault:
    """A read-only view over an Obsidian vault rooted at ``root``."""

    def __init__(self, root: Path) -> None:
        self.root = root.expanduser().resolve()
        if not self.root.is_dir():
            raise VaultError(f"Vault path is not a directory: {self.root}")

    # --- path safety ---------------------------------------------------------
    def safe_resolve(self, rel: str) -> Path:
        """Resolve a vault-relative path, rejecting anything outside the root.

        Blocks absolute paths, ``..`` traversal, and symlinks that escape.
        """
        rel = (rel or "").strip().lstrip("/")
        candidate = (self.root / rel).resolve()
        if candidate != self.root and self.root not in candidate.parents:
            raise VaultError(f"Path escapes the vault: {rel!r}")
        return candidate

    def relpath(self, path: Path) -> str:
        """Return the vault-relative posix path for an absolute path."""
        return path.resolve().relative_to(self.root).as_posix()

    # --- iteration -----------------------------------------------------------
    def iter_notes(self):
        """Yield every Markdown note in the vault (skipping ``.obsidian``)."""
        for path in sorted(self.root.rglob("*.md")):
            if ".obsidian" in path.parts or ".trash" in path.parts:
                continue
            yield path

    def find_note(self, name: str) -> Path | None:
        """Resolve a note by relative path or by basename (with/without .md)."""
        name = name.strip()
        stem = name[:-3] if name.endswith(".md") else name
        # exact relative path first
        for candidate in (f"{stem}.md", name):
            try:
                p = self.safe_resolve(candidate)
            except VaultError:
                continue
            if p.is_file():
                return p
        # fall back to unique basename match
        target = Path(stem).name.lower()
        for note in self.iter_notes():
            if note.stem.lower() == target:
                return note
        return None


# --- parsing -----------------------------------------------------------------
def parse_wikilinks(text: str) -> list[str]:
    """Return the link targets of all ``[[wikilinks]]`` in ``text``."""
    return [m.group(1).strip() for m in _WIKILINK_RE.finditer(text)]


def parse_tags(text: str) -> set[str]:
    """Return the set of ``#tags`` in ``text`` (without the leading ``#``)."""
    return {m.group(1) for m in _TAG_RE.finditer(text)}
