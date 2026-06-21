"""Read-only vault tools exposed to the agent (Claude Code-style).

``build_tools(vault)`` returns a list of LangChain tools bound to a specific
:class:`~obsidian_agent.vault.Vault`. Every tool stays inside the vault root via
``vault.safe_resolve`` and never mutates anything.
"""

from __future__ import annotations

import re

from langchain_core.tools import tool

from .vault import SearchHit, Vault, VaultError, parse_tags, parse_wikilinks

MAX_FILE_CHARS = 40_000  # guard against dumping huge notes into the context


def build_tools(vault: Vault) -> list:
    """Create the agent's tool set, closed over ``vault``."""

    @tool
    def list_dir(subpath: str = "") -> str:
        """List the folders and Markdown notes directly inside a vault folder.

        Use this to explore the vault structure. ``subpath`` is a vault-relative
        folder path (empty string = vault root). Returns folders (with a trailing
        ``/``) and ``.md`` notes.
        """
        try:
            base = vault.safe_resolve(subpath)
        except VaultError as exc:
            return f"ERROR: {exc}"
        if not base.is_dir():
            return f"ERROR: not a folder: {subpath!r}"
        dirs, notes = [], []
        for child in sorted(base.iterdir()):
            if child.name.startswith("."):
                continue
            if child.is_dir():
                dirs.append(vault.relpath(child) + "/")
            elif child.suffix == ".md":
                notes.append(vault.relpath(child))
        lines = dirs + notes
        return "\n".join(lines) if lines else "(empty)"

    @tool
    def glob_notes(pattern: str) -> str:
        """Find notes by filename glob pattern (e.g. ``**/Onboarding*.md``).

        Matches against vault-relative paths. Returns matching note paths.
        """
        matches = []
        for path in sorted(vault.root.glob(pattern)):
            if path.suffix == ".md" and path.is_file():
                if ".obsidian" in path.parts or ".trash" in path.parts:
                    continue
                matches.append(vault.relpath(path))
        if not matches:
            return f"(no notes match {pattern!r})"
        return "\n".join(matches)

    @tool
    def search_notes(query: str, regex: bool = False, max_results: int = 20) -> str:
        """Search note CONTENT across the whole vault (case-insensitive).

        By default ``query`` is matched as a literal substring; set ``regex=True``
        to use a regular expression. Returns up to ``max_results`` matches as
        ``path:line: text`` so you know which notes to ``read_note``.
        """
        try:
            pat = re.compile(query if regex else re.escape(query), re.IGNORECASE)
        except re.error as exc:
            return f"ERROR: invalid regex: {exc}"
        hits: list[SearchHit] = []
        for note in vault.iter_notes():
            try:
                text = note.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            for i, line in enumerate(text.splitlines(), start=1):
                if pat.search(line):
                    hits.append(SearchHit(vault.relpath(note), i, line.strip()))
                    if len(hits) >= max_results:
                        break
            if len(hits) >= max_results:
                break
        if not hits:
            return f"(no matches for {query!r})"
        return "\n".join(f"{h.path}:{h.line}: {h.text}" for h in hits)

    @tool
    def read_note(path: str) -> str:
        """Read the full text of a note. This is the primary evidence source.

        ``path`` is a vault-relative path or a note name (with or without ``.md``).
        Output is line-numbered. Cite this note's path in your answer.
        """
        note = vault.find_note(path)
        if note is None:
            return f"ERROR: note not found: {path!r}"
        try:
            text = note.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            return f"ERROR: could not read {path!r}: {exc}"
        rel = vault.relpath(note)
        truncated = ""
        if len(text) > MAX_FILE_CHARS:
            text = text[:MAX_FILE_CHARS]
            truncated = "\n... [truncated]"
        numbered = "\n".join(
            f"{i:>4} | {line}" for i, line in enumerate(text.splitlines(), start=1)
        )
        return f"# {rel}\n{numbered}{truncated}"

    @tool
    def get_backlinks(note: str) -> str:
        """List notes that link TO ``note`` via an Obsidian ``[[wikilink]]``.

        Use this to discover related context. ``note`` is a note name or path.
        """
        target = vault.find_note(note)
        if target is None:
            return f"ERROR: note not found: {note!r}"
        target_stem = target.stem.lower()
        backlinks = []
        for other in vault.iter_notes():
            if other == target:
                continue
            try:
                text = other.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            links = {link.split("/")[-1].lower() for link in parse_wikilinks(text)}
            if target_stem in links:
                backlinks.append(vault.relpath(other))
        if not backlinks:
            return f"(no backlinks to {vault.relpath(target)})"
        return "\n".join(backlinks)

    @tool
    def search_by_tag(tag: str) -> str:
        """Find notes containing a given Obsidian ``#tag`` (omit the ``#``).

        Returns the paths of notes whose content includes that tag.
        """
        wanted = tag.lstrip("#").lower()
        matches = []
        for note in vault.iter_notes():
            try:
                text = note.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            if any(t.lower() == wanted for t in parse_tags(text)):
                matches.append(vault.relpath(note))
        if not matches:
            return f"(no notes tagged #{wanted})"
        return "\n".join(matches)

    return [
        list_dir,
        glob_notes,
        search_notes,
        read_note,
        get_backlinks,
        search_by_tag,
    ]
