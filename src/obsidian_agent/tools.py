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
LIST_MAX = 100  # cap listing-style output so a huge vault can't blow the window


def _bounded(lines: list[str]) -> str:
    """Join listing output, truncating to ``LIST_MAX`` with a count footer."""
    if len(lines) <= LIST_MAX:
        return "\n".join(lines)
    shown = "\n".join(lines[:LIST_MAX])
    return f"{shown}\n(showing {LIST_MAX} of {len(lines)})"


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
        return _bounded(lines) if lines else "(empty)"

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
        return _bounded(matches)

    @tool
    def search_notes(
        query: str, regex: bool = False, max_results: int = 20, max_per_file: int = 3
    ) -> str:
        """Search note CONTENT across the whole vault (case-insensitive).

        By default ``query`` is matched as a literal substring; set ``regex=True``
        to use a regular expression. Returns up to ``max_results`` matches as
        ``path:line: text``, capped at ``max_per_file`` per note so results span
        more notes instead of piling up in the first one. A footer reports how
        many notes matched; narrow the query if it says results were capped.
        """
        try:
            pat = re.compile(query if regex else re.escape(query), re.IGNORECASE)
        except re.error as exc:
            return f"ERROR: invalid regex: {exc}"
        hits: list[SearchHit] = []
        n_files = 0
        for note in vault.iter_notes():
            try:
                text = note.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            file_hits = 0
            for i, line in enumerate(text.splitlines(), start=1):
                if pat.search(line):
                    hits.append(SearchHit(vault.relpath(note), i, line.strip()))
                    file_hits += 1
                    if file_hits >= max_per_file or len(hits) >= max_results:
                        break
            if file_hits:
                n_files += 1
            if len(hits) >= max_results:
                break
        if not hits:
            return f"(no matches for {query!r})"
        body = "\n".join(f"{h.path}:{h.line}: {h.text}" for h in hits)
        if len(hits) >= max_results:
            footer = (
                f"\n(showing {len(hits)} matches across {n_files} notes; "
                "narrow the query to see more)"
            )
        else:
            footer = f"\n({len(hits)} matches across {n_files} notes)"
        return f"{body}{footer}"

    @tool
    def read_note(path: str, offset: int = 0, limit: int | None = None) -> str:
        """Read a note's text, line-numbered. This is the primary evidence source.

        ``path`` is a vault-relative path or a note name (with or without ``.md``).
        ``offset`` is the 0-based starting line; ``limit`` caps how many lines are
        returned (default: a window up to the size limit). The first output line
        is always ``# <relative-path>`` — cite that path in your answer. If a long
        note is only partly shown, a ``[more: ...]`` footer reports its total size
        and how to fetch the rest with ``offset``.
        """
        note = vault.find_note(path)
        if note is None:
            return f"ERROR: note not found: {path!r}"
        try:
            text = note.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            return f"ERROR: could not read {path!r}: {exc}"
        rel = vault.relpath(note)
        all_lines = text.splitlines()
        total_lines = len(all_lines)
        total_chars = len(text)
        start = max(0, offset)

        # Take lines from `start`, bounded by `limit` and by MAX_FILE_CHARS so the
        # body never overflows the context window (always keep at least one line).
        selected: list[str] = []
        budget = MAX_FILE_CHARS
        for line in all_lines[start:]:
            if limit is not None and len(selected) >= limit:
                break
            if selected and budget - (len(line) + 1) < 0:
                break
            selected.append(line)
            budget -= len(line) + 1
        end = start + len(selected)

        numbered = "\n".join(
            f"{i:>4} | {line}" for i, line in enumerate(selected, start=start + 1)
        )
        footer = ""
        if end < total_lines:
            footer = (
                f"\n[more: showing lines {start + 1}-{end} of {total_lines} "
                f"({total_chars} chars). Call read_note(path, offset={end}) for more.]"
            )
        return f"# {rel}\n{numbered}{footer}"

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
        return _bounded(backlinks)

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
