"""Command-line interface for the Obsidian research agent."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from langchain_core.messages import AIMessage, ToolMessage
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

from .agent import AgentResult, run_agent, trim_history
from .config import Settings, load_settings

app = typer.Typer(add_completion=False, help="Query your Obsidian vault with an agent.")
console = Console()


def _verbose_event(node: str, payload: object) -> None:
    """Render one step of the agent loop to the terminal."""
    if not isinstance(payload, dict):
        return
    for msg in payload.get("messages", []):
        if isinstance(msg, AIMessage) and msg.tool_calls:
            for call in msg.tool_calls:
                args = ", ".join(f"{k}={v!r}" for k, v in call["args"].items())
                console.print(f"[bold cyan]→ tool[/] {call['name']}({args})")
        elif isinstance(msg, AIMessage) and msg.content:
            console.print("[dim]· agent is composing the answer…[/]")
        elif isinstance(msg, ToolMessage):
            preview = str(msg.content).splitlines()[:1]
            head = preview[0] if preview else ""
            console.print(f"[green]  ↳ {msg.name}[/] {head[:100]}")


def _banner(settings: Settings) -> None:
    """Print the vault/model/endpoint panel shown in verbose mode."""
    console.print(
        Panel.fit(
            f"[bold]vault[/] {settings.resolved_vault()}\n"
            f"[bold]model[/] {settings.model}  "
            f"[bold]endpoint[/] {settings.openai_base_url}",
            title="obsidian-agent",
        )
    )


def _render_result(result: AgentResult) -> None:
    """Print an agent answer, its references, and any truncation warning."""
    console.print()
    console.print(Markdown(result.answer or "_(no answer produced)_"))

    if result.references:
        console.print("\n[bold]References:[/]")
        for ref in result.references:
            console.print(f"  - {ref}")

    if result.truncated:
        console.print(
            "\n[yellow]Note: stopped at the iteration limit; "
            "answer may be incomplete (raise --max-iters).[/]"
        )


def _run_chat(settings: Settings, on_event, first_query: str | None) -> None:
    """Interactive multi-turn REPL with trimmed conversation memory."""
    console.print(
        "[dim]obsidian-agent chat — ask a question. "
        "/reset clears memory, /exit (or Ctrl-D) quits.[/]"
    )
    history: list = []
    pending = first_query
    while True:
        if pending is not None:
            user, pending = pending, None
        else:
            try:
                user = console.input("\n[bold green]you ›[/] ").strip()
            except (EOFError, KeyboardInterrupt):
                console.print()
                break
        if not user:
            continue
        if user in ("/exit", "/quit"):
            break
        if user == "/reset":
            history = []
            console.print("[dim](memory cleared)[/]")
            continue
        try:
            result = run_agent(user, settings, on_event=on_event, history=history)
        except Exception as exc:  # keep the REPL alive on a bad turn
            console.print(f"[bold red]Error:[/] {exc}")
            continue
        _render_result(result)
        history = trim_history(result.messages)


@app.command()
def main(
    query: Optional[str] = typer.Argument(
        None, help="The question to ask your vault (omit with --chat)."
    ),
    chat: bool = typer.Option(
        False, "--chat", "-c", help="Start an interactive multi-turn session."
    ),
    vault: Optional[Path] = typer.Option(
        None, "--vault", help="Vault path (overrides OBSIDIAN_VAULT_PATH)."
    ),
    model: Optional[str] = typer.Option(
        None, "--model", help="Model name (overrides OBSIDIAN_MODEL)."
    ),
    max_iters: Optional[int] = typer.Option(
        None, "--max-iters", help="Max agent/tool iterations."
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Stream the agent loop (tool calls + results)."
    ),
) -> None:
    """Answer QUERY from your Obsidian notes, or run an interactive --chat session."""
    if not chat and not query:
        console.print(
            "[bold red]Error:[/] provide a question, or use --chat for an "
            "interactive session."
        )
        raise typer.Exit(code=2)

    settings = load_settings(
        vault_path=vault,
        model=model,
        max_iterations=max_iters,
    )
    on_event = _verbose_event if verbose else None

    if verbose:
        _banner(settings)

    if chat:
        _run_chat(settings, on_event, first_query=query)
        return

    try:
        result = run_agent(query, settings, on_event=on_event)
    except Exception as exc:  # surface config/endpoint errors cleanly
        console.print(f"[bold red]Error:[/] {exc}")
        raise typer.Exit(code=1) from exc

    _render_result(result)


if __name__ == "__main__":
    app()
