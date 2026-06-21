"""Command-line interface for the Obsidian research agent."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from langchain_core.messages import AIMessage, ToolMessage
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

from .agent import run_agent
from .config import load_settings

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


@app.command()
def main(
    query: str = typer.Argument(..., help="The question to ask your vault."),
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
    """Answer QUERY using your Obsidian notes, with relative-path citations."""
    settings = load_settings(
        vault_path=vault,
        model=model,
        max_iterations=max_iters,
    )

    if verbose:
        console.print(
            Panel.fit(
                f"[bold]vault[/] {settings.resolved_vault()}\n"
                f"[bold]model[/] {settings.model}  "
                f"[bold]endpoint[/] {settings.openai_base_url}",
                title="obsidian-agent",
            )
        )

    on_event = _verbose_event if verbose else None
    try:
        result = run_agent(query, settings, on_event=on_event)
    except Exception as exc:  # surface config/endpoint errors cleanly
        console.print(f"[bold red]Error:[/] {exc}")
        raise typer.Exit(code=1) from exc

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


if __name__ == "__main__":
    app()
