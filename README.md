# obsidian-agent

A LangGraph agent that answers a question by **navigating your Obsidian vault
with tools** (list / glob / grep / read / backlinks / tags) in a Claude-style
agent loop, then replies grounded in your notes with **relative-path citations**.

No embeddings or vector database — the LLM explores the vault itself, the same
way Claude Code explores a codebase.

## How it works

```
your query ──▶ agent (LLM + tools) ──▶ tool calls? ──yes──▶ run tools ──▶ back to agent
                                            │
                                            no ──▶ final answer + References
```

Built with a manual LangGraph `StateGraph` + `ToolNode`:

- `agent` node calls an OpenAI-compatible endpoint (`ChatOpenAI(base_url=...)`)
  with the tools bound.
- `tools` node runs any requested vault tools and feeds the results back.
- The loop repeats until the model answers with no further tool calls.
- References are collected deterministically from the notes the agent actually
  read (`read_note`).

## Setup

Requires [uv](https://docs.astral.sh/uv/) and Python 3.12+.

```bash
uv sync                 # create .venv and install deps
cp .env.example .env    # then edit .env
```

Fill in `.env`:

| Var | Meaning |
|-----|---------|
| `OPENAI_BASE_URL` | Your company OpenAI-compatible endpoint (ends in `/v1`). |
| `OPENAI_API_KEY` | Any non-empty string if the endpoint ignores it. |
| `OBSIDIAN_MODEL` | Model served by the endpoint (e.g. `gpt-oss`). |
| `OBSIDIAN_VAULT_PATH` | Absolute path to your vault root. |
| `AGENT_MAX_ITERATIONS` | Loop safety limit (default 12). |
| `AGENT_TEMPERATURE` | Sampling temperature (default 0). |

> The model must support OpenAI-style **tool calling** (the `tools` parameter).

## Usage

```bash
uv run obsidian-agent "How do we onboard a new operations vendor?"

# overrides + see the agent's tool calls live
uv run obsidian-agent "..." --vault /path/to/Vault --model gpt-oss --max-iters 16 --verbose
```

Output: a grounded answer followed by a `References:` list of note paths.

## Tools the agent can use

| Tool | Purpose |
|------|---------|
| `list_dir` | List folders/notes in a vault folder. |
| `glob_notes` | Find notes by filename pattern. |
| `search_notes` | Grep note content (substring or regex). |
| `read_note` | Read a note in full (primary evidence). |
| `get_backlinks` | Notes linking to a note via `[[wikilink]]`. |
| `search_by_tag` | Notes containing a `#tag`. |

All tools are **read-only** and sandboxed to the vault root.

## Development

```bash
uv run pytest           # tool tests against a fixture vault
```

## Project layout

```
src/obsidian_agent/
  config.py   # typed settings from .env / CLI
  vault.py    # path sandboxing + wikilink/tag parsing
  tools.py    # the read-only vault tools
  agent.py    # StateGraph loop, prompt, reference collection
  cli.py      # typer entrypoint
tests/        # fixture vault + tool tests
```
