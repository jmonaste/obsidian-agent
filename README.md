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

Requires Python 3.12+. Pick whichever package manager your machine allows.

### Option A — pip + venv (works on restricted / corporate laptops)

```bash
python -m venv .venv
# Windows:        .venv\Scripts\activate
# macOS / Linux:  source .venv/bin/activate

python -m pip install --upgrade pip
pip install -r requirements.txt   # runtime deps
pip install -e . --no-deps        # install the CLI itself (no re-resolve)

cp .env.example .env              # then edit .env  (Windows: copy .env.example .env)
```

For development extras (tests):

```bash
pip install -e ".[dev]"
```

> **Behind a corporate proxy / internal index?** Point pip at your mirror, e.g.
> `pip install -r requirements.txt --index-url https://<your-artifactory>/api/pypi/pypi/simple`.
> If SSL inspection breaks TLS, add `--trusted-host <host>`. To make it permanent
> put those under `[global]` in `pip.conf` (`~/.config/pip/pip.conf` on
> macOS/Linux, `%APPDATA%\pip\pip.ini` on Windows).

### Option B — conda / miniforge

```bash
conda create -n obsidian-agent python=3.12 -y
conda activate obsidian-agent
pip install -r requirements.txt
pip install -e . --no-deps
cp .env.example .env
```

> miniforge defaults to conda-forge; `pip` inside the env will still respect any
> corporate `pip.conf` index settings described above.

### Option C — uv (if available and unrestricted)

```bash
uv venv && uv pip install -e ".[dev]"
cp .env.example .env
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

With the virtual environment activated, the `obsidian-agent` command is on your
PATH:

```bash
obsidian-agent "How do we onboard a new operations vendor?"

# overrides + see the agent's tool calls live
obsidian-agent "..." --vault /path/to/Vault --model gpt-oss --max-iters 16 --verbose
```

You can also invoke it without activating the env:

```bash
python -m obsidian_agent.cli "How do we onboard a new operations vendor?"
```

Output: a grounded answer followed by a `References:` list of note paths.

### Interactive chat

Use `--chat` (or `-c`) for a multi-turn session that remembers the conversation,
so you can ask follow-up questions:

```bash
obsidian-agent --chat --vault /path/to/Vault
```

In the REPL: type a question, `/reset` clears the memory, and `/exit` (or
Ctrl-D) quits. To keep the context window small, only the question/answer
transcript is carried forward (the last ~6 turns); bulky tool output from prior
turns is dropped.

## Tools the agent can use

| Tool | Purpose |
|------|---------|
| `list_dir` | List folders/notes in a vault folder. |
| `glob_notes` | Find notes by filename pattern. |
| `search_notes` | Grep note content (substring or regex), capped per file so results span more notes. |
| `read_note` | Read a note as evidence; long notes are paged (`offset`/`limit`) with a `[more: ...]` hint instead of silently truncated. |
| `get_backlinks` | Notes linking to a note via `[[wikilink]]`. |
| `search_by_tag` | Notes carrying a tag, from inline `#tags` **or** YAML `tags:` frontmatter. |

All tools are **read-only** and sandboxed to the vault root. Listing/search
output is bounded (with a `(showing N of M)` footer) so a large vault can't
overflow a limited context window.

## Development

```bash
pip install -e ".[dev]"   # once, to get pytest
pytest                    # tool tests against a fixture vault
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
