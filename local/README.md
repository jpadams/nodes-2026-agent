# Run the NODES 2026 agent locally

A local copy of the Aura Agent: the same graph, system prompt, and tools, with [Relate](https://relatedb.ai/) (an embeddable graph database)  running inside the app, plus OpenAI. There's no Aura account, database server, or setup step: each run loads the graph into memory from `data/*.csv` in well under a second.

There are two variants with the same graph, prompt, and tools. Only the agent framework differs:

| Script | Framework |
|---|---|
| [`strands-relate-agent.py`](strands-relate-agent.py) | [Strands Agents](https://strandsagents.com/) |
| [`langchain-relate-agent.py`](langchain-relate-agent.py) | [LangChain](https://docs.langchain.com/) |

Each script builds its agent the way its framework recommends:

| | Strands Agents | LangChain |
|---|---|---|
| Tools | `@tool` functions, named and described from `agent.json` | `@tool` functions, named and described from `agent.json` |
| Text2Cypher | An agent as a tool: a specialist agent writes the Cypher | The chat model writes the Cypher |
| Embeddings | OpenAI client (Strands has no embeddings API) | `init_embeddings` |
| Conversation | Kept by the `Agent` | `InMemorySaver` checkpointer, one thread |
| `--verbose` | A `Plugin` with `@hook`s on tool calls | `stream_mode="updates"` |

Both run their Cypher on the graph in [`nodes_graph.py`](nodes_graph.py), the Relate side, which doesn't depend on either framework. It loads the graph, runs each kind of `agent.json` tool's Cypher, and holds the shared command line.

## Prerequisites

- [uv](https://docs.astral.sh/uv/)
- An OpenAI API key

On first run, uv installs each script's dependencies, including Relate ([`relate-db`](https://pypi.org/project/relate-db/) on PyPI).

## Run

Either script takes the same arguments:

```sh
export OPENAI_API_KEY=sk-...
uv run local/strands-relate-agent.py                         # chat until Ctrl-C or Ctrl-D
uv run local/strands-relate-agent.py "What's on at 18:00 UTC?"  # answer one question and exit
uv run local/langchain-relate-agent.py "What's on at 18:00 UTC?"
```

Add `--verbose` to see each tool call and the Cypher it ran.

Set `OPENAI_MODEL` to use a different chat model (default `gpt-5.4-mini`).

## How it works

- On start, the agent opens an in-memory Relate database and runs [`data/load.cypher`](../data/load.cypher), the same script Aura uses, with small changes for Relate: `LOAD CSV` reads the local CSVs instead of GitHub, embeddings are stored with a plain `SET`, and each speaker gets an accent-folded `nameFolded`.
- It builds the agent from [`agent/agent.json`](../agent/agent.json):
  - **Cypher templates** run as written, with the model's parameters.
  - **Similarity search** embeds the question with the configured OpenAI model, calls `db.index.vector.queryNodes`, then runs the post-processing Cypher.
  - **Text2Cypher** asks the model for a read-only Cypher query from the tool's schema description, retrying once with the error if it fails. It returns up to 300 rows, and says so when there are more.
- The local chat keeps the conversation, as the Aura console does, so follow-up questions work.

## Differences from Aura

- `sessions-by-speaker` uses a local Cypher query instead of the full-text index. Relate's full-text search handles `AND`, `OR`, and fuzzy `~` terms, but not wildcards or accent folding, so Aura's query would miss names like "Kollegg", "Eifrém", or "andreas sateras". The local version matches every word of the name against the accent-folded speaker names instead. If that finds no one, it corrects typos and tries again: each word of 3 or more letters becomes the closest word in a speaker's name, so "Kolleger" finds Andreas Kollegger, as on Aura. (Relate accepts `~` fuzzy terms but doesn't match misspellings yet, so this runs in Python.)
