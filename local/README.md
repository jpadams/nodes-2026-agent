# Run the NODES 2026 agent locally

A local copy of the Aura Agent: the same graph, system prompt, and tools, with [Relate](https://relatedb.ai/) (an embeddable graph database)  running inside the app, plus OpenAI. There's no Aura account, database server, or setup step: each run loads the graph into memory from `data/*.csv` in well under a second.

There are two variants with the same graph, prompt, and tools. Only the agent framework differs:

| Script | Framework |
|---|---|
| [`strands-relate-agent.py`](strands-relate-agent.py) | [Strands Agents](https://strandsagents.com/) |
| [`langchain-relate-agent.py`](langchain-relate-agent.py) | [LangChain](https://docs.langchain.com/) |

## Prerequisites

- [uv](https://docs.astral.sh/uv/)
- An OpenAI API key
- The Relate Python wheels (`relate-db`). Relate isn't on PyPI yet, so download them from its GitHub release into `local/wheels/` (uv picks the one for your platform):

  ```sh
  gh release download v0.1.2 -R relatedb/relate -p 'relate_db-*.whl' -D local/wheels
  ```

## Run

From the repo root (either script takes the same arguments):

```sh
export OPENAI_API_KEY=sk-...
uv run local/strands-relate-agent.py                         # chat until Ctrl-C or Ctrl-D
uv run local/strands-relate-agent.py "What's on Wednesday?"  # answer one question and exit
uv run local/langchain-relate-agent.py "What's on Wednesday?"
```

Add `--verbose` to see each tool call and the Cypher it ran.

Set `OPENAI_MODEL` to use a different chat model (default `gpt-5.4-mini`).

## How it works

- On start, the agent opens an in-memory Relate database and runs [`data/load.cypher`](../data/load.cypher), the same script Aura uses, with small changes for Relate: `LOAD CSV` reads the local CSVs instead of GitHub, embeddings are stored with a plain `SET`, and each speaker gets an accent-folded `nameFolded`.
- It builds the agent from [`agent/agent.json`](../agent/agent.json):
  - **Cypher templates** run as written, with the model's parameters.
  - **Similarity search** embeds the question with the configured OpenAI model, calls `db.index.vector.queryNodes`, then runs the post-processing Cypher.
  - **Text2Cypher** asks the model for a read-only Cypher query from the tool's schema description, retrying once if it fails.
- Unlike the Aura agent's API, the local chat keeps the conversation, so follow-up questions work.

## Differences from Aura

- `sessions-by-speaker` uses a local Cypher query instead of the full-text index. Relate's full-text search matches single whole words only (no wildcards, `OR`/`AND`, or accent folding), so the local version matches every word of the name against the accent-folded speaker names instead.
