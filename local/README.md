# Run the NODES 2026 agent locally

A local copy of the Aura Agent: the same graph, system prompt, and tools, with [CypherLite](https://neo4j-labs.github.io/cypherlite/) (an embeddable graph database) running inside the app, plus LangChain and OpenAI. There's no Aura account, database server, or setup step: each run loads the graph into memory from `data/*.csv` in well under a second.

## Prerequisites

- [uv](https://docs.astral.sh/uv/)
- An OpenAI API key
- The CypherLite Python wheels. CypherLite isn't on PyPI yet, so download them from its GitHub release into `local/wheels/` (uv picks the one for your platform):

  ```sh
  gh release download v0.13.0 -R neo4j-labs/cypherlite -p '*.whl' -D local/wheels
  ```

## Run

From the repo root:

```sh
export OPENAI_API_KEY=sk-...
uv run local/chat.py                         # chat until Ctrl-C or Ctrl-D
uv run local/chat.py "What's on Wednesday?"  # answer one question and exit
```

Add `--verbose` to see each tool call and the Cypher it ran.

Set `OPENAI_MODEL` to use a different chat model (default `gpt-5.4-mini`).

## How it works

- On start, `chat.py` opens an in-memory CypherLite database and runs [`data/load.cypher`](../data/load.cypher), the same script Aura uses, with small changes for CypherLite: `LOAD CSV` reads the local CSVs instead of GitHub, embeddings are stored with a plain `SET`, and each speaker gets an accent-folded `nameFolded`.
- It builds the agent from [`agent/agent.json`](../agent/agent.json):
  - **Cypher templates** run as written, with the model's parameters.
  - **Similarity search** embeds the question with the configured OpenAI model, calls `db.index.vector.queryNodes`, then runs the post-processing Cypher.
  - **Text2Cypher** asks the model for a read-only Cypher query from the tool's schema description, retrying once if it fails.
- Unlike the Aura agent's API, the local chat keeps the conversation, so follow-up questions work.

## Differences from Aura

- `sessions-by-speaker` uses a local Cypher query instead of the full-text index. CypherLite's full-text search matches single whole words only (no wildcards, `OR`/`AND`, or accent folding), so the local version matches every word of the name against the accent-folded speaker names instead.
