# Run the NODES 2026 agent locally

A local copy of the Aura Agent: the same graph, system prompt, and tools, running on [CypherLite](https://neo4j-labs.github.io/cypherlite/) (an embeddable graph database) with LangChain and OpenAI. There's no Aura account to set up and nothing runs in the cloud except OpenAI's models.

## Prerequisites

- [CypherLite](https://neo4j-labs.github.io/cypherlite/) on your `PATH` (`cypherlite --version`)
- [uv](https://docs.astral.sh/uv/)
- An OpenAI API key

## Run

```sh
uv run local/setup.py           # loads data/*.csv into local/nodes.db (add --reset to rebuild)
export OPENAI_API_KEY=sk-...
uv run local/chat.py            # add --verbose to see each tool call and the Cypher it ran
```

Set `OPENAI_MODEL` to use a different chat model (default `gpt-5.4-mini`).

## How it works

- `setup.py` runs [`data/load.cypher`](../data/load.cypher), the same script Aura uses, with small changes for CypherLite: `LOAD CSV` reads the local CSVs instead of GitHub, embeddings are stored with a plain `SET`, and each speaker gets an accent-folded `nameFolded`.
- `chat.py` serves `local/nodes.db` read-only over Bolt (`cypherlite serve`) and connects with the Neo4j Python driver. It builds the agent from [`agent/agent.json`](../agent/agent.json):
  - **Cypher templates** run as written, with the model's parameters.
  - **Similarity search** embeds the question with the configured OpenAI model, calls `db.index.vector.queryNodes`, then runs the post-processing Cypher.
  - **Text2Cypher** asks the model for a read-only Cypher query from the tool's schema description, retrying once if it fails.
- Unlike the Aura agent's API, the local chat keeps the conversation, so follow-up questions work.

## Differences from Aura

- `sessions-by-speaker` uses a local Cypher query instead of the full-text index. CypherLite's full-text search matches single whole words only (no wildcards, `OR`/`AND`, or accent folding), so the local version matches every word of the name against the accent-folded speaker names instead.
