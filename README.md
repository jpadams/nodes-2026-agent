# NODES 2026 agent on Relate

Chat with the [NODES 2026](https://neo4j.com/nodes/) sessions, speakers, and schedule. The whole graph, with its vector and full-text indexes, is one [Relate](https://relatedb.ai/) database file, `nodes-2026.rl`, that runs inside the app. The agent's system prompt and tools come from `agent.json`.

There are two versions of the agent, one per framework:

- `strands-relate-agent.py`: [Strands Agents](https://strandsagents.com/)
- `langchain-relate-agent.py`: [LangChain](https://docs.langchain.com/)

Both use `nodes_graph.py` to query the graph.

## Run

You need [uv](https://docs.astral.sh/uv/) and an OpenAI API key.

```sh
export OPENAI_API_KEY=sk-...
uv run strands-relate-agent.py                          # chat until Ctrl-C or Ctrl-D
uv run strands-relate-agent.py "What's on Wednesday?"   # answer one question and exit
uv run langchain-relate-agent.py "What's on Wednesday?"
```

Add `--verbose` to see each tool call. Set `OPENAI_MODEL` to use a different chat model (default `gpt-5.4-mini`).

## Graph

```
(:Speaker)-[:PRESENTS]->(:Session)-[:HAS_SLOT]->(:Slot {startTime, endTime, durationMinutes})
(:Session)-[:IN_TRACK]->(:Track)
(:Session)-[:HAS_TOPIC]->(:Topic)
(:Session)-[:HAS_USE_CASE]->(:UseCase)
(:Session)-[:MENTIONS]->(:Technology|Concept|Organization|Domain)
(:Chunk:SessionChunk {text, embedding})-[:FROM_DOCUMENT]->(:Session)
```

Times are UTC. To query it directly with the [`relate` CLI](https://relatedb.ai/):

```sh
relate -d nodes-2026.rl --read-only "MATCH (s:Speaker) RETURN count(s)"
```
