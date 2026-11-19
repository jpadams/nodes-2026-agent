# NODES 2026 agent on Relate

Chat with the [NODES 2026](https://neo4j.com/nodes/) sessions, speakers, and schedule. It's one [Strands Agents](https://strandsagents.com/) script, `strands-relate-agent.py`, and one [Relate](https://relatedb.ai/) database file, `nodes-2026.rl`, which holds the whole graph with its vector and full-text indexes and runs inside the app.

## Run

You need [uv](https://docs.astral.sh/uv/) and an OpenAI API key.

```sh
export OPENAI_API_KEY=sk-...
uv run strands-relate-agent.py                          # chat until Ctrl-C or Ctrl-D
uv run strands-relate-agent.py "What's on Thursday?"    # answer one question and exit
```

Set `OPENAI_MODEL` to use a different chat model (default `gpt-5.4-mini`).

## Graph

```
(:Speaker)-[:PRESENTS]->(:Session)-[:HAS_SLOT]->(:Slot {startTime, endTime, durationMinutes, displayTime})
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
