# Building the graph

`data/*.csv` are generated here with [neo4j-graphrag](https://neo4j.com/docs/neo4j-graphrag-python/). Attendees don't need to run this; they load the CSVs with `data/load.cypher`.

Each talk is one `Session` (the graphrag document) with one `Chunk` holding `Title`, `Speakers`, and `Description`. The chunk is embedded with OpenAI `text-embedding-3-small`. An LLM (`gpt-5.4-mini`) extracts `Technology`, `Concept`, `Organization`, and `Domain` entities from it, using the schema in `build_graph.py`. Speaker, track, topics, use cases, and time slots come straight from the sheet's columns.

## Run

Needs [uv](https://docs.astral.sh/uv/), a scratch Neo4j database (`--reset` **deletes everything in it**), and a `.env` file:

```sh
OPENAI_API_KEY=...
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=...
```

```sh
uv run --env-file .env build_graph.py path/to/accepted-sessions.xlsx --reset   # ~30s for ~100 talks
uv run --env-file .env export_csv.py                                           # writes ../data/*.csv
```

Then commit `data/`. Don't commit the spreadsheet.

## The input sheet

One row per speaker per session id, with `Session Id`, `FirstName`, `LastName`, and, on each session's first row, `Title`, `Description`, `Session format`, `Level`, `Topic of your presentation`, `Neo4j Use-Case`, `Room`, `Scheduled At`, and `Scheduled Duration`. Any other columns, such as emails, are ignored.

- Rows with the same title and description are one talk broadcast in several slots. Each becomes a `Slot`, and the lowest session id becomes the talk's id.
- Rows with no title under an existing session id are co-speakers.
- Speakers are keyed by a slug of their full name (`andrew-ng`). `Speaker Id` is ignored, because it's per submission and one person can have several.
- `Scheduled At` is UTC.

## Tuning extraction

- `SCHEMA` in `build_graph.py` holds the node types and their descriptions. These are the main lever.
- When the same thing shows up under two names, add a pair to `ALIASES`.
- Plain "Neo4j" is dropped, because nearly every talk mentions it.

## Maintaining

- Keep the commands in the top-level README in sync with `src/lib/content.ts` in the NODES promo website.
- The repo must be public, because Aura fetches the CSVs itself. If you move or rename it, update the raw GitHub URLs in `data/load.cypher`.
- Keep `agent/agent.json` in sync with the graph's labels, relationships, and index names.
