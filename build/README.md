# Building the graph

`data/*.csv` are generated here with [neo4j-graphrag](https://neo4j.com/docs/neo4j-graphrag-python/). Attendees don't need to run this; they load the CSVs with `data/load.cypher`.

Each talk is one `Session` (the graphrag document) with one `Chunk` holding `Title`, `Speakers`, and `Description`. The chunk is embedded with OpenAI `text-embedding-3-small`. An LLM (`gpt-5.4-mini`) extracts `Technology`, `Concept`, `Organization`, and `Domain` entities from it, using the schema in `build_graph.py`. Speaker, track, topics, use cases, and time slots come straight from the sheet's columns.

## From the spreadsheet to the live agents

This repo is published from the `companion-repo/` folder of the NODES promo website (nodes-promo-app). Edit there, then publish with that repo's `scripts/sync-companion.sh` (see step 4).

1. **Get the sheet.** Export the accepted sessions to `.xlsx` (see [The input sheet](#the-input-sheet)). Keep it out of git: `*.xlsx` is ignored here.

2. **Build the graph in a scratch Neo4j database.** Run this in nodes-promo-app's `companion-repo/build/`, not in a clone of this repo, so the export lands in `companion-repo/data/`, ready to publish. You need [uv](https://docs.astral.sh/uv/), an OpenAI API key, and a scratch database: `--reset` **deletes everything in it**. A throwaway Docker container from neo4j-cli works: `neo4j-cli docker create --name nodes-build --ephemeral --wait --env-out-file neo4j.env --rw` writes its `NEO4J_*` lines to `neo4j.env`. Put those lines and your key in `build/.env` (gitignored):

   ```sh
   OPENAI_API_KEY=...
   NEO4J_URI=bolt://localhost:7687
   NEO4J_USERNAME=neo4j
   NEO4J_PASSWORD=...
   ```

   ```sh
   cd build
   uv run --env-file .env build_graph.py path/to/accepted-sessions.xlsx --reset   # ~30s for ~100 talks
   uv run --env-file .env export_csv.py                                           # writes ../data/*.csv
   ```

   This calls OpenAI once per talk for embeddings and entity extraction. Re-runs give slightly different entities.

3. **Check the result.** In nodes-promo-app, `git diff --stat companion-repo/data/` shows what changed. The count query at the end of `data/load.cypher` prints the new totals. The docs and agent prompts don't hard-code them, so a different number of sessions needs no other edits.

4. **Publish.** From nodes-promo-app: `scripts/sync-companion.sh` shows what would change, and `scripts/sync-companion.sh "Update sessions from the <date> sheet"` commits it on `main` with a sign-off. Then push `main`. Until it's pushed, nobody sees the new data, because `data/load.cypher` loads the CSVs from this repo's raw GitHub URLs on `main`.

5. **Reload the live databases.** `data/load.cypher` only merges, so re-running it adds new talks and updates changed ones, but talks, speakers, or entities that were dropped stay behind. For the chat's pool instances, which hold nothing else, clear the graph first, then load. Run this once per instance:

   ```sh
   neo4j-cli query --credential <dbms-credential> --rw "MATCH (n) DETACH DELETE n"
   neo4j-cli query --credential <dbms-credential> --rw < data/load.cypher
   ```

   Attendees who already built their agent can re-run `data/load.cypher`. They get the new and changed talks, and dropped ones linger.

6. **Clear stale answers.** The website caches chat answers for 6 hours. Delete the `ask:answer:*` keys in its Upstash console so the chat reflects the new data straight away.

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
- Editing `agent/agent.json` doesn't change agents that already exist. Push it to each one with `neo4j-cli aura agent update <agent-id> --tools "$(jq -c .tools agent/agent.json)" --system-prompt "$(jq -r .system_prompt agent/agent.json)" --rw`.

## Agent design notes

- **Speaker lookup (`sessions-by-speaker`)** queries the `speaker_names` full-text index (`standard-folding`, so accents are optional). Each word of the name must match, as itself, as a prefix (`kol*`), or with small typos: one edit for words of 3–7 letters, two for 8 or more, none for 1–2 letters. Two edits on short words matched the wrong people ("tom" found Tony and Jim, "andreas" found Andrew), and fuzzy one- or two-letter words match almost anyone. Lucene syntax characters are stripped from the input first, so a name can't break the query.
- **Every presenter is returned** with each session (`speakers`), so the agent can answer "who is X presenting with".
- **Times** come from `Slot.displayTime` (like "Wed 11 Nov, 18:00 UTC"), and the system prompt tells the agent to copy them as written. `Slot.startTime` is only for filtering and sorting. A talk broadcast in several slots lists them all.
- **Pronouns:** the system prompt tells the agent to refer to speakers by name or as "they", and not to guess anyone's pronouns.
