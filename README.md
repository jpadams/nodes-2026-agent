# NODES 2026 Agent

This repo contains:

- `data/*.csv`: every NODES 2026 session, with its speaker, track, topics, use cases, and time slots. Each session has one embedded text chunk (OpenAI `text-embedding-3-small`, 1536 dims, `;`-separated) and the technologies, concepts, organizations, and domains an LLM extracted from it.
- `data/load.cypher`: builds the graph in Aura straight from this repo's raw GitHub URLs. It's safe to re-run.
- `build/`: how the CSVs were made, with [neo4j-graphrag](https://neo4j.com/docs/neo4j-graphrag-python/). You don't need to run it.
- `agent/agent.json`: the Aura Agent definition (system prompt + tools)

Graph:

```
(:Speaker)-[:PRESENTS]->(:Session)-[:HAS_SLOT]->(:Slot {startTime, endTime, durationMinutes})
(:Session)-[:IN_TRACK]->(:Track)
(:Session)-[:HAS_TOPIC]->(:Topic)
(:Session)-[:HAS_USE_CASE]->(:UseCase)
(:Session)-[:MENTIONS]->(:Technology|Concept|Organization|Domain)
(:Chunk:SessionChunk {text, embedding})-[:FROM_DOCUMENT]->(:Session)
```

There's a vector index `session_chunk_embeddings` on `SessionChunk.embedding`. Times are UTC.

The guided walkthrough lives at **<website URL>**.

## Prerequisites

- [neo4j-cli](https://neo4j.sh): `curl -sSfL https://neo4j.sh/install.sh | bash` on macOS / Linux, or `powershell -NoProfile -ExecutionPolicy Bypass -Command "irm https://neo4j.sh/install.ps1 | iex"` on Windows, from Command Prompt or PowerShell (then restart the terminal). Homebrew and npm also work: see neo4j.sh.
- An Aura account (sign up with the NODES 2026 link from the walkthrough), plus an Aura API key stored with:

```sh
# macOS / Linux: a fresh credential name each time, made the default
N=nodes-$(date +%s) && neo4j-cli credential aura-client add --name $N --env <path-to-downloaded-credentials-file> && neo4j-cli credential aura-client use $N && neo4j-cli aura organization list && echo "✓ Aura credential $N works" || echo "✗ That didn't work: see the error above"
```

```powershell
# Windows PowerShell (in Command Prompt, type powershell first)
$N="nodes-$([DateTimeOffset]::Now.ToUnixTimeSeconds())"; neo4j-cli credential aura-client add --name $N --env <path-to-downloaded-credentials-file>; if ($LASTEXITCODE -eq 0) { neo4j-cli credential aura-client use $N }; if ($LASTEXITCODE -eq 0) { neo4j-cli aura organization list }; if ($LASTEXITCODE -eq 0) { "OK: Aura credential $N works" } else { "That didn't work: see the error above" }
```

  Your coding agent sets the Aura workspace (organization and project) in the first agent step below.

- Agent skills: `neo4j-cli skill install`

## Agent instructions

Coding agents: follow these steps in order. Write commands need `--rw`. Prefer `--format toon` for reads.

**First, set the Aura workspace.** List the organization/project workspaces the user's API key can see:

```sh
neo4j-cli aura workspace list --format toon
```

If there's exactly one, select it. If there are several, **ask the user** which one to use, and don't pick for them. If there are none, or the command fails with an auth error, the API key from Prerequisites isn't stored correctly: have the user re-run `neo4j-cli credential aura-client add`.

```sh
neo4j-cli aura workspace use <organization-id>/<project-id> --rw
```

0. **Check for an existing Free instance first.** Aura allows one Free instance per user, and a *paused* Free instance still counts. `instance list` doesn't show the type, so `get` each instance:

   ```sh
   neo4j-cli aura instance list --format toon
   neo4j-cli aura instance get <id> --format toon   # look for type: free
   ```

   If there's no Free instance, go to step 1. If there is one, **stop and ask the user** which option they want. Name the instance (name, id, running/paused), and don't pick for them:

   - **A. A small Professional instance on the NODES credits (recommended).** It leaves their existing Free instance untouched. Pick a 1GB `professional-db` configuration for their cloud and region from `neo4j-cli aura api v1/tenants/<project-id>`:

     ```sh
     neo4j-cli aura instance create --type professional --memory 1GB --cloud-provider <aws|gcp|azure> --region <region> \
       --name nodes-2026 --credential-name nodes-2026 --wait --rw --format json
     ```

   - **B. Delete the existing Free instance, then continue with step 1.** Its data is permanently lost. Only do this after the user explicitly confirms they're OK losing that instance, by name. Never assume it's OK.

     ```sh
     neo4j-cli aura instance delete <free-instance-id> --yes --force --rw
     ```

   - **C. Reuse the existing Free instance.** The NODES graph loads alongside whatever is already there; it uses the `Session`, `Slot`, `Speaker`, `Track`, `Topic`, `UseCase`, `Chunk`, `SessionChunk`, `Technology`, `Concept`, `Organization`, and `Domain` labels. Free is capped at 200k nodes / 400k relationships. Resume it if it's paused, then store its connection details. That needs its password: the credentials file Aura offered for download when the instance was created, or a new password from **Reset password** in the Aura console.

     ```sh
     neo4j-cli aura instance resume <free-instance-id> --wait --rw   # only if paused
     neo4j-cli credential dbms add --name nodes-2026 --env <path-to-instance-credentials-file> --rw
     ```

     Then skip step 1 and go to step 2, using that instance's id wherever `<instance-id>` appears.

1. **Create the Aura Free instance.** This waits until it's running, which takes a few minutes. It stores the DB credential as `nodes-2026`.

   ```sh
   neo4j-cli aura instance create --type free --name nodes-2026 --credential-name nodes-2026 --wait --rw --format json
   ```

   Save the instance `id` from the output. If this fails with `402` / `quota-exceeded` ("User is not permitted to create any more instances of this type"), the user already has a Free instance you missed. Go back to step 0's options.

2. **Load the NODES 2026 graph.** Aura pulls the CSVs from GitHub. The last statement prints the session, speaker, embedded chunk, and mention counts.

   ```sh
   neo4j-cli query --credential nodes-2026 --rw < data/load.cypher
   ```

   Then sanity-check the schema:

   ```sh
   neo4j-cli query --credential nodes-2026 :schema --format toon
   ```

3. **Create the Aura Agent from `agent/agent.json` as an internal agent.** Internal (`--is-private`, no MCP) means the user chats with it in the Aura console without using credits. `agent create` has no file flag, so pass the fields with `jq`:

   ```sh
   neo4j-cli aura agent create \
     --name "$(jq -r .name agent/agent.json)" \
     --description "$(jq -r .description agent/agent.json)" \
     --system-prompt "$(jq -r .system_prompt agent/agent.json)" \
     --tools "$(jq -c .tools agent/agent.json)" \
     --dbid <instance-id> \
     --is-private \
     --rw --format json
   ```

   Save the agent `id`. If this fails with `Billing account is not funded` (neo4j-cli may report it as an auth/token error, so rerun with `--debug` to see it), the NODES credits weren't applied. Have the user check Billing in the Aura console.

4. **Check the graph.** An internal agent can't be invoked from neo4j-cli, so check the data instead. Expect 93 sessions:

   ```sh
   neo4j-cli query --credential nodes-2026 "MATCH (s:Session) RETURN count(s) AS sessions" --format toon
   ```

5. Tell the user the instance ID and the agent ID. Then tell them to chat with the agent in the Aura console: **Agents**, then click `nodes-2026-agent`.

## Bonus: use it from neo4j-cli and your coding agent (MCP)

The agent starts internal. To call it from neo4j-cli or over MCP, switch it to external with MCP enabled. External agents use the NODES credits. Both flags go in one update, because a private agent can't have MCP enabled:

```sh
neo4j-cli aura agent list --format toon   # the agent's id and mcp_endpoint_link
neo4j-cli aura agent update <agent-id> --is-private=false --is-mcp-enabled=true --rw
neo4j-cli aura agent invoke <agent-id> --input "Who is speaking about knowledge graphs for agents, and when?" --rw
```

Then add it to your coding agent, using its `mcp_endpoint_link` (or **⋯** → **Copy MCP server endpoint** in the Aura console):

```sh
claude mcp add --transport http nodes-2026 "https://mcp.neo4j.io/agent?project_id=<project-id>&agent_id=<agent-id>"
```

The first time it connects, you'll be asked to sign in with your Aura account. To make the agent internal again:

```sh
neo4j-cli aura agent update <agent-id> --is-private=true --is-mcp-enabled=false --rw
```

## Troubleshooting

### MCP connection returns `Forbidden` (403) even though you're signed in to the right Aura account

Aura MCP sign-in uses its own session on `aura-mcp.eu.auth0.com`, which is separate from the Aura console login. If that session belongs to a different Aura user, a client can be handed a token for that user without being asked to sign in, and `mcp.neo4j.io` then returns `Forbidden`. This happens even when the console shows the right account and the agent has MCP enabled.

Fix: open https://aura-mcp.eu.auth0.com/v2/logout in the browser (or clear cookies for `aura-mcp.eu.auth0.com`, or use a private window). Then reconnect (in Claude Code: `/mcp`, then re-authenticate `nodes-2026`) and sign in with the Aura account that owns the agent.
