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

- [neo4j-cli](https://neo4j.sh): `curl -sSfL https://neo4j.sh/install.sh | bash` (or `brew install neo4j-labs/tap/neo4j-cli`, or `npm install -g @neo4j-labs/cli`)
- An Aura account (sign up with the NODES 2026 link from the walkthrough), plus an Aura API key stored with:

```sh
neo4j-cli credential aura-client add --name nodes --env <path-to-downloaded-credentials-file>
neo4j-cli aura organization list
neo4j-cli aura project list --organization-id <org-id>
neo4j-cli aura workspace use <org-id>/<project-id>
```

- Agent skills: `neo4j-cli skill install`

## Agent instructions

Coding agents: follow these steps in order. Write commands need `--rw`. Prefer `--format toon` for reads.

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

3. **Create the Aura Agent from `agent/agent.json`, with MCP enabled.** `agent create` has no file flag, so pass the fields with `jq`:

   ```sh
   neo4j-cli aura agent create \
     --name "$(jq -r .name agent/agent.json)" \
     --description "$(jq -r .description agent/agent.json)" \
     --system-prompt "$(jq -r .system_prompt agent/agent.json)" \
     --tools "$(jq -c .tools agent/agent.json)" \
     --dbid <instance-id> \
     --is-mcp-enabled \
     --rw --format json
   ```

   Save the agent `id`. If this fails with `Billing account is not funded` (neo4j-cli may report it as an auth/token error, so rerun with `--debug` to see it), the NODES credits weren't applied. Have the user check Billing in the Aura console.

4. **Invoke it:**

   ```sh
   neo4j-cli aura agent invoke <agent-id> --input "What sessions at NODES 2026 cover GraphRAG?" --rw
   ```

5. Tell the user the instance ID, the agent ID, and the agent's MCP server endpoint (the `mcp_endpoint_link` field in `neo4j-cli aura agent list --format toon`). Then tell them to open the agent's **⋯** menu in the Aura console and click **Copy MCP server endpoint** for the submission form.

## Use it from your coding agent (MCP)

```sh
claude mcp add --transport http nodes-2026 "https://mcp.neo4j.io/agent?project_id=<project-id>&agent_id=<agent-id>"
```

The first time it connects, you'll be asked to sign in with your Aura account.

## Troubleshooting

### MCP connection returns `Forbidden` (403) even though you're signed in to the right Aura account

Aura MCP sign-in uses its own session on `aura-mcp.eu.auth0.com`, which is separate from the Aura console login. If that session belongs to a different Aura user, a client can be handed a token for that user without being asked to sign in, and `mcp.neo4j.io` then returns `Forbidden`. This happens even when the console shows the right account and the agent has MCP enabled.

Fix: open https://aura-mcp.eu.auth0.com/v2/logout in the browser (or clear cookies for `aura-mcp.eu.auth0.com`, or use a private window). Then reconnect (in Claude Code: `/mcp`, then re-authenticate `nodes-2026`) or resubmit on the website, and sign in with the Aura account that owns the agent.
