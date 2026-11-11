# /// script
# requires-python = ">=3.10"
# dependencies = ["strands-agents[openai]==1.63.0", "openai==3.45.0", "relate-db==0.1.2"]
#
# [tool.uv]
# # Relate isn't on PyPI yet: its wheels come from the GitHub release (see local/README.md).
# # The path is relative to where you run uv, so run from the repo root.
# find-links = ["local/wheels"]
# ///
"""Chat with a local copy of the NODES 2026 Aura Agent on Strands Agents, with Relate embedded in-process.

On start, the graph is loaded into memory from data/*.csv with the same data/load.cypher Aura
uses (well under a second). The system prompt and tools come from agent/agent.json:
- cypherTemplate tools run their Cypher with the model's parameters.
- similaritySearch embeds the question, queries the vector index, then runs the post-processing Cypher.
- text2cypher asks the model to write a read-only Cypher query from the tool's schema description.

Needs OPENAI_API_KEY.
Usage: uv run local/strands-relate-agent.py [--verbose] ["question"]
  With a question: answer it and exit. Without: chat until Ctrl-C or Ctrl-D.
"""

import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor
import unicodedata
from pathlib import Path

from openai import OpenAI
from strands import Agent
from strands.models.openai import OpenAIModel
from strands.tools import PythonAgentTool

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
# LOAD CSV may read only from this directory. Set before relate opens a database.
os.environ["RELATE_LOAD_CSV_DIR"] = str(DATA)
import relate  # noqa: E402

MODEL = os.environ.get("OPENAI_MODEL", "gpt-5.4-mini")
MAX_ROWS = 25
# Text2Cypher answers questions like "list every speaker", so it may return the whole graph's
# worth of rows (~130 speakers, ~100 sessions). Past this, the model is told what it's missing.
MAX_TEXT2CYPHER_ROWS = 300
VERBOSE = "--verbose" in sys.argv

# Relate's full-text index matches single whole words only (no wildcards, OR/AND, or
# accent folding), so the speaker lookup matches each word of the input against the
# accent-folded names instead. With ~120 speakers a scan is instant.
LOCAL_TEMPLATES = {
    "sessions-by-speaker": """
WITH [w IN split(toLower($name), ' ') WHERE w <> ''] AS words WHERE size(words) > 0
MATCH (sp:Speaker)
WHERE all(w IN words WHERE toLower(coalesce(sp.nameFolded, sp.name)) CONTAINS w)
MATCH (sp)-[:PRESENTS]->(s:Session)
OPTIONAL MATCH (s)-[:HAS_SLOT]->(sl:Slot)
WITH sp, s, sl ORDER BY sl.startTime
WITH sp, s, collect(sl.displayTime) AS times
OPTIONAL MATCH (s)<-[:PRESENTS]-(co:Speaker)
WITH sp, s, times, collect(co.name) AS speakers
RETURN sp.name AS speaker, s.title AS title, speakers, times ORDER BY speaker, title LIMIT 25
""",
}

FOLD_CHARS = str.maketrans({"ł": "l", "Ł": "L", "ø": "o", "Ø": "O", "đ": "d", "Đ": "D", "ß": "ss", "æ": "ae", "Æ": "AE"})


def fold(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text.translate(FOLD_CHARS))
    return "".join(c for c in decomposed if not unicodedata.combining(c))


class Graph:
    """The NODES graph in an in-memory Relate database."""

    def __init__(self):
        # The embedded database must stay on the thread that opened it, but the agent calls tools
        # from worker threads. So one dedicated thread opens it and runs every query, in turn.
        self.thread = ThreadPoolExecutor(max_workers=1)
        self.db = self.thread.submit(relate.connect, ":memory:").result()
        self.load()

    def execute(self, cypher: str, params: dict | None = None) -> list[dict]:
        def work():
            return [{k: r[k] for k in r.keys()} for r in self.db.execute(cypher, params)]

        return self.thread.submit(work).result()

    def close(self):
        """Close and drop the database on its own thread, as it requires."""

        def work():
            self.db.close()
            del self.db

        self.thread.submit(work).result()
        self.thread.shutdown()

    def load(self):
        """Run data/load.cypher, adjusted for Relate: local CSV paths and a plain SET for
        embeddings (there's no db.create.setNodeVectorProperty). Then add accent-folded speaker names."""
        script = (DATA / "load.cypher").read_text()
        script = re.sub(r"https://raw\.githubusercontent\.com/[^']*/data/", "", script)
        script = re.sub(r"CALL db\.create\.setNodeVectorProperty\(c, 'embedding', (.*)\);", r"SET c.embedding = \1;", script)
        # Statements end with ; at the end of a line, as in neo4j-cli.
        for statement in re.split(r";[ \t]*$", script, flags=re.M):
            if any(line.strip() and not line.strip().startswith("//") for line in statement.splitlines()):
                self.execute(statement)
        names = [r["name"] for r in self.run("MATCH (p:Speaker) RETURN p.name AS name", limit=None)]
        self.execute(
            "UNWIND $rows AS row MATCH (p:Speaker {name: row.name}) SET p.nameFolded = row.folded",
            {"rows": [{"name": n, "folded": fold(n)} for n in names]},
        )

    def run(self, cypher: str, limit: int | None = MAX_ROWS, **params) -> list[dict]:
        return self.execute(cypher, params)[:limit]


def plain(value):
    """Relate's nodes, relationships, and paths as JSON-friendly values, like the Neo4j driver's."""
    if isinstance(value, relate.Node):
        return {"labels": list(value.labels), **value.properties}
    if isinstance(value, relate.Relationship):
        return {"type": value.type, **value.properties}
    if isinstance(value, relate.Path):
        return {"nodes": [plain(n) for n in value.nodes], "relationships": [plain(r) for r in value.relationships]}
    return str(value)


def to_json(rows) -> str:
    return json.dumps(rows, default=plain, ensure_ascii=False)


def make_tool(spec: dict, properties: dict[str, str], run) -> PythonAgentTool:
    """A Strands tool named and described as in agent.json, taking string arguments
    (name -> description) and returning run(**arguments) to the model as text."""

    def call(tool_use, **_) -> dict:
        text = run(**tool_use["input"])
        return {"toolUseId": tool_use["toolUseId"], "status": "success", "content": [{"text": text}]}

    schema = {
        "type": "object",
        "properties": {name: {"type": "string", "description": d} for name, d in properties.items()},
        "required": list(properties),
    }
    tool_spec = {"name": spec["name"], "description": spec["description"], "inputSchema": {"json": schema}}
    return PythonAgentTool(spec["name"], tool_spec, call)


def template_tool(graph: Graph, client: OpenAI, spec: dict) -> PythonAgentTool:
    config = spec["config"]
    template = LOCAL_TEMPLATES.get(spec["name"], config["template"])

    def run(**params) -> str:
        if spec["name"] == "sessions-by-speaker":
            params = {k: fold(v) for k, v in params.items()}
        return to_json(graph.run(template, **params))

    return make_tool(spec, {p["name"]: p["description"] for p in config["parameters"]}, run)


def similarity_tool(graph: Graph, client: OpenAI, spec: dict) -> PythonAgentTool:
    config = spec["config"]
    cypher = (
        f"CALL db.index.vector.queryNodes('{config['index']}', {int(config['top_k'])}, $embedding) "
        f"YIELD node, score {config['post_processing_cypher']}"
    )

    def run(query: str) -> str:
        embedding = client.embeddings.create(model=config["model"], input=query).data[0].embedding
        return to_json(graph.run(cypher, embedding=embedding))

    return make_tool(spec, {"query": "What to search for, in natural language."}, run)


def text2cypher_tool(graph: Graph, client: OpenAI, spec: dict) -> PythonAgentTool:
    instructions = (
        "Write one read-only Cypher query that answers the question, for this graph:\n"
        f"{spec['description']}\n"
        "Return every row the question asks for: add a LIMIT only when it asks for a few, like \"the top 5\". "
        "Use only the labels, relationships, and properties above. "
        "Reply with the Cypher only, no explanation and no code fences."
    )

    def run(question: str) -> str:
        messages = [{"role": "system", "content": instructions}, {"role": "user", "content": question}]
        for _ in range(2):
            reply = client.chat.completions.create(model=MODEL, messages=messages).choices[0].message.content
            cypher = re.sub(r"^```\w*\s*|\s*```$", "", reply.strip())
            try:
                rows = graph.run(cypher, limit=None)
                result = {"cypher": cypher, "rows": rows[:MAX_TEXT2CYPHER_ROWS]}
                if len(rows) > MAX_TEXT2CYPHER_ROWS:
                    result["note"] = f"Showing the first {MAX_TEXT2CYPHER_ROWS} of {len(rows)} rows."
                return to_json(result)
            except Exception as err:  # Retry once with the error, the way a person would.
                error = err
                messages += [
                    {"role": "assistant", "content": cypher},
                    {"role": "user", "content": f"That failed: {err}. Fix the query."},
                ]
        return to_json({"cypher": cypher, "error": str(error)})

    return make_tool(spec, {"question": "The question to answer from the graph."}, run)


BUILDERS = {"cypherTemplate": template_tool, "similaritySearch": similarity_tool, "text2cypher": text2cypher_tool}


def show_steps(messages) -> None:
    for m in messages:
        for block in m["content"]:
            if "toolUse" in block:
                call = block["toolUse"]
                print(f"  → {call['name']}({json.dumps(call['input'], ensure_ascii=False)})")
            elif "toolResult" in block:
                text = " ".join(c.get("text", "") for c in block["toolResult"]["content"])
                print(f"  ← {text[:300]}{'…' if len(text) > 300 else ''}")


def ask(agent: Agent, question: str) -> None:
    """Answer one question, printing the answer. The agent keeps the conversation."""
    start = len(agent.messages)
    result = agent(question)
    if VERBOSE:
        show_steps(agent.messages[start:])
    print(f"\nagent> {str(result).strip()}\n")


def main() -> None:
    if not os.environ.get("OPENAI_API_KEY"):
        sys.exit("Set OPENAI_API_KEY first.")
    args = [a for a in sys.argv[1:] if a != "--verbose"]
    spec = json.loads((ROOT / "agent" / "agent.json").read_text())
    graph = Graph()
    try:
        client = OpenAI()
        tools = [BUILDERS[t["type"]](graph, client, t) for t in spec["tools"] if t.get("enabled", True)]
        # callback_handler=None: print only the final answer, not the streamed tokens.
        model = OpenAIModel(model_id=MODEL)
        agent = Agent(model=model, tools=tools, system_prompt=spec["system_prompt"], callback_handler=None)

        if args:
            ask(agent, " ".join(args))
            return

        print(f"{spec['name']} (local, {MODEL}). {spec['description']}")
        print("Ask a question, or press Ctrl-D to quit.\n")
        while True:
            try:
                question = input("you> ").strip()
            except EOFError:
                print()
                return
            if question:
                ask(agent, question)
    except KeyboardInterrupt:  # Ctrl-C, at the prompt or mid-answer
        print()
    finally:
        graph.close()


if __name__ == "__main__":
    main()
