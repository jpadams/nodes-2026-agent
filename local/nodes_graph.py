"""The NODES 2026 graph on Relate, for the local agents. Nothing in here knows about agent frameworks.

- Graph: the graph in an in-memory Relate database, loaded from data/*.csv with the same
  data/load.cypher Aura uses (well under a second). Its methods run the Cypher behind each
  kind of agent.json tool and return the result as JSON text for the model.
- The helpers below read agent/agent.json: the agent's spec, each tool's arguments as a JSON
  schema, and the text2cypher instructions.
- chat_loop(): the command line both agents share.

The *-relate-agent.py scripts build their agent and tools on top of this, each the way its
framework recommends.
"""

import json
import os
import re
import sys
import unicodedata
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
# LOAD CSV may read only from this directory. Set before relate opens a database.
os.environ["RELATE_LOAD_CSV_DIR"] = str(DATA)
import relate  # noqa: E402

MODEL = os.environ.get("OPENAI_MODEL", "gpt-5.4-mini")
VERBOSE = "--verbose" in sys.argv
MAX_ROWS = 25
# Text2Cypher answers questions like "list every speaker", so it may return the whole graph's
# worth of rows (~130 speakers, ~100 sessions). Past this, the model is told what it's missing.
MAX_TEXT2CYPHER_ROWS = 300

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
# Template parameters the local lookup compares with accent-folded names.
FOLDED_PARAMETERS = {"sessions-by-speaker": {"name"}}

FOLD_CHARS = str.maketrans({"ł": "l", "Ł": "L", "ø": "o", "Ø": "O", "đ": "d", "Đ": "D", "ß": "ss", "æ": "ae", "Æ": "AE"})


def fold(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text.translate(FOLD_CHARS))
    return "".join(c for c in decomposed if not unicodedata.combining(c))


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


class Graph:
    """The NODES graph in an in-memory Relate database."""

    def __init__(self):
        # The embedded database must stay on the thread that opened it, but agents call tools
        # from worker threads. So one dedicated thread opens it and runs every query, in turn.
        self.thread = ThreadPoolExecutor(max_workers=1)
        self.db = self.thread.submit(relate.connect, ":memory:").result()
        self.load()

    def execute(self, cypher: str, params: dict | None = None) -> list[dict]:
        def work():
            return [{k: r[k] for k in r.keys()} for r in self.db.execute(cypher, params)]

        return self.thread.submit(work).result()

    def close(self):
        """Close and drop the database on its own thread."""

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

    def run_template(self, tool: dict, **params: str) -> str:
        """A cypherTemplate tool: its Cypher (or the local version) with the model's parameters."""
        folded = FOLDED_PARAMETERS.get(tool["name"], set())
        params = {k: fold(v) if k in folded else v for k, v in params.items()}
        template = LOCAL_TEMPLATES.get(tool["name"], tool["config"]["template"])
        return to_json(self.run(template, **params))

    def vector_search(self, tool: dict, embedding: list[float]) -> str:
        """A similaritySearch tool: query its vector index, then run its post-processing Cypher."""
        config = tool["config"]
        cypher = (
            f"CALL db.index.vector.queryNodes('{config['index']}', {int(config['top_k'])}, $embedding) "
            f"YIELD node, score {config['post_processing_cypher']}"
        )
        return to_json(self.run(cypher, embedding=embedding))

    def run_generated(self, cypher: str) -> str:
        """Model-written Cypher, for text2cypher, with up to MAX_TEXT2CYPHER_ROWS rows.
        Raises the database's error if the query fails, so the model can fix it."""
        rows = self.run(cypher, limit=None)
        result = {"cypher": cypher, "rows": rows[:MAX_TEXT2CYPHER_ROWS]}
        if len(rows) > MAX_TEXT2CYPHER_ROWS:
            result["note"] = f"Showing the first {MAX_TEXT2CYPHER_ROWS} of {len(rows)} rows."
        return to_json(result)


def load_agent_spec() -> dict:
    """agent/agent.json: the Aura Agent's name, system prompt, and tools."""
    return json.loads((ROOT / "agent" / "agent.json").read_text())


def tools_of_type(spec: dict, kind: str) -> list[dict]:
    return [t for t in spec["tools"] if t["type"] == kind and t.get("enabled", True)]


def template_schema(tool: dict, expected: list[str]) -> dict:
    """A cypherTemplate tool's parameters as a JSON schema, checked against the parameter
    names the agent's tool function takes, so a change to agent.json fails here, loudly."""
    names = [p["name"] for p in tool["config"]["parameters"]]
    if names != expected:
        raise ValueError(f"{tool['name']} takes {names} in agent.json; the local tool expects {expected}")
    return string_schema({p["name"]: p["description"] for p in tool["config"]["parameters"]})


def string_schema(arguments: dict[str, str]) -> dict:
    """A JSON schema for required string arguments (name -> description)."""
    return {
        "type": "object",
        "properties": {name: {"type": "string", "description": d} for name, d in arguments.items()},
        "required": list(arguments),
    }


SIMILARITY_SCHEMA = string_schema({"query": "What to search for, in natural language."})
TEXT2CYPHER_SCHEMA = string_schema({"question": "The question to answer from the graph."})


def text2cypher_instructions(tool: dict) -> str:
    """The system prompt for writing Cypher, from the text2cypher tool's schema description."""
    return (
        "Write one read-only Cypher query that answers the question, for this graph:\n"
        f"{tool['description']}\n"
        "Return every row the question asks for: add a LIMIT only when it asks for a few, like \"the top 5\". "
        "Use only the labels, relationships, and properties above. "
        "Reply with the Cypher only, no explanation and no code fences."
    )


def cypher_from_reply(reply: str) -> str:
    """The Cypher in a model's reply, without code fences if it added them anyway."""
    return re.sub(r"^```\w*\s*|\s*```$", "", reply.strip())


def chat_loop(spec: dict, ask: Callable[[str], str]) -> None:
    """Answer the question on the command line and exit, or chat until Ctrl-C or Ctrl-D.
    ask(question) returns the agent's answer, keeping the conversation for follow-ups."""
    args = [a for a in sys.argv[1:] if a != "--verbose"]
    try:
        if args:
            print(f"\nagent> {ask(' '.join(args)).strip()}\n")
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
                print(f"\nagent> {ask(question).strip()}\n")
    except KeyboardInterrupt:  # Ctrl-C, at the prompt or mid-answer
        print()
