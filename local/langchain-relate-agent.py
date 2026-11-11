# /// script
# requires-python = ">=3.10"
# dependencies = ["langchain==1.4.5", "langchain-openai==1.7.1", "relate-db==0.1.2"]
#
# [tool.uv]
# # Relate isn't on PyPI yet: its wheels come from the GitHub release (see local/README.md).
# # The path is relative to where you run uv, so run from the repo root.
# find-links = ["local/wheels"]
# ///
"""Chat with a local copy of the NODES 2026 Aura Agent on LangChain, with Relate embedded in-process.

On start, the graph is loaded into memory from data/*.csv with the same data/load.cypher Aura
uses (well under a second). The system prompt and tools come from agent/agent.json:
- cypherTemplate tools run their Cypher with the model's parameters.
- similaritySearch embeds the question, queries the vector index, then runs the post-processing Cypher.
- text2cypher asks the model to write a read-only Cypher query from the tool's schema description.

Needs OPENAI_API_KEY.
Usage: uv run local/langchain-relate-agent.py [--verbose] ["question"]
  With a question: answer it and exit. Without: chat until Ctrl-C or Ctrl-D.
"""

import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor
import unicodedata
from pathlib import Path

from langchain.agents import create_agent
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import StructuredTool
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from pydantic import Field, create_model

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


def template_tool(graph: Graph, spec: dict) -> StructuredTool:
    config = spec["config"]
    template = LOCAL_TEMPLATES.get(spec["name"], config["template"])
    fields = {p["name"]: (str, Field(description=p["description"])) for p in config["parameters"]}

    def run(**params) -> str:
        if spec["name"] == "sessions-by-speaker":
            params = {k: fold(v) for k, v in params.items()}
        return to_json(graph.run(template, **params))

    return StructuredTool.from_function(
        func=run,
        name=spec["name"],
        description=spec["description"],
        args_schema=create_model(f"{spec['name']}_args", **fields),
    )


def similarity_tool(graph: Graph, spec: dict) -> StructuredTool:
    config = spec["config"]
    embeddings = OpenAIEmbeddings(model=config["model"])
    cypher = (
        f"CALL db.index.vector.queryNodes('{config['index']}', {int(config['top_k'])}, $embedding) "
        f"YIELD node, score {config['post_processing_cypher']}"
    )

    def run(query: str) -> str:
        return to_json(graph.run(cypher, embedding=embeddings.embed_query(query)))

    return StructuredTool.from_function(
        func=run,
        name=spec["name"],
        description=spec["description"],
        args_schema=create_model(f"{spec['name']}_args", query=(str, Field(description="What to search for, in natural language."))),
    )


def text2cypher_tool(graph: Graph, spec: dict) -> StructuredTool:
    llm = ChatOpenAI(model=MODEL)
    instructions = (
        "Write one read-only Cypher query that answers the question, for this graph:\n"
        f"{spec['description']}\n"
        "Return every row the question asks for: add a LIMIT only when it asks for a few, like \"the top 5\". "
        "Use only the labels, relationships, and properties above. "
        "Reply with the Cypher only, no explanation and no code fences."
    )

    def run(question: str) -> str:
        messages = [("system", instructions), ("human", question)]
        for _ in range(2):
            cypher = re.sub(r"^```\w*\s*|\s*```$", "", llm.invoke(messages).content.strip())
            try:
                rows = graph.run(cypher, limit=None)
                result = {"cypher": cypher, "rows": rows[:MAX_TEXT2CYPHER_ROWS]}
                if len(rows) > MAX_TEXT2CYPHER_ROWS:
                    result["note"] = f"Showing the first {MAX_TEXT2CYPHER_ROWS} of {len(rows)} rows."
                return to_json(result)
            except Exception as err:  # Retry once with the error, the way a person would.
                error = err
                messages += [("ai", cypher), ("human", f"That failed: {err}. Fix the query.")]
        return to_json({"cypher": cypher, "error": str(error)})

    return StructuredTool.from_function(
        func=run,
        name=spec["name"],
        description=spec["description"],
        args_schema=create_model(f"{spec['name']}_args", question=(str, Field(description="The question to answer from the graph."))),
    )


BUILDERS = {"cypherTemplate": template_tool, "similaritySearch": similarity_tool, "text2cypher": text2cypher_tool}


def show_steps(messages) -> None:
    for m in messages:
        if isinstance(m, AIMessage):
            for call in m.tool_calls:
                print(f"  → {call['name']}({json.dumps(call['args'], ensure_ascii=False)})")
        elif isinstance(m, ToolMessage):
            text = str(m.content)
            print(f"  ← {text[:300]}{'…' if len(text) > 300 else ''}")


def ask(agent, history: list, question: str) -> list:
    """Answer one question, printing the answer; returns the conversation so far."""
    history = history + [HumanMessage(question)]
    result = agent.invoke({"messages": history})
    if VERBOSE:
        show_steps(result["messages"][len(history) :])
    print(f"\nagent> {result['messages'][-1].content}\n")
    return result["messages"]


def main() -> None:
    if not os.environ.get("OPENAI_API_KEY"):
        sys.exit("Set OPENAI_API_KEY first.")
    args = [a for a in sys.argv[1:] if a != "--verbose"]
    spec = json.loads((ROOT / "agent" / "agent.json").read_text())
    graph = Graph()
    try:
        tools = [BUILDERS[t["type"]](graph, t) for t in spec["tools"] if t.get("enabled", True)]
        agent = create_agent(model=ChatOpenAI(model=MODEL), tools=tools, system_prompt=spec["system_prompt"])

        if args:
            ask(agent, [], " ".join(args))
            return

        print(f"{spec['name']} (local, {MODEL}). {spec['description']}")
        print("Ask a question, or press Ctrl-D to quit.\n")
        history = []
        while True:
            try:
                question = input("you> ").strip()
            except EOFError:
                print()
                return
            if question:
                history = ask(agent, history, question)
    except KeyboardInterrupt:  # Ctrl-C, at the prompt or mid-answer
        print()
    finally:
        graph.close()


if __name__ == "__main__":
    main()
