# /// script
# requires-python = ">=3.10"
# dependencies = ["langchain==1.4.5", "langchain-openai==1.7.1", "neo4j==6.4.0"]
# ///
"""Chat with a local copy of the NODES 2026 Aura Agent, backed by CypherLite.

The system prompt and tools come from agent/agent.json, the same definition Aura uses:
- cypherTemplate tools run their Cypher with the model's parameters.
- similaritySearch embeds the question, queries the vector index, then runs the post-processing Cypher.
- text2cypher asks the model to write a read-only Cypher query from the tool's schema description.

Needs OPENAI_API_KEY, and local/nodes.db from `uv run local/setup.py`.
Usage: uv run local/chat.py [--verbose]
"""

import atexit
import json
import os
import re
import secrets
import shutil
import socket
import subprocess
import sys
import time
import unicodedata
from pathlib import Path

from langchain.agents import create_agent
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import StructuredTool
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from neo4j import GraphDatabase
from pydantic import Field, create_model

ROOT = Path(__file__).resolve().parent.parent
DB = Path(__file__).resolve().parent / "nodes.db"
MODEL = os.environ.get("OPENAI_MODEL", "gpt-5.4-mini")
MAX_ROWS = 25
VERBOSE = "--verbose" in sys.argv

# CypherLite's full-text index matches single whole words only (no wildcards, OR/AND, or
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
RETURN sp.name AS speaker, s.title AS title, times ORDER BY speaker, title LIMIT 25
""",
}

FOLD_CHARS = str.maketrans({"ł": "l", "Ł": "L", "ø": "o", "Ø": "O", "đ": "d", "Đ": "D", "ß": "ss", "æ": "ae", "Æ": "AE"})


def fold(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text.translate(FOLD_CHARS))
    return "".join(c for c in decomposed if not unicodedata.combining(c))


def start_cypherlite() -> tuple[str, tuple[str, str]]:
    """Serve local/nodes.db read-only over Bolt on a free port, with a one-off password."""
    if not shutil.which("cypherlite"):
        sys.exit("cypherlite isn't installed: see https://neo4j-labs.github.io/cypherlite/")
    if not DB.exists():
        sys.exit("local/nodes.db doesn't exist yet. Run: uv run local/setup.py")
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    password = secrets.token_hex(16)
    proc = subprocess.Popen(
        ["cypherlite", "serve", "-d", str(DB), "--bolt", str(port), "--auth", f"neo4j:{password}", "--read-only"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )
    atexit.register(proc.terminate)
    for _ in range(50):
        if proc.poll() is not None:
            sys.exit(f"cypherlite serve exited:\n{proc.stderr.read()}")
        try:
            socket.create_connection(("127.0.0.1", port), timeout=0.2).close()
            return f"bolt://127.0.0.1:{port}", ("neo4j", password)
        except OSError:
            time.sleep(0.1)
    sys.exit("cypherlite serve didn't start")


class Graph:
    def __init__(self, uri: str, auth: tuple[str, str]):
        self.driver = GraphDatabase.driver(uri, auth=auth)
        atexit.register(self.driver.close)

    def run(self, cypher: str, **params) -> list[dict]:
        records, _, _ = self.driver.execute_query(cypher, params)
        return [r.data() for r in records[:MAX_ROWS]]


def to_json(rows) -> str:
    return json.dumps(rows, default=str, ensure_ascii=False)


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
        f"Return at most {MAX_ROWS} rows. Use only the labels, relationships, and properties above. "
        "Reply with the Cypher only, no explanation and no code fences."
    )

    def run(question: str) -> str:
        messages = [("system", instructions), ("human", question)]
        for _ in range(2):
            cypher = re.sub(r"^```\w*\s*|\s*```$", "", llm.invoke(messages).content.strip())
            try:
                return to_json({"cypher": cypher, "rows": graph.run(cypher)})
            except Exception as err:  # Retry once with the error, the way a person would.
                messages += [("ai", cypher), ("human", f"That failed: {err}. Fix the query.")]
        return to_json({"cypher": cypher, "error": str(err)})

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


def main() -> None:
    if not os.environ.get("OPENAI_API_KEY"):
        sys.exit("Set OPENAI_API_KEY first.")
    spec = json.loads((ROOT / "agent" / "agent.json").read_text())
    graph = Graph(*start_cypherlite())
    tools = [BUILDERS[t["type"]](graph, t) for t in spec["tools"] if t.get("enabled", True)]
    agent = create_agent(model=ChatOpenAI(model=MODEL), tools=tools, system_prompt=spec["system_prompt"])

    print(f"{spec['name']} (local, {MODEL}). {spec['description']}")
    print("Ask a question, or press Ctrl-D to quit.\n")
    history = []
    while True:
        try:
            question = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not question:
            continue
        history.append(HumanMessage(question))
        result = agent.invoke({"messages": history})
        new = result["messages"][len(history) :]
        if VERBOSE:
            show_steps(new)
        history = result["messages"]
        print(f"\nagent> {history[-1].content}\n")


if __name__ == "__main__":
    main()
