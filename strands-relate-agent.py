# /// script
# requires-python = ">=3.10"
# dependencies = ["strands-agents[openai]==1.64.0", "openai==3.49.0", "relate-db==1.0.0"]
# ///
"""A NODES 2026 conference guide: a Strands agent with tools that query a Relate graph.

Needs OPENAI_API_KEY.
Usage: uv run strands-relate-agent.py ["question"]
  With a question: answer it and exit. Without: chat until Ctrl-C or Ctrl-D.
"""

import difflib
import json
import os
import sys
import unicodedata
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

if not os.environ.get("OPENAI_API_KEY"):
    sys.exit("OPENAI_API_KEY is not set. Run: export OPENAI_API_KEY=sk-...")

import relate
from openai import OpenAI
from strands import Agent, tool
from strands.models.openai import OpenAIModel

# --- The graph -------------------------------------------------------------------------------
# The whole graph, with its vector and full-text indexes, is one Relate file, opened read-only.
# A Relate connection only works on the thread that opened it, and Strands runs tools on
# worker threads, so one dedicated thread opens the database and runs every query.

db_thread = ThreadPoolExecutor(max_workers=1)
db = db_thread.submit(relate.connect, str(Path(__file__).with_name("nodes-2026.rl")), read_only=True).result()


def run(cypher: str, **params) -> list[dict]:
    return db_thread.submit(lambda: [row.data() for row in db.execute(cypher, params)]).result()


def to_json(value) -> str:
    """Query results as JSON for the model, with nodes and relationships as their properties."""
    return json.dumps(value, default=lambda v: getattr(v, "properties", str(v)), ensure_ascii=False)


def fold(text: str) -> str:
    """Lowercase text without accents, like each speaker's nameFolded: "Gołąb" becomes "golab"."""
    text = unicodedata.normalize("NFKD", text.lower().translate(str.maketrans({"ł": "l", "ø": "o", "đ": "d", "ß": "ss", "æ": "ae"})))
    return "".join(c for c in text if not unicodedata.combining(c))


NAME_WORDS = sorted({w for r in run("MATCH (p:Speaker) RETURN p.nameFolded AS n") for w in r["n"].lower().split()})

# --- The tools -------------------------------------------------------------------------------
# Strands turns each function into a tool: its name, its docstring as the description, and
# its arguments (with the descriptions under Args) as the input schema.

SPEAKER_SESSIONS = """
MATCH (sp:Speaker) WHERE all(w IN split($name, ' ') WHERE toLower(sp.nameFolded) CONTAINS w)
MATCH (sp)-[:PRESENTS]->(s:Session)
OPTIONAL MATCH (s)-[:HAS_SLOT]->(sl:Slot)
WITH sp, s, sl ORDER BY sl.startTime
WITH sp, s, collect(sl.displayTime) AS times
OPTIONAL MATCH (s)<-[:PRESENTS]-(co:Speaker)
RETURN sp.name AS speaker, s.title AS title, collect(co.name) AS speakers, times
ORDER BY speaker, title LIMIT 25
"""


@tool
def sessions_by_speaker(name: str) -> str:
    """List the sessions a speaker is presenting, with their time slots, by speaker name or
    part of it. Accents and small typos are OK: Eifrem matches Eifrém, and Kolleger matches Kollegger.

    Args:
        name: Speaker name or part of it.
    """
    words = fold(name).split()
    if not words:
        return "[]"
    rows = run(SPEAKER_SESSIONS, name=" ".join(words))
    if not rows:  # Nobody matched: try again with each word swapped for the closest word in a speaker's name.
        close = [difflib.get_close_matches(w, NAME_WORDS, n=1, cutoff=0.8) or [w] for w in words]
        rows = run(SPEAKER_SESSIONS, name=" ".join(c[0] for c in close))
    return to_json(rows)


@tool
def sessions_about(name: str) -> str:
    """List sessions tagged with or mentioning a topic, technology, concept, company, or
    industry, by (partial) name, e.g. GraphRAG, LangChain, Agent Memory, Adobe, healthcare.

    Args:
        name: Topic, technology, concept, organization, or domain name, or part of it.
    """
    return to_json(run("""
        MATCH (s:Session)-[:HAS_TOPIC|HAS_USE_CASE|MENTIONS]->(x) WHERE toLower(x.name) CONTAINS toLower($name)
        WITH DISTINCT s
        OPTIONAL MATCH (s)<-[:PRESENTS]-(sp:Speaker)
        OPTIONAL MATCH (s)-[:HAS_SLOT]->(sl:Slot)
        WITH s, sp, sl ORDER BY sl.startTime
        WITH s, collect(DISTINCT sp.name) AS speakers, collect(DISTINCT sl.displayTime) AS times, min(sl.startTime) AS first
        RETURN s.title AS title, speakers, times ORDER BY first LIMIT 25
        """, name=name))


openai = OpenAI()


@tool
def find_sessions(query: str) -> str:
    """Semantic search over session titles, speakers, and descriptions to find sessions about a topic.

    Args:
        query: What to search for, in natural language.
    """
    embedding = openai.embeddings.create(model="text-embedding-3-small", input=query).data[0].embedding
    return to_json(run("""
        CALL db.index.vector.queryNodes('session_chunk_embeddings', 5, $embedding) YIELD node, score
        MATCH (node)-[:FROM_DOCUMENT]->(s:Session)
        OPTIONAL MATCH (s)<-[:PRESENTS]-(sp:Speaker)
        OPTIONAL MATCH (s)-[:HAS_SLOT]->(sl:Slot)
        WITH s, score, sp, sl ORDER BY sl.startTime
        WITH s, score, collect(DISTINCT sp.name) AS speakers, collect(DISTINCT sl.displayTime) AS times
        RETURN s.title AS title, speakers, times, s.description AS description, score ORDER BY score DESC
        """, embedding=embedding))


model = OpenAIModel(model_id=os.environ.get("OPENAI_MODEL", "gpt-5.4-mini"))

CYPHER_WRITER_PROMPT = """Write one read-only Cypher query that answers the question, for this graph:
(:Speaker {name})-[:PRESENTS]->(:Session)-[:HAS_SLOT]->(:Slot {startTime, endTime, durationMinutes, displayTime})
(:Session)-[:IN_TRACK]->(:Track {name})
(:Session)-[:HAS_TOPIC]->(:Topic {name})
(:Session)-[:HAS_USE_CASE]->(:UseCase {name})
(:Session)-[:MENTIONS]->(:Technology|Concept|Organization|Domain {name})
Session has title, description, levels and formats (lists). Slot.startTime and endTime are UTC
datetimes for filtering and sorting; return Slot.displayTime to show a time.
Return every row the question asks for: add a LIMIT only when it asks for a few, like "the top 5".
Reply with the Cypher only, no explanation and no code fences."""


@tool
def ask_nodes_graph(question: str) -> str:
    """Answer any other question about NODES 2026 sessions, speakers, tracks, topics, use cases,
    and time slots, by writing a Cypher query. Good for counts, lists, and schedules.

    Args:
        question: The question to answer from the graph.
    """
    # Agents as tools: a specialist agent writes the Cypher. If the query fails, it sees the
    # error and gets one more try.
    cypher_writer = Agent(model=model, system_prompt=CYPHER_WRITER_PROMPT, callback_handler=None)
    prompt = question
    for _ in range(2):
        cypher = str(cypher_writer(prompt)).strip().removeprefix("```cypher").strip("`\n ")
        try:
            return to_json({"cypher": cypher, "rows": run(cypher)[:300]})
        except relate.Error as err:
            error = str(err)
            prompt = f"That failed: {error}. Fix the query."
    return to_json({"cypher": cypher, "error": error})


# --- The agent -------------------------------------------------------------------------------

agent = Agent(
    model=model,
    tools=[sessions_by_speaker, sessions_about, find_sessions, ask_nodes_graph],
    system_prompt="""You are a helpful guide to the NODES 2026 conference. Answer using only the graph.
Always include session titles, speakers, and times when relevant. Write times exactly as the tools
return them, like "Thu 12 Nov, 02:00 UTC". Some sessions are broadcast in several time slots; list
them all. If the graph doesn't contain the answer, say so. Refer to speakers by name or as "they";
don't guess anyone's pronouns.""",
)

if __name__ == "__main__":
    try:
        if sys.argv[1:]:
            agent(" ".join(sys.argv[1:]))
        else:
            print("Ask about NODES 2026 sessions, speakers, and schedule. Press Ctrl-D to quit.")
            while True:
                if question := input("\nyou> ").strip():
                    agent(question)
                    print()
    except (EOFError, KeyboardInterrupt):
        pass
    finally:
        db_thread.submit(db.close).result()  # Close the database on its own thread.
    print()
