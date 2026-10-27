"""Build the NODES 2026 session graph in a scratch Neo4j database.

Each talk becomes one Session node (the neo4j-graphrag "document") with one
Chunk holding its title, speakers and description. The chunk is embedded and
entities are extracted from it. Columns from the sheet (speakers, track, topics,
use cases, time slots) are loaded directly, without the LLM.

    uv run --env-file .env build_graph.py talks.xlsx --reset

Needs OPENAI_API_KEY, NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD.
--reset deletes everything in the target database first.
"""

import argparse
import asyncio
import os
import re
import unicodedata
from dataclasses import dataclass, field
from datetime import datetime, timezone

import neo4j
import openpyxl
from neo4j_graphrag.components.resolver import SinglePropertyExactMatchResolver
from neo4j_graphrag.components.text_splitters.fixed_size_splitter import FixedSizeSplitter
from neo4j_graphrag.components.types import LexicalGraphConfig
from neo4j_graphrag.embeddings import OpenAIEmbeddings
from neo4j_graphrag.experimental.pipeline.kg_builder import SimpleKGPipeline
from neo4j_graphrag.llm import OpenAILLM

LLM_MODEL = "gpt-5.4-mini"
# Must match the similaritySearch tool in agent/agent.json.
EMBEDDING_MODEL = "text-embedding-3-small"
MAX_CHUNK_CHARS = 4000
CONCURRENCY = 8

# Talk-level facts come from the sheet. The LLM only extracts what the
# description mentions. No Person type, so the "Speakers:" line yields nothing.
SCHEMA = {
    "node_types": [
        {
            "label": "Technology",
            "description": "A specific named software product, framework, library, programming language, "
            "cloud platform or protocol, e.g. LangChain, Google Cloud, Apache Kafka, Python, MCP, "
            "Neo4j Graph Data Science, Neo4j Aura. A proper noun, not a technique. "
            "Skip plain \"Neo4j\", which every talk mentions.",
            "properties": [{"name": "name", "type": "STRING", "required": True}],
        },
        {
            "label": "Concept",
            "description": "One of the 3-5 central techniques, methods or ideas the talk is about, e.g. GraphRAG, "
            "Entity Resolution, Agent Memory, Context Graphs, Temporal Reasoning, Graph Neural Networks, "
            "Ontology Management. Use broad, reusable names in Title Case that other talks could share, "
            "not phrases coined by this talk. Skip generic terms like AI, Data or Knowledge Graph.",
            "properties": [{"name": "name", "type": "STRING", "required": True}],
        },
        {
            "label": "Organization",
            "description": "A company, public body, university or project named in the talk as a user, "
            "customer or partner, e.g. Adobe, Deutsche Bahn. Not Neo4j, not the speaker.",
            "properties": [{"name": "name", "type": "STRING", "required": True}],
        },
        {
            "label": "Domain",
            "description": "A real-world industry or field the talk applies graphs to, e.g. cybersecurity, "
            "wealth management, healthcare, biodiversity, sports sponsorship, supply chain. "
            "Not AI or software engineering itself. Extract none if the talk is not about a specific industry.",
            "properties": [{"name": "name", "type": "STRING", "required": True}],
        },
    ],
    "relationship_types": [
        {"label": "USES", "description": "The organization uses the technology."},
        {"label": "OPERATES_IN", "description": "The organization works in the domain."},
    ],
    "patterns": [
        ("Organization", "USES", "Technology"),
        ("Organization", "OPERATES_IN", "Domain"),
    ],
    "additional_node_types": False,
    "additional_relationship_types": False,
    "additional_patterns": False,
}


# Merged before entity resolution. Add pairs here as duplicates show up.
ALIASES = {
    "Model Context Protocol": "MCP",
    "Model Context Protocol (MCP)": "MCP",
    "Neo4j AuraDB": "Neo4j Aura",
    "AuraDB": "Neo4j Aura",
    "Text2Cypher": "Text-to-Cypher",
    "RAG": "Retrieval-Augmented Generation",
    "GDS": "Neo4j Graph Data Science",
}

# Fold plurals ("Knowledge Graphs") into the singular when both were extracted,
# then give names that differ only by case one spelling.
NORMALIZE = """
UNWIND keys($aliases) AS alias
MATCH (e:__Entity__ {name: alias})
SET e.name = $aliases[alias]
WITH count(*) AS _
MATCH (p:__Entity__) WHERE p.name ENDS WITH 's'
MATCH (s:__Entity__) WHERE toLower(s.name) = toLower(left(p.name, size(p.name) - 1))
SET p.name = s.name
WITH count(*) AS _
MATCH (e:__Entity__)
WITH [l IN labels(e) WHERE NOT l STARTS WITH '__'][0] AS label, toLower(e.name) AS key,
     collect(e) AS es, min(e.name) AS name
WHERE size(es) > 1
UNWIND es AS e
SET e.name = name
"""


@dataclass
class Talk:
    id: str
    title: str
    description: str
    track: str | None
    formats: list[str]
    levels: list[str]
    topics: list[str]
    use_cases: list[str]
    speakers: list[dict] = field(default_factory=list)
    slots: list[dict] = field(default_factory=list)

    def text(self) -> str:
        names = ", ".join(sp["name"] for sp in self.speakers)
        return f"Title: {self.title}\nSpeakers: {names}\nDescription: {self.description}"


def clean(v) -> str:
    return re.sub(r"\s+", " ", str(v or "")).strip()


def split_list(v) -> list[str]:
    return list(dict.fromkeys(p for p in (clean(x) for x in str(v or "").split(",")) if p))


def session_id(v) -> str:
    return str(int(float(v)))


def iso_utc(v) -> str | None:
    if not v:
        return None
    dt = v if isinstance(v, datetime) else datetime.fromisoformat(str(v))
    return dt.replace(tzinfo=timezone.utc, microsecond=0).isoformat().replace("+00:00", "Z")


# Letters NFKD doesn't decompose to ASCII.
SLUG_CHARS = str.maketrans({"ł": "l", "Ł": "l", "ø": "o", "Ø": "o", "đ": "d", "Đ": "d", "ß": "ss", "æ": "ae", "Æ": "ae"})


def slug(name: str) -> str:
    ascii_name = unicodedata.normalize("NFKD", name.translate(SLUG_CHARS)).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", "-", ascii_name.lower()).strip("-")


def read_talks(path: str) -> list[Talk]:
    """One Talk per distinct title + description.

    The export has one row per speaker per session id. Only a session's first
    row has the title and other session columns. A talk broadcast in several
    slots repeats under a different session id per slot.
    """
    ws = openpyxl.load_workbook(path, read_only=True).worksheets[0]
    rows = ws.iter_rows(values_only=True)
    header = [clean(h) for h in next(rows)]
    rows = [r for r in (dict(zip(header, v)) for v in rows) if r.get("Session Id")]

    talks: dict[tuple[str, str], Talk] = {}
    by_session: dict[str, Talk] = {}
    for r in rows:
        title, description = clean(r["Title"]), clean(r["Description"])
        if not title:
            continue
        sid = session_id(r["Session Id"])
        talk = talks.get((title, description))
        if talk is None:
            talk = talks[(title, description)] = Talk(
                id=sid,
                title=title,
                description=description,
                track=clean(r["Room"]) or None,
                formats=split_list(r["Session format"]),
                levels=split_list(r["Level"]),
                topics=split_list(r["Topic of your presentation"]),
                use_cases=split_list(r["Neo4j Use-Case"]),
            )
        talk.id = min(talk.id, sid, key=int)
        talk.track = talk.track or clean(r["Room"]) or None
        if start := iso_utc(r["Scheduled At"]):
            talk.slots.append(
                {"id": sid, "startTime": start, "durationMinutes": int(float(r["Scheduled Duration"] or 0))}
            )
        by_session[sid] = talk

    for r in rows:
        talk = by_session.get(session_id(r["Session Id"]))
        if talk is None:
            raise SystemExit(f"Session {r['Session Id']} has speaker rows but no title row")
        name = clean(f"{clean(r['FirstName'])} {clean(r['LastName'])}")
        # Speaker Id is per submission (one person can have several), so key by name.
        speaker = {"id": slug(name), "name": name}
        if speaker not in talk.speakers:
            talk.speakers.append(speaker)

    for talk in talks.values():
        if len(talk.text()) > MAX_CHUNK_CHARS:
            raise SystemExit(f"Talk {talk.id} is {len(talk.text())} chars; raise MAX_CHUNK_CHARS")
    return list(talks.values())


STRUCTURED = """
UNWIND $talks AS t
MATCH (s:Session {id: t.id})
REMOVE s.path, s.createdAt, s.document_type
SET s.title = t.title, s.description = t.description,
    s.levels = t.levels, s.formats = t.formats
FOREACH (speaker IN t.speakers |
  MERGE (sp:Speaker {id: speaker.id}) SET sp.name = speaker.name
  MERGE (sp)-[:PRESENTS]->(s))
FOREACH (name IN CASE WHEN t.track IS NULL THEN [] ELSE [t.track] END |
  MERGE (tr:Track {name: name}) MERGE (s)-[:IN_TRACK]->(tr))
FOREACH (name IN t.topics | MERGE (tp:Topic {name: name}) MERGE (s)-[:HAS_TOPIC]->(tp))
FOREACH (name IN t.use_cases | MERGE (u:UseCase {name: name}) MERGE (s)-[:HAS_USE_CASE]->(u))
FOREACH (slot IN t.slots |
  MERGE (sl:Slot {id: slot.id})
  SET sl.startTime = datetime(slot.startTime), sl.durationMinutes = slot.durationMinutes,
      sl.endTime = datetime(slot.startTime) + duration({minutes: slot.durationMinutes})
  MERGE (s)-[:HAS_SLOT]->(sl))
"""

# Give entities stable ids (label + lowercased name) and a direct Session link,
# so text2cypher doesn't need to know about chunks. Plain "Neo4j" is in nearly
# every talk, so it's dropped even when the LLM extracts it.
FINALIZE = """
MATCH (e:__Entity__) WHERE toLower(e.name) = 'neo4j'
DETACH DELETE e
WITH count(*) AS _
MATCH (e:__Entity__)
SET e.id = toLower(head([l IN labels(e) WHERE NOT l STARTS WITH '__'])) + ':' + toLower(e.name)
WITH e
MATCH (e)-[:FROM_CHUNK]->(:Chunk)-[:FROM_DOCUMENT]->(s:Session)
MERGE (s)-[:MENTIONS]->(e)
"""


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("xlsx")
    ap.add_argument("--reset", action="store_true", help="delete everything in the database first")
    ap.add_argument("--limit", type=int, help="only build the first N talks")
    args = ap.parse_args()

    talks = read_talks(args.xlsx)[: args.limit]
    print(f"{len(talks)} talks, {sum(len(t.slots) for t in talks)} slots")

    driver = neo4j.GraphDatabase.driver(
        os.environ["NEO4J_URI"], auth=(os.environ["NEO4J_USERNAME"], os.environ["NEO4J_PASSWORD"])
    )
    if args.reset:
        with driver.session() as session:
            session.run("MATCH (n) CALL (n) { DETACH DELETE n } IN TRANSACTIONS OF 1000 ROWS").consume()
        # Drop constraints too, e.g. ones left by running data/load.cypher here.
        for r in driver.execute_query("SHOW CONSTRAINTS YIELD name").records:
            driver.execute_query(f"DROP CONSTRAINT `{r['name']}`")
    driver.execute_query("CREATE CONSTRAINT session_id IF NOT EXISTS FOR (s:Session) REQUIRE s.id IS UNIQUE")

    pipeline = SimpleKGPipeline(
        llm=OpenAILLM(model_name=LLM_MODEL, model_params={"response_format": {"type": "json_object"}}),
        driver=driver,
        embedder=OpenAIEmbeddings(model=EMBEDDING_MODEL),
        schema=SCHEMA,
        from_file=False,
        text_splitter=FixedSizeSplitter(chunk_size=MAX_CHUNK_CHARS, chunk_overlap=0),
        lexical_graph_config=LexicalGraphConfig(document_node_label="Session"),
        # Resolved once at the end; concurrent runs would race each other.
        perform_entity_resolution=False,
        on_error="RAISE",
    )

    sem = asyncio.Semaphore(CONCURRENCY)

    async def build(talk: Talk) -> None:
        async with sem:
            await pipeline.run_async(text=talk.text(), file_path=talk.id, document_metadata={"id": talk.id})
            print(f"  {talk.id} {talk.title[:70]}")

    await asyncio.gather(*(build(t) for t in talks))

    driver.execute_query(STRUCTURED, talks=[vars(t) for t in talks])
    driver.execute_query(NORMALIZE, aliases=ALIASES)
    stats = await SinglePropertyExactMatchResolver(driver=driver).run()
    print(f"entity resolution: {stats}")
    driver.execute_query(FINALIZE)
    driver.close()


if __name__ == "__main__":
    asyncio.run(main())
