"""Export the graph built by build_graph.py to the CSVs that data/load.cypher loads.

    uv run --env-file .env export_csv.py [--out ../data]

Rows are sorted and ids are stable. Entity files still change when the LLM
extracts differently on a rebuild.
Lists are ';'-separated, as are the 1536 embedding floats.
"""

import argparse
import csv
import os
from pathlib import Path

import neo4j

LIST_SEP = ";"

EXPORTS = {
    "sessions.csv": """
        MATCH (s:Session)
        OPTIONAL MATCH (s)-[:IN_TRACK]->(t:Track)
        RETURN s.id AS id, s.title AS title, s.description AS description, t.name AS track,
               s.levels AS levels, s.formats AS formats,
               COLLECT { MATCH (s)-[:HAS_TOPIC]->(x) RETURN x.name ORDER BY x.name } AS topics,
               COLLECT { MATCH (s)-[:HAS_USE_CASE]->(x) RETURN x.name ORDER BY x.name } AS useCases
        ORDER BY id
    """,
    "slots.csv": """
        MATCH (s:Session)-[:HAS_SLOT]->(sl:Slot)
        RETURN sl.id AS id, s.id AS sessionId, toString(sl.startTime) AS startTime,
               sl.durationMinutes AS durationMinutes
        ORDER BY startTime, id
    """,
    "speakers.csv": """
        MATCH (p:Speaker) RETURN p.id AS id, p.name AS name ORDER BY id
    """,
    "presents.csv": """
        MATCH (p:Speaker)-[:PRESENTS]->(s:Session)
        RETURN p.id AS speakerId, s.id AS sessionId ORDER BY sessionId, speakerId
    """,
    # One chunk per session, so the chunk id is the session id.
    "chunks.csv": """
        MATCH (c:Chunk)-[:FROM_DOCUMENT]->(s:Session)
        RETURN s.id AS id, s.id AS sessionId, c.text AS text, c.embedding AS embedding
        ORDER BY id
    """,
    "entities.csv": """
        MATCH (e:__Entity__)
        RETURN e.id AS id, [l IN labels(e) WHERE NOT l STARTS WITH '__'][0] AS label, e.name AS name
        ORDER BY id
    """,
    "mentions.csv": """
        MATCH (s:Session)-[:MENTIONS]->(e:__Entity__)
        RETURN DISTINCT s.id AS sessionId, e.id AS entityId,
               [l IN labels(e) WHERE NOT l STARTS WITH '__'][0] AS entityLabel
        ORDER BY sessionId, entityId
    """,
    "entity_relationships.csv": """
        MATCH (a:__Entity__)-[r]->(b:__Entity__)
        RETURN DISTINCT a.id AS startId, [l IN labels(a) WHERE NOT l STARTS WITH '__'][0] AS startLabel,
               type(r) AS type, b.id AS endId, [l IN labels(b) WHERE NOT l STARTS WITH '__'][0] AS endLabel
        ORDER BY startId, type, endId
    """,
}


def cell(v):
    if isinstance(v, list):
        return LIST_SEP.join(str(x) for x in v)
    return "" if v is None else v


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent.parent / "data"))
    out = Path(ap.parse_args().out)
    out.mkdir(parents=True, exist_ok=True)

    driver = neo4j.GraphDatabase.driver(
        os.environ["NEO4J_URI"], auth=(os.environ["NEO4J_USERNAME"], os.environ["NEO4J_PASSWORD"])
    )
    for name, query in EXPORTS.items():
        records, _, keys = driver.execute_query(query)
        with open(out / name, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(keys)
            w.writerows([cell(r[k]) for k in keys] for r in records)
        print(f"{name}: {len(records)} rows")
    driver.close()


if __name__ == "__main__":
    main()
