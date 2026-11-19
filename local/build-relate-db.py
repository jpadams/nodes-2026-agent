# /// script
# requires-python = ">=3.10"
# dependencies = ["relate-db==1.0.0"]
# ///
"""Build data/nodes-2026.rl, the NODES 2026 graph as a single Relate file, from data/*.csv.

Runs data/load.cypher, the same script Aura uses, with small changes for Relate: LOAD CSV reads
the local CSVs instead of GitHub, and embeddings are stored with a plain SET (there's no
db.create.setNodeVectorProperty). Then gives each speaker an accent-folded nameFolded.

The load runs in a temporary directory store, then `relate clone` copies it into the .rl file.
A .rl file keeps the space of the files it no longer needs, like the write-ahead log of the
load (6 MB here), while a directory frees it. So the clone is a fifth of the size of a .rl the
load writes to directly.

Only needed when the CSVs change. Needs the relate CLI (https://relatedb.ai/) for the clone.
Usage: uv run local/build-relate-db.py
"""

import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data"
# LOAD CSV may read only from this directory. Set before relate opens a database.
os.environ["RELATE_LOAD_CSV_DIR"] = str(DATA)

import relate  # noqa: E402

from nodes_graph import DATABASE, fold  # noqa: E402


def statements():
    """data/load.cypher's statements, adjusted for Relate."""
    script = (DATA / "load.cypher").read_text()
    script = re.sub(r"https://raw\.githubusercontent\.com/[^']*/data/", "", script)
    script = re.sub(r"CALL db\.create\.setNodeVectorProperty\(c, 'embedding', (.*)\);", r"SET c.embedding = \1;", script)
    # Statements end with ; at the end of a line, as in neo4j-cli.
    for statement in re.split(r";[ \t]*$", script, flags=re.M):
        if any(line.strip() and not line.strip().startswith("//") for line in statement.splitlines()):
            yield statement


def load(path: Path) -> None:
    db = relate.connect(str(path))
    try:
        for statement in statements():
            rows = [r.data() for r in db.execute(statement)]
        print(f"Loaded: {rows[-1]}")  # load.cypher ends by counting what it loaded.
        names = [r["name"] for r in db.execute("MATCH (p:Speaker) RETURN p.name AS name")]
        db.execute(
            "UNWIND $rows AS row MATCH (p:Speaker {name: row.name}) SET p.nameFolded = row.folded",
            {"rows": [{"name": n, "folded": fold(n)} for n in names]},
        )
    finally:
        db.close()


def main() -> None:
    cli = shutil.which("relate")
    if not cli:
        sys.exit("Install the relate CLI first, for `relate clone`: see https://relatedb.ai/")
    with tempfile.TemporaryDirectory() as tmp:
        store, bundle = Path(tmp) / "store", Path(tmp) / DATABASE.name
        load(store)
        subprocess.run([cli, "clone", str(store), str(bundle)], check=True, stdout=subprocess.DEVNULL)
        os.replace(bundle, DATABASE)
    print(f"Wrote {DATABASE.relative_to(DATA.parent)} ({DATABASE.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
