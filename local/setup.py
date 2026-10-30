# /// script
# requires-python = ">=3.10"
# ///
"""Load the NODES 2026 graph into a local CypherLite database (local/nodes.db).

Runs the same data/load.cypher that Aura uses, with three adjustments for CypherLite:
- LOAD CSV reads the CSVs in data/ instead of fetching them from GitHub.
- Embeddings are set with a plain SET (CypherLite has no db.create.setNodeVectorProperty).
- Speakers get an accent-folded copy of their name (nameFolded), which the local speaker search
  matches against (CypherLite's full-text index has no standard-folding analyzer or wildcards yet).

Usage: uv run local/setup.py [--reset]
"""

import json
import re
import shutil
import subprocess
import sys
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
DB = Path(__file__).resolve().parent / "nodes.db"

# Letters NFKD doesn't decompose into a base letter plus an accent.
FOLD_CHARS = str.maketrans({"ł": "l", "Ł": "L", "ø": "o", "Ø": "O", "đ": "d", "Đ": "D", "ß": "ss", "æ": "ae", "Æ": "AE"})


def fold(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text.translate(FOLD_CHARS))
    return "".join(c for c in decomposed if not unicodedata.combining(c))


def cypherlite(*args: str, stdin: str | None = None) -> str:
    if not shutil.which("cypherlite"):
        sys.exit("cypherlite isn't installed: see https://neo4j-labs.github.io/cypherlite/")
    result = subprocess.run(
        ["cypherlite", "-d", str(DB), "--load-csv-dir", str(DATA), *args],
        input=stdin,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        sys.exit(f"cypherlite failed:\n{result.stderr or result.stdout}")
    return result.stdout


def load_script() -> str:
    script = (DATA / "load.cypher").read_text()
    script = re.sub(r"https://raw\.githubusercontent\.com/[^']*/data/", "", script)
    script = re.sub(
        r"CALL db\.create\.setNodeVectorProperty\(c, 'embedding', (.*)\);",
        r"SET c.embedding = \1;",
        script,
    )
    return script


def main() -> None:
    if "--reset" in sys.argv and DB.exists():
        shutil.rmtree(DB)
        for extra in DB.parent.glob(DB.name + ".*"):
            extra.unlink()
    if DB.exists():
        sys.exit(f"{DB} already exists. Run with --reset to rebuild it.")

    print("Loading data/*.csv into local/nodes.db…")
    out = cypherlite("--jsonl", "query", stdin=load_script())
    counts = json.loads(out.strip().splitlines()[-1])

    rows = [json.loads(line) for line in cypherlite("--jsonl", "-c", "MATCH (p:Speaker) RETURN p.name AS name").splitlines()]
    folded = [{"name": r["name"], "folded": fold(r["name"])} for r in rows]
    if folded:
        cypherlite("-c", f"UNWIND {json.dumps(folded)} AS row MATCH (p:Speaker {{name: row.name}}) SET p.nameFolded = row.folded")

    print(
        f"Done: {counts['sessions']} sessions, {counts['speakers']} speakers, "
        f"{counts['embeddedChunks']} embedded chunks, {counts['mentions']} mentions."
    )


if __name__ == "__main__":
    main()
