"""Build the SQLite database and the policy index from the challenge data.

    python scripts/build_db.py            # CSVs -> store.db
    python scripts/build_db.py --index    # also build the policy index

Both steps are idempotent: the database is recreated and the index cache is
rewritten, so a fresh clone only needs these two commands.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.db import rebuild_database  # noqa: E402
from app.rag import build_index, split_sections  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", action="store_true", help="also build the policy index")
    parser.add_argument("--no-embeddings", action="store_true", help="sections only, no API call")
    args = parser.parse_args()

    counts = rebuild_database()
    print("store.db:", counts)

    sections = split_sections()
    print(f"policy sections: {len(sections)}")
    for section in sections:
        print(f"  §{section.section_id:<4} p{section.page} {section.title}")

    if args.index:
        report = build_index(sections=None if not args.no_embeddings else sections)
        print("index:", report)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
