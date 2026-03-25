#!/usr/bin/env python3
"""
Schema Dumper for Benchmark Databases
======================================

Extracts CREATE TABLE statements from all SQLite databases in a directory
and saves them to a single JSON file for use by the baseline LLM script.

USAGE:
------
  python dump_schemas.py --database-dir /srv/train/train_databases --output schemas.json

Run this once on the VPS (or wherever the .sqlite files live). The output
schemas.json can then be used locally by run_baseline.py.

OUTPUT FORMAT:
--------------
{
  "financial": "CREATE TABLE loan (...)\n\nCREATE TABLE account (...)\n...",
  "olympics": "CREATE TABLE city (...)\n\nCREATE TABLE games (...)\n...",
  ...
}
"""

import argparse
import json
import sqlite3
from pathlib import Path


def extract_schema(db_path: Path) -> str:
    """Extract CREATE TABLE statements from a SQLite database."""
    conn = sqlite3.connect(str(db_path))
    try:
        rows = conn.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND sql IS NOT NULL ORDER BY name"
        ).fetchall()
        return "\n\n".join(row[0] for row in rows)
    finally:
        conn.close()


def dump_schemas(database_dir: Path, output_file: Path):
    """Walk database_dir, extract schemas from all .sqlite files, save to JSON."""
    schemas = {}
    errors = []

    db_dirs = sorted(p for p in database_dir.iterdir() if p.is_dir())
    if not db_dirs:
        print(f"No subdirectories found in {database_dir}")
        return

    for db_dir in db_dirs:
        db_id = db_dir.name
        db_path = db_dir / f"{db_id}.sqlite"

        if not db_path.exists():
            errors.append(f"  SKIP {db_id}: no file at {db_path}")
            continue

        try:
            schema = extract_schema(db_path)
            schemas[db_id] = schema
            table_count = schema.count("CREATE TABLE")
            print(f"  OK   {db_id} ({table_count} tables)")
        except Exception as e:
            errors.append(f"  ERR  {db_id}: {e}")

    with open(output_file, "w") as f:
        json.dump(schemas, f, indent=2)

    print(f"\nDumped {len(schemas)} databases → {output_file}")
    if errors:
        print("\nWarnings:")
        for e in errors:
            print(e)


def main():
    parser = argparse.ArgumentParser(
        description="Dump SQLite schemas to a JSON file for use by run_baseline.py"
    )
    parser.add_argument(
        "--database-dir",
        required=True,
        type=str,
        help="Directory containing database subdirectories (e.g. /srv/train/train_databases)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="schemas.json",
        help="Output JSON file path (default: schemas.json)",
    )
    args = parser.parse_args()

    database_dir = Path(args.database_dir)
    output_file = Path(args.output)

    if not database_dir.exists():
        print(f"Error: database directory not found: {database_dir}")
        raise SystemExit(1)

    print(f"Scanning: {database_dir}")
    dump_schemas(database_dir, output_file)


if __name__ == "__main__":
    main()
