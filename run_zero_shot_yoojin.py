#!/usr/bin/env python3
"""
Zero-shot task difficulty evaluator.

Two-turn pipeline (no tools):
  1. Claude sees schema + question → generates SQL queries for each database
  2. Script executes those SQLs against real SQLite databases
  3. Claude sees SQL results → computes and returns the final answer

If the model gets most tasks right, the tasks are too easy.

Usage:
  python run_zero_shot.py [--task-file dev/task.json] [--db-dir dev/databases]
                          [--api-key api_key.txt] [--model claude-haiku-4-5-20251001]
                          [--output results_zero_shot.json]
"""

import argparse
import json
import re
import sqlite3
from pathlib import Path

import anthropic


# ---------------------------------------------------------------------------
# Schema extraction (with 3 example values per column)
# ---------------------------------------------------------------------------

def get_schema(db_dir: Path, db_id: str, prefix_tables: bool = False) -> str:
    """Return schema DDL for a database.
    If prefix_tables=True, tables are shown as db_id.table_name for cross-db queries."""
    db_path = db_dir / db_id / f"{db_id}.sqlite"
    conn = sqlite3.connect(str(db_path))
    lines = []
    cur = conn.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
    for (table,) in cur.fetchall():
        col_cur = conn.execute(f"PRAGMA table_info([{table}])")
        col_infos = col_cur.fetchall()
        col_strs = []
        for col in col_infos:
            col_name, col_type = col[1], col[2]
            ex_cur = conn.execute(
                f"SELECT DISTINCT [{col_name}] FROM [{table}] WHERE [{col_name}] IS NOT NULL LIMIT 3"
            )
            examples = [str(r[0]) for r in ex_cur.fetchall()]
            example_str = f"  -- e.g. {', '.join(examples)}" if examples else ""
            col_strs.append(f"{col_name} {col_type}{example_str}")
        table_ref = f"[{db_id}].{table}" if prefix_tables else table
        lines.append(f"  {table_ref}(\n    " + ",\n    ".join(col_strs) + "\n  )")
    conn.close()
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# SQL execution
# ---------------------------------------------------------------------------

def run_sql(db_dir: Path, db_id: str, sql: str) -> tuple[list[str], list[tuple]]:
    db_path = db_dir / db_id / f"{db_id}.sqlite"
    conn = sqlite3.connect(str(db_path))
    try:
        cur = conn.execute(sql)
        cols = [d[0] for d in cur.description] if cur.description else []
        rows = cur.fetchall()
        return cols, rows
    finally:
        conn.close()


def run_sql_cross(db_dir: Path, db_ids: list[str], sql: str) -> tuple[list[str], list[tuple]]:
    """Execute a SQL query that spans multiple databases using ATTACH DATABASE.
    Tables must be referenced as db_name.table_name in the query."""
    conn = sqlite3.connect(":memory:")
    try:
        for db_id in db_ids:
            db_path = db_dir / db_id / f"{db_id}.sqlite"
            conn.execute(f"ATTACH DATABASE '{db_path}' AS [{db_id}]")
        cur = conn.execute(sql)
        cols = [d[0] for d in cur.description] if cur.description else []
        rows = cur.fetchall()
        return cols, rows
    finally:
        conn.close()


def format_result(cols: list[str], rows: list[tuple], max_rows: int = 500) -> str:
    if not rows:
        return "(no rows)"
    header = " | ".join(cols)
    sep = "-" * len(header)
    lines = [header, sep]
    for row in rows[:max_rows]:
        lines.append(" | ".join(str(v) for v in row))
    if len(rows) > max_rows:
        lines.append(f"... ({len(rows)} rows total, showing {max_rows})")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# JSON extraction helper
# ---------------------------------------------------------------------------

def extract_json(text: str) -> dict | None:
    text = re.sub(r"```(?:json)?\s*", "", text).strip().rstrip("`").strip()
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    for i, ch in enumerate(text[start:], start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start:i + 1])
                except json.JSONDecodeError:
                    return None
    return None


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def within_tolerance(got, expected, tolerance) -> bool:
    if isinstance(expected, dict):
        if not isinstance(got, dict):
            return False
        for key, exp_val in expected.items():
            tol = tolerance[key] if isinstance(tolerance, dict) else tolerance
            got_val = got.get(key)
            if got_val is None:
                return False
            if not within_tolerance(got_val, exp_val, tol):
                return False
        return True
    try:
        return abs(float(got) - float(expected)) <= float(tolerance)
    except (TypeError, ValueError):
        return False


def extract_answer(parsed: dict, expected):
    if isinstance(expected, dict):
        return parsed
    if "result" in parsed:
        return parsed["result"]
    vals = list(parsed.values())
    return vals[0] if len(vals) == 1 else parsed


# ---------------------------------------------------------------------------
# Two-turn pipeline
# ---------------------------------------------------------------------------

CROSS_DB_PROMPT = """\
You are an expert SQL analyst. The databases below are all attached in a single SQLite session. \
You can join tables across databases in ONE query using the syntax [db_name].table_name.

RULES:
- SQLite syntax only: SUBSTR(col, 1, 7) for year-month, CAST(x AS REAL) for division, strftime() for dates.
- Write ONE cross-database SQL query that retrieves all data needed to answer the question.
- Filter out NULL values on join and group-by keys.
- Only reference columns that exist in the schemas below.

QUESTION:
{question}

EVIDENCE:
{evidence}

DATABASE SCHEMAS (reference tables as [db_name].table_name):
{schemas}

Return ONLY a JSON object — no explanation, no markdown:
{{
  "sqls": [
    {{"db_id": "__cross__", "sql": "<single SQL joining across databases>"}}
  ]
}}"""

SQL_PROMPT = """\
You are an expert SQL analyst. Given the SQLite database schemas below, write SQL queries \
to retrieve the raw data needed to answer the question.

RULES:
- SQLite syntax only: SUBSTR(col, 1, 7) for year-month, CAST(x AS REAL) for division, strftime() for dates. No DATE_TRUNC, no ::float.
- Write the fewest queries needed. Do NOT write exploratory or probe queries (no LIMIT 5, no schema inspection).
- Each query must return data ready for statistical computation — aggregated, not raw rows.
- Always filter out NULL values on join keys and group-by columns (add WHERE col IS NOT NULL).
- In subqueries, only reference columns that the subquery actually exposes in its SELECT list.

QUESTION:
{question}

EVIDENCE:
{evidence}

DATABASE SCHEMAS:
{schemas}

Return ONLY a JSON object — no explanation, no markdown:
{{
  "sqls": [
    {{"db_id": "<database_id>", "sql": "<SQL query>"}},
    ...
  ]
}}"""

ANSWER_PROMPT = """\
You are an expert data analyst. Using the query results below, compute the final answer \
to the question. Perform all required statistics (correlation, t-test, regression, etc.) yourself.

RULES:
- Skip any row where a key column or value is NULL.
- When joining results across databases, only include keys present in ALL result sets.
- Compute statistics precisely using exact formulas (Pearson, Spearman, Welch's t, etc.).
- Return the result rounded to the same precision as the hint format.

QUESTION:
{question}

EVIDENCE:
{evidence}

QUERY RESULTS:
{results}

Return ONLY a JSON object with the numeric answer — no explanation, no markdown.
Expected format hint: {hint}"""


def exec_sqls(db_dir: Path, sql_items: list[dict], db_key: str = "db_id", all_db_ids: list[str] = None) -> tuple[list[dict], list[str]]:
    """Execute a list of SQL items, return (records, results_block).
    Each record stores columns, sample_rows, row_count, and error."""
    records = []
    results_block = []
    for item in sql_items:
        db_id = item.get(db_key, "")
        sql = item.get("sql", "")
        record = {"db_id": db_id, "sql": sql}
        try:
            if db_id == "__cross__" and all_db_ids:
                cols, rows = run_sql_cross(db_dir, all_db_ids, sql)
            else:
                cols, rows = run_sql(db_dir, db_id, sql)
            record["columns"] = cols
            record["row_count"] = len(rows)
            record["sample_rows"] = [list(r) for r in rows[:5]]
            record["error"] = None
            results_block.append(f"[{db_id}]\nSQL: {sql}\n{format_result(cols, rows)}")
        except Exception as e:
            record["columns"] = []
            record["row_count"] = 0
            record["sample_rows"] = []
            record["error"] = str(e)
            results_block.append(f"[{db_id}]\nSQL: {sql}\nERROR: {e}")
        records.append(record)
    return records, results_block


def compare_sql_results(gold: list[dict], generated: list[dict]) -> list[dict]:
    """Compare gold vs generated SQL results per database.
    Groups by db_id and compares the last query for each db (the most complete one)."""
    def last_ok(records, db_id):
        matches = [r for r in records if r["db_id"] == db_id and r["error"] is None]
        return matches[-1] if matches else None

    all_dbs = {r["db_id"] for r in gold + generated}
    comparisons = []
    for db_id in sorted(all_dbs):
        g = last_ok(gold, db_id)
        m = last_ok(generated, db_id)
        comp = {"db_id": db_id}

        if g is None and m is None:
            comp["status"] = "both_failed"
        elif g is None:
            comp["status"] = "gold_failed"
        elif m is None:
            comp["status"] = "generated_failed"
        else:
            comp["gold_row_count"] = g["row_count"]
            comp["generated_row_count"] = m["row_count"]
            comp["row_count_match"] = g["row_count"] == m["row_count"]
            comp["gold_columns"] = g["columns"]
            comp["generated_columns"] = m["columns"]

            # Compare numeric values if both return same number of rows and ≥1 numeric column
            if g["row_count"] == m["row_count"] and g["row_count"] > 0:
                g_rows = sorted([list(r) for r in g["sample_rows"]], key=lambda r: str(r[0]))
                m_rows = sorted([list(r) for r in m["sample_rows"]], key=lambda r: str(r[0]))
                numeric_diffs = []
                for gr, mr in zip(g_rows, m_rows):
                    for gv, mv in zip(gr[1:], mr[1:]):  # skip key column
                        try:
                            numeric_diffs.append(abs(float(gv) - float(mv)))
                        except (TypeError, ValueError):
                            pass
                if numeric_diffs:
                    comp["mean_abs_diff"] = round(sum(numeric_diffs) / len(numeric_diffs), 6)
                    comp["max_abs_diff"] = round(max(numeric_diffs), 6)
                    comp["status"] = "close" if comp["max_abs_diff"] < 0.01 else "diverged"
                else:
                    comp["status"] = "matched" if g["sample_rows"] == m["sample_rows"] else "different_values"
            else:
                comp["status"] = "row_count_mismatch" if g["row_count"] != m["row_count"] else "empty"

        comparisons.append(comp)
    return comparisons


def run_task(task: dict, db_dir: Path, client: anthropic.Anthropic, model: str, cross_db: bool = False) -> tuple[bool, str, any, list[dict], list[dict], list[dict]]:
    db_ids = task["db_id"]

    if cross_db:
        schemas = "\n\n".join(
            f"-- Database: {db_id}\n{get_schema(db_dir, db_id, prefix_tables=True)}" for db_id in db_ids
        )
        sql_prompt = CROSS_DB_PROMPT.format(
            question=task["question"],
            evidence=task.get("evidence", ""),
            schemas=schemas,
        )
    else:
        schemas = "\n\n".join(
            f"-- Database: {db_id}\n{get_schema(db_dir, db_id)}" for db_id in db_ids
        )
        sql_prompt = SQL_PROMPT.format(
            question=task["question"],
            evidence=task.get("evidence", ""),
            schemas=schemas,
        )

    # Execute gold intermediate SQLs for reference (db key is "db" in task.json)
    gold_sqls, _ = exec_sqls(db_dir, task.get("intermediate_sqls", []), db_key="db", all_db_ids=db_ids)

    # Turn 1: get SQL from model
    sql_response = client.messages.create(
        model=model,
        max_tokens=1024,
        messages=[
            {"role": "user", "content": sql_prompt},
            {"role": "assistant", "content": "{"},
        ],
    ).content[0].text
    sql_response = "{" + sql_response

    parsed_sqls = extract_json(sql_response)
    if not parsed_sqls or "sqls" not in parsed_sqls:
        comparisons = compare_sql_results(gold_sqls, [])
        return False, f"Could not parse SQL response: {sql_response[:200]}", None, [], gold_sqls, comparisons

    # Execute model-generated SQLs (cross-db queries use __cross__ db_id)
    generated_sqls, results_block = exec_sqls(db_dir, parsed_sqls["sqls"], all_db_ids=db_ids)
    comparisons = compare_sql_results(gold_sqls, generated_sqls)

    # Turn 2: compute final answer from results
    expected = task["result"]
    hint = json.dumps(expected) if isinstance(expected, dict) else f'{{"result": {expected}}}'
    answer_prompt = ANSWER_PROMPT.format(
        question=task["question"],
        evidence=task.get("evidence", ""),
        results="\n\n".join(results_block),
        hint=hint,
    )
    answer_response = client.messages.create(
        model=model,
        max_tokens=2048,
        messages=[
            {"role": "user", "content": answer_prompt},
            {"role": "assistant", "content": "{"},
        ],
    ).content[0].text
    answer_response = "{" + answer_response

    parsed_answer = extract_json(answer_response)
    if parsed_answer is None:
        return False, f"Could not parse answer: {answer_response[:200]}", None, generated_sqls, gold_sqls, comparisons

    got = extract_answer(parsed_answer, expected)
    ok = within_tolerance(got, expected, task.get("tolerance", 0.01))
    return ok, "", got, generated_sqls, gold_sqls, comparisons


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-file", default="dev/task.json")
    parser.add_argument("--db-dir", default="dev/databases")
    parser.add_argument("--api-key", default="api_key.txt")
    parser.add_argument("--model", default="claude-haiku-4-5-20251001")
    parser.add_argument("--output", default="results_zero_shot.json")
    parser.add_argument("--cross-db", action="store_true",
                        help="Use ATTACH DATABASE to let the model write a single cross-db SQL query")
    args = parser.parse_args()

    api_key = Path(args.api_key).read_text().strip()
    tasks = json.loads(Path(args.task_file).read_text())
    client = anthropic.Anthropic(api_key=api_key)

    print(f"Model: {args.model}")
    print(f"Mode:  {'cross-db (ATTACH)' if args.cross_db else 'per-db (separate queries)'}")
    print(f"Tasks: {len(tasks)}")
    print("=" * 60)

    passed = 0
    output_records = []

    for task in tasks:
        tid = task["id"]
        difficulty = task["difficulty"]
        expected = task["result"]

        ok, err, got, generated_sqls, gold_sqls, comparisons = run_task(task, Path(args.db_dir), client, args.model, cross_db=args.cross_db)
        if ok:
            passed += 1

        status = "PASS" if ok else "FAIL"
        print(f"Task {tid} [{difficulty:<12}] {status}")
        print(f"  expected: {expected}")
        print(f"  got:      {got}")
        print(f"  gold sqls:")
        for s in gold_sqls:
            tag = "OK" if s["error"] is None else "ERR"
            print(f"    [{tag}] {s['db_id']} ({s['row_count']} rows): {s['sql'][:80]}...")
        print(f"  generated sqls:")
        for s in generated_sqls:
            tag = "OK" if s["error"] is None else "ERR"
            print(f"    [{tag}] {s['db_id']} ({s['row_count']} rows): {s['sql'][:80]}...")
        print(f"  result comparison (gold vs generated):")
        for c in comparisons:
            db = c["db_id"]
            st = c["status"]
            if st in ("both_failed", "gold_failed", "generated_failed"):
                print(f"    {db}: {st}")
            elif st == "row_count_mismatch":
                print(f"    {db}: row count mismatch — gold {c['gold_row_count']} vs generated {c['generated_row_count']}")
            elif st in ("close", "diverged"):
                print(f"    {db}: {st} — mean_diff={c.get('mean_abs_diff')}, max_diff={c.get('max_abs_diff')} (rows: {c['gold_row_count']})")
            else:
                print(f"    {db}: {st}")
        if err:
            print(f"  error:    {err}")

        output_records.append({
            "id": tid,
            "difficulty": difficulty,
            "passed": ok,
            "expected": expected,
            "got": got,
            "gold_sqls": gold_sqls,
            "generated_sqls": generated_sqls,
            "result_comparison": comparisons,
            "error": err or None,
        })

    total = len(tasks)
    pct = passed / total * 100
    print("=" * 60)
    print(f"Accuracy: {passed}/{total} ({pct:.0f}%)")
    if pct >= 60:
        print("WARNING: Model solved most tasks zero-shot — tasks may be too easy.")
    else:
        print("Tasks appear suitably difficult for zero-shot models.")

    Path(args.output).write_text(json.dumps(output_records, indent=2))
    print(f"Results saved to: {args.output}")


if __name__ == "__main__":
    main()
