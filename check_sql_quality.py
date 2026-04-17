"""
Level-2 SQL quality check: run gold intermediate_sqls and compare result sets
against what the model actually produced, per (task, db).

Usage:
    python check_sql_quality.py \
        --task dev/task.json \
        --db   dev/databases \
        --log  data/baseline_20260416_163510/log.json
"""

import argparse
import json
import os
import sqlite3


def run_sql(db_dir: str, db_name: str, sql: str) -> list[dict] | dict:
    db_path = os.path.join(db_dir, db_name, f"{db_name}.sqlite")
    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        cur.execute(sql)
        rows = cur.fetchall()
        cols = [d[0] for d in cur.description]
        conn.close()
        return [dict(zip(cols, row)) for row in rows]
    except Exception as e:
        return {"error": str(e)}


def numeric_values(rows: list[dict]) -> list[float]:
    """Extract all numeric cell values from result rows, flattened and sorted."""
    vals = []
    for row in rows:
        for v in row.values():
            try:
                vals.append(float(v))
            except (TypeError, ValueError):
                pass
    return sorted(vals)


def compare_value_sets(gold_vals: list[float], model_vals: list[float], tol: float = 1e-4) -> dict:
    """Compare two sorted numeric value lists element-wise."""
    if len(gold_vals) != len(model_vals):
        return {
            "match": False,
            "reason": f"value count differs: gold={len(gold_vals)}, model={len(model_vals)}",
        }
    mismatches = []
    for i, (g, m) in enumerate(zip(gold_vals, model_vals)):
        if abs(g - m) > tol:
            mismatches.append({"index": i, "gold": g, "model": m, "diff": abs(g - m)})
    if mismatches:
        return {"match": False, "reason": f"{len(mismatches)} value(s) differ", "mismatches": mismatches[:5]}
    return {"match": True}


def check_task(task: dict, db_dir: str, model_exec_results: list) -> dict:
    """Compare gold vs model execution results for one task."""
    gold_sqls = task.get("intermediate_sqls", [])
    if not gold_sqls:
        return {"task_id": task["id"], "skipped": "no intermediate_sqls"}

    # Index model results by db (take last one if multiple for same db)
    model_by_db = {}
    for er in (model_exec_results or []):
        model_by_db[er["db"]] = er

    per_db = []
    for gs in gold_sqls:
        db = gs["db"]
        sql = gs["sql"]

        gold_rows = run_sql(db_dir, db, sql)
        model_entry = model_by_db.get(db)

        entry = {"db": db, "gold_sql": sql}

        if isinstance(gold_rows, dict) and "error" in gold_rows:
            entry["gold_error"] = gold_rows["error"]
            per_db.append(entry)
            continue

        entry["gold_row_count"] = len(gold_rows)

        if model_entry is None:
            entry["model_error"] = "no model result for this db"
            per_db.append(entry)
            continue

        model_rows = model_entry.get("result", [])
        if isinstance(model_rows, dict) and "error" in model_rows:
            entry["model_error"] = model_rows["error"]
            per_db.append(entry)
            continue

        entry["model_row_count"] = len(model_rows)
        entry["row_count_match"] = len(gold_rows) == len(model_rows)

        gold_vals = numeric_values(gold_rows)
        model_vals = numeric_values(model_rows)
        entry["gold_numeric_count"] = len(gold_vals)
        entry["model_numeric_count"] = len(model_vals)
        entry["value_comparison"] = compare_value_sets(gold_vals, model_vals)

        per_db.append(entry)

    all_match = all(
        e.get("value_comparison", {}).get("match", False)
        for e in per_db
        if "value_comparison" in e
    )
    has_value_check = any("value_comparison" in e for e in per_db)

    return {
        "task_id": task["id"],
        "difficulty": task.get("difficulty", ""),
        "question": task["question"][:80] + "...",
        "overall_match": all_match if has_value_check else None,
        "per_db": per_db,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="dev/task.json")
    parser.add_argument("--db", default="dev/databases")
    parser.add_argument("--log", required=True, help="Path to log.json from a baseline run")
    parser.add_argument("--output", default=None, help="Save full report to JSON file")
    args = parser.parse_args()

    with open(args.task) as f:
        tasks = json.load(f)
    with open(args.log) as f:
        log = json.load(f)

    # Index log by task_id → execution results
    log_by_task: dict[int, list] = {}
    for entry in log:
        tid = entry["task_id"]
        exec_step = next((s for s in entry["steps"] if s["step"] == "execution"), None)
        if exec_step:
            log_by_task[tid] = exec_step.get("results", [])

    tasks_with_gold = [t for t in tasks if "intermediate_sqls" in t and "db_id" in t]
    print(f"Tasks with intermediate_sqls: {len(tasks_with_gold)}")
    print(f"Tasks in log: {len(log_by_task)}\n")

    results = []
    for task in tasks_with_gold:
        tid = task["id"]
        if tid not in log_by_task:
            print(f"[Task {tid}] NOT IN LOG — skipping")
            continue

        r = check_task(task, args.db, log_by_task[tid])
        results.append(r)

        match_str = {True: "MATCH", False: "MISMATCH", None: "N/A"}[r.get("overall_match")]
        print(f"[Task {tid}] {match_str}  — {r['question']}")
        for db_r in r.get("per_db", []):
            db = db_r["db"]
            if "gold_error" in db_r:
                print(f"  {db}: gold SQL error: {db_r['gold_error']}")
            elif "model_error" in db_r:
                print(f"  {db}: model error: {db_r['model_error']}")
            else:
                vc = db_r.get("value_comparison", {})
                row_ok = "✓" if db_r.get("row_count_match") else "✗"
                val_ok = "✓" if vc.get("match") else "✗"
                print(f"  {db}: rows {row_ok} (gold={db_r.get('gold_row_count')} model={db_r.get('model_row_count')})  values {val_ok}", end="")
                if not vc.get("match"):
                    print(f"  → {vc.get('reason', '')}", end="")
                print()

    matched = sum(1 for r in results if r.get("overall_match") is True)
    checked = sum(1 for r in results if r.get("overall_match") is not None)
    print(f"\nSQL quality: {matched}/{checked} tasks fully matched gold results")

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Full report saved to {args.output}")


if __name__ == "__main__":
    main()
