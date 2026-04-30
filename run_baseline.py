import argparse
import json
import os
import re
import sqlite3
import traceback
from datetime import datetime
from pathlib import Path

import anthropic
from dotenv import load_dotenv

load_dotenv()

MAX_ROWS = 500


def get_schema(db_path: str) -> str:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    cur.execute(
        "SELECT name, sql FROM sqlite_master WHERE type='table' AND sql IS NOT NULL ORDER BY name"
    )
    tables = cur.fetchall()

    parts = []
    for table in tables:
        name, ddl = table["name"], table["sql"]
        parts.append(ddl.strip() + ";")
        try:
            cur.execute(f'SELECT * FROM "{name}" LIMIT 3')
            rows = cur.fetchall()
            if rows:
                cols = [d[0] for d in cur.description]
                parts.append(f"-- Sample rows from {name}:")
                parts.append("-- " + " | ".join(cols))
                for row in rows:
                    parts.append("-- " + " | ".join(str(v) for v in row))
        except Exception:
            pass
        parts.append("")

    conn.close()
    return "\n".join(parts)


def strip_markdown(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*\n?", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\n?```$", "", text)
    return text.strip()


def execute_sqls(db_dir: str, sqls: list) -> list:
    results = []
    for item in sqls:
        db_name = item.get("db", "")
        sql = item.get("sql", "")
        db_path = os.path.join(db_dir, db_name, f"{db_name}.sqlite")
        try:
            conn = sqlite3.connect(db_path)
            conn.row_factory = sqlite3.Row
            cur = conn.cursor()
            cur.execute(sql)
            rows = cur.fetchall()
            cols = [d[0] for d in cur.description]
            result_rows = [dict(zip(cols, row)) for row in rows]
            conn.close()
            truncated = len(result_rows) > MAX_ROWS
            results.append({
                "db": db_name,
                "sql": sql,
                "result": result_rows[:MAX_ROWS],
                "truncated": truncated,
                "total_rows": len(result_rows),
            })
        except Exception as e:
            results.append({
                "db": db_name,
                "sql": sql,
                "result": {"error": str(e)},
                "truncated": False,
                "total_rows": 0,
            })
    return results


def score_task(answer, gold, tolerance) -> bool:
    if isinstance(gold, dict):
        if not isinstance(answer, dict):
            return False
        for key, gold_val in gold.items():
            if key not in answer:
                return False
            if isinstance(tolerance, dict):
                tol = tolerance.get(key)
                if tol is None:
                    return False
            else:
                tol = float(tolerance)
            try:
                if abs(float(answer[key]) - float(gold_val)) > float(tol):
                    return False
            except (TypeError, ValueError):
                return False
        return True
    else:
        if isinstance(tolerance, dict):
            vals = list(tolerance.values())
            if not vals:
                return False
            tol = float(vals[0])
        else:
            tol = float(tolerance)
        try:
            return abs(float(answer) - float(gold)) <= tol
        except (TypeError, ValueError):
            return False


def build_format_hint(gold) -> str:
    if isinstance(gold, dict):
        return "{" + ", ".join(f'"{k}": <float>' for k in gold.keys()) + "}"
    return "<float>"


def build_answer_tool(gold) -> dict:
    """Build an Anthropic tool definition whose input schema matches the gold answer shape."""
    if isinstance(gold, dict):
        props = {k: {"type": "number"} for k in gold.keys()}
        required = list(gold.keys())
    else:
        props = {"answer": {"type": "number"}}
        required = ["answer"]
    return {
        "name": "submit_answer",
        "description": "Submit the final numeric answer.",
        "input_schema": {"type": "object", "properties": props, "required": required},
    }


def build_schema_message(task: dict, db_dir: str) -> str:
    parts = []
    for db_id in task["db_id"]:
        db_path = os.path.join(db_dir, db_id, f"{db_id}.sqlite")
        parts.append(f"=== Database: {db_id} ===")
        parts.append(get_schema(db_path))

    parts.append(f"Question: {task['question']}")
    if task.get("evidence"):
        parts.append(f"Evidence: {task['evidence']}")

    message = (
        "You are a data analyst working with SQLite databases. Given database schemas and a question, "
        "return the SQLite compatible SQL SELECT queries needed to answer it. "
        "Write queries that return only aggregated or pre-joined results needed "
        "for the final computation — never return raw individual rows. "
        "For example, if the question asks for a correlation between two rates, "
        "each query should return one rate per group, not one row per record. "
        'Return ONLY a JSON array: [{"db": "<db_name>", "sql": "<SELECT ...>"}, ...]. '
        "No explanation, no markdown." + "\n\n" + "\n".join(parts)
    )
    
    return message


def build_turn2_message(exec_results: list, format_hint: str) -> str:
    parts = []
    for er in exec_results:
        entry = {"db": er["db"], "sql": er["sql"], "result": er["result"]}
        if er.get("truncated"):
            entry["warning"] = (
                f"Results truncated to {MAX_ROWS} rows "
                f"(full result had {er['total_rows']} rows). "
                "The final answer may be approximate."
            )
        parts.append(entry)
    return (
        "You are a data analyst. Given SQL query results, compute the final numeric answer. "
        f"Return ONLY a JSON object in exactly this format: {format_hint}. "
        "No explanation, no markdown."
        f"Here are the SQL execution results:\n{json.dumps(parts, indent=2)}\n\n"
        "Using these results, compute the final answer to the question."
    )


def run_task(task: dict, db_dir: str, client: anthropic.Anthropic, model: str) -> tuple[dict, list]:
    steps = []

    result_entry = {
        "id": task["id"],
        "difficulty": task.get("difficulty", ""),
        "db_id": task.get("db_id", []),
        "question": task["question"],
        "turn1_sqls": None,
        "execution_results": None,
        "model_answer": None,
        "gold_answer": task["result"],
        "correct": False,
        "error": None,
    }

    gold = task["result"]
    format_hint = build_format_hint(gold)

    # system_prompt_turn1 = (
    #     "You are a data analyst working with SQLite databases. Given database schemas and a question, "
    #     "return the SQL SELECT queries needed to answer it. "
    #     "Write queries that return only aggregated or pre-joined results needed "
    #     "for the final computation — never return raw individual rows. "
    #     "For example, if the question asks for a correlation between two rates, "
    #     "each query should return one rate per group, not one row per record. "
    #     'Return ONLY a JSON array: [{"db": "<db_name>", "sql": "<SELECT ...>"}, ...]. '
    #     "No explanation, no markdown."
    # )

    # system_prompt_turn2 = (
    #     "You are a data analyst. Given SQL query results, compute the final numeric answer. "
    #     f"Return ONLY a JSON object in exactly this format: {format_hint}. "
    #     "No explanation, no markdown."
    # )
    
    messages = []
    turn1_raw = None

    # ── Turn 1 ────────────────────────────────────────────────────────────────
    try:
        turn1_user = build_schema_message(task, db_dir)
        messages.append({"role": "user", "content": turn1_user})
        # messages.append({"role": "assistant", "content": "["})  # prefill

        resp1 = client.messages.create(
            model=model,
            max_tokens=2048,
            temperature=0,
            messages=messages,
        )
        turn1_raw = resp1.content[0].text.strip()
        messages.append({"role": "assistant", "content": turn1_raw})

        sqls = json.loads(strip_markdown(turn1_raw))
        result_entry["turn1_sqls"] = sqls
        steps.append({"step": "turn1", "raw": turn1_raw, "sqls": sqls, "error": None})

    except json.JSONDecodeError as e:
        result_entry["error"] = f"turn1_parse_error: {e}"
        steps.append({"step": "turn1", "raw": turn1_raw, "sqls": None, "error": result_entry["error"]})
        return result_entry, steps
    except Exception as e:
        result_entry["error"] = f"turn1_error: {e}"
        steps.append({"step": "turn1", "raw": turn1_raw, "sqls": None, "error": result_entry["error"]})
        return result_entry, steps

    # ── SQL execution ─────────────────────────────────────────────────────────
    try:
        exec_results = execute_sqls(db_dir, sqls)
        result_entry["execution_results"] = exec_results
        log_exec = [{"db": er["db"], "sql": er["sql"], "result": er["result"][:3]} for er in exec_results]
        steps.append({"step": "execution", "results": log_exec, "error": None})
    except Exception as e:
        result_entry["error"] = f"execution_error: {e}"
        steps.append({"step": "execution", "results": None, "error": result_entry["error"]})
        return result_entry, steps

    # ── Turn 2 ────────────────────────────────────────────────────────────────
    try:
        turn2_user = build_turn2_message(exec_results, format_hint)
        messages.append({"role": "user", "content": turn2_user})

        answer_tool = build_answer_tool(gold)
        resp2 = client.messages.create(
            model=model,
            max_tokens=1024,
            temperature=0,
            messages=messages,
            tools=[answer_tool],
            tool_choice={"type": "any"},
        )

        tool_block = next((b for b in resp2.content if b.type == "tool_use"), None)
        if tool_block is None:
            raise ValueError("model did not call submit_answer tool")

        tool_input = tool_block.input
        # flat float answer: unwrap {"answer": x} → x
        if isinstance(gold, dict):
            model_answer = tool_input
        else:
            model_answer = tool_input["answer"]

        result_entry["model_answer"] = model_answer
        steps.append({"step": "turn2", "tool_input": tool_input, "model_answer": model_answer, "error": None})

    except Exception as e:
        result_entry["error"] = f"turn2_error: {e}"
        steps.append({"step": "turn2", "tool_input": None, "model_answer": None, "error": result_entry["error"]})
        return result_entry, steps

    # ── Scoring ───────────────────────────────────────────────────────────────
    try:
        result_entry["correct"] = score_task(model_answer, gold, task["tolerance"])
    except Exception as e:
        result_entry["error"] = f"scoring_error: {e}"

    steps.append({
        "step": "final",
        "model_answer": result_entry["model_answer"],
        "gold_answer": result_entry["gold_answer"],
        "correct": result_entry["correct"],
        "error": result_entry["error"],
    })
    return result_entry, steps


def compute_summary(all_results: list) -> dict:
    total = len(all_results)
    if total == 0:
        return {}

    correct_total = sum(1 for r in all_results if r["correct"])

    difficulties = {}
    for r in all_results:
        d = r.get("difficulty", "unknown")
        difficulties.setdefault(d, {"total": 0, "correct": 0})
        difficulties[d]["total"] += 1
        if r["correct"]:
            difficulties[d]["correct"] += 1

    exec_success = sum(
        1 for r in all_results
        if r.get("execution_results")
        and len(r["execution_results"]) > 0
        and all(
            not isinstance(er.get("result"), dict) or "error" not in er["result"]
            for er in r["execution_results"]
        )
    )

    truncated_count = sum(
        1 for r in all_results
        if any(
            er.get("truncated")
            for er in (r.get("execution_results") or [])
        )
    )

    return {
        "total": total,
        "correct": correct_total,
        "accuracy": round(correct_total / total, 4),
        "execution_success": exec_success,
        "execution_success_rate": round(exec_success / total, 4),
        "truncated_tasks": truncated_count,
        "truncated_rate": round(truncated_count / total, 4),
        "by_difficulty": {
            d: {
                "total": v["total"],
                "correct": v["correct"],
                "accuracy": round(v["correct"] / v["total"], 4),
            }
            for d, v in sorted(difficulties.items())
        },
    }


def print_summary(summary: dict) -> None:
    total = summary.get("total", 0)
    if total == 0:
        print("No tasks run.")
        return
    correct   = summary["correct"]
    exec_s    = summary["execution_success"]
    truncated = summary["truncated_tasks"]
    print(f"\n{'='*48}")
    print(f"  Total tasks       : {total}")
    print(f"  Overall accuracy  : {correct}/{total} ({100*summary['accuracy']:.1f}%)")
    print(f"  Execution success : {exec_s}/{total} ({100*summary['execution_success_rate']:.1f}%)")
    print(f"  Truncated tasks   : {truncated}/{total} ({100*summary['truncated_rate']:.1f}%)")
    print(f"\n  By difficulty:")
    for diff, counts in summary["by_difficulty"].items():
        t, c = counts["total"], counts["correct"]
        print(f"    {diff:<15} {c}/{t} ({100*counts['accuracy']:.1f}%)")
    print(f"{'='*48}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="dev/task.json")
    parser.add_argument("--db", default="dev/databases")
    parser.add_argument("--model", default="claude-haiku-4-5-20251001")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--ids", nargs="+", type=int)
    args = parser.parse_args()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) if args.output_dir else Path(f"data/baseline_{ts}")
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "results.json"
    log_path     = output_dir / "log.json"
    summary_path = output_dir / "summary.json"
    print(f"Output folder: {output_dir}/")

    with open(args.task) as f:
        tasks = json.load(f)

    tasks = [t for t in tasks if "db_id" in t]
    if args.ids:
        tasks = [t for t in tasks if t["id"] in args.ids]

    client = anthropic.Anthropic()

    all_results = []
    task_logs   = []

    for i, task in enumerate(tasks):
        print(f"[{i+1}/{len(tasks)}] Task {task['id']} ({task.get('difficulty', '')}) ...", end=" ", flush=True)
        try:
            entry, steps = run_task(task, args.db, client, args.model)
        except Exception:
            entry = {
                "id":                task["id"],
                "difficulty":        task.get("difficulty", ""),
                "db_id":             task.get("db_id", []),
                "question":          task["question"],
                "turn1_sqls":        None,
                "execution_results": None,
                "model_answer":      None,
                "gold_answer":       task["result"],
                "correct":           False,
                "error":             f"unexpected_error: {traceback.format_exc()}",
            }
            steps = [{"step": "error", "error": entry["error"]}]

        status = "CORRECT" if entry["correct"] else f"WRONG ({entry.get('error') or 'scoring mismatch'})"
        print(status)

        all_results.append(entry)
        task_logs.append({"task_id": task["id"], "steps": steps})

        # Write incrementally so partial runs are recoverable
        with open(log_path, "w") as f:
            json.dump(task_logs, f, indent=4)

    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=4)

    summary = compute_summary(all_results)
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=4)

    print(f"Results  → {results_path}")
    print(f"Log      → {log_path}")
    print(f"Summary  → {summary_path}")
    print_summary(summary)


if __name__ == "__main__":
    main()