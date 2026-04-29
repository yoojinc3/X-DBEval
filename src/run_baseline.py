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

PRICING = {
    "claude-sonnet-4-6": {"input": 3.0,  "cache_write": 3.75, "cache_read": 0.30, "output": 15.0},
    "claude-opus-4-7":   {"input": 5.0,  "cache_write": 6.25, "cache_read": 0.50, "output": 25.0},
}


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
    match = re.search(r"```(?:json)?\s*\n?([\s\S]*?)\n?```", text, re.IGNORECASE)
    if match:
        candidate = match.group(1).strip()
        try:
            json.loads(candidate)
            return candidate
        except: pass
    for m in re.finditer(r'\[[\s\S]*?\](?=\s*(?:\n|$|[^\],\s]))', text):
        try:
            json.loads(m.group())
            last_valid = m.group()
        except: pass
    if 'last_valid' in dir():
        return last_valid
    s, e = text.find('['), text.rfind(']')
    if s != -1 and e != -1:
        return text[s:e+1]
    return text


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


def normalize_rows(result) -> list:
    """Normalize result rows for order-independent comparison."""
    if not isinstance(result, list):
        return []
    rows = []
    for row in result:
        if isinstance(row, dict):
            rows.append(tuple(sorted(
                (k, round(float(v), 4) if isinstance(v, (int, float)) else str(v))
                for k, v in row.items()
            )))
        else:
            rows.append(str(row))
    return sorted(str(r) for r in rows)


def results_match(r1, r2) -> bool:
    return normalize_rows(r1) == normalize_rows(r2)


def check_intermediate_sqls(task: dict, exec_results: list, db_dir: str) -> dict:
    """Execute gold intermediate_sqls and compare with model-generated SQL results."""
    gold_sqls = task.get("intermediate_sqls", [])
    if not gold_sqls:
        return {"available": False, "all_matched": None, "details": []}

    gold_exec = execute_sqls(db_dir, gold_sqls)
    model_results = [er["result"] for er in exec_results]

    details = []
    for gold_er in gold_exec:
        gold_res = gold_er["result"]
        if isinstance(gold_res, dict) and "error" in gold_res:
            details.append({"gold_sql": gold_er["sql"], "matched": False, "reason": "gold_sql_error"})
            continue
        matched = any(results_match(gold_res, mr) for mr in model_results)
        details.append({"gold_sql": gold_er["sql"], "matched": matched})

    return {
        "available": True,
        "all_matched": all(d["matched"] for d in details),
        "details": details,
    }


def compute_cost(usage_dict: dict, model: str) -> float:
    prices = next((v for k, v in PRICING.items() if k in model), PRICING["claude-sonnet-4-6"])
    return (
        usage_dict.get("input_tokens", 0)                    / 1_000_000 * prices["input"]
        + usage_dict.get("cache_creation_input_tokens", 0)   / 1_000_000 * prices["cache_write"]
        + usage_dict.get("cache_read_input_tokens", 0)        / 1_000_000 * prices["cache_read"]
        + usage_dict.get("output_tokens", 0)                  / 1_000_000 * prices["output"]
    )


def _usage_to_dict(usage) -> dict:
    return {
        "input_tokens":                  getattr(usage, "input_tokens", 0),
        "output_tokens":                 getattr(usage, "output_tokens", 0),
        "cache_creation_input_tokens":   getattr(usage, "cache_creation_input_tokens", 0),
        "cache_read_input_tokens":       getattr(usage, "cache_read_input_tokens", 0),
    }


def _add_usage(a: dict, b: dict) -> dict:
    return {k: a.get(k, 0) + b.get(k, 0) for k in set(a) | set(b)}


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
    token_usage = {"input_tokens": 0, "output_tokens": 0,
                   "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0}

    result_entry = {
        "id": task["id"],
        "difficulty": task.get("difficulty", ""),
        "db_id": task.get("db_id", []),
        "question": task["question"],
        "turn1_sqls": None,
        "execution_results": None,
        "intermediate_sqls_check": None,
        "model_answer": None,
        "gold_answer": task["result"],
        "correct": False,
        "error": None,
        "token_usage": None,
        "cost_usd": None,
    }

    gold = task["result"]
    format_hint = build_format_hint(gold)

    messages = []
    turn1_raw = None

    # ── Turn 1 ────────────────────────────────────────────────────────────────
    try:
        turn1_user = build_schema_message(task, db_dir)
        messages.append({"role": "user", "content": turn1_user})

        resp1 = client.messages.create(
            model=model,
            max_tokens=4096,
            messages=messages,
        )
        token_usage = _add_usage(token_usage, _usage_to_dict(resp1.usage))
        turn1_raw = resp1.content[0].text.strip()
        messages.append({"role": "assistant", "content": turn1_raw})

        sqls = json.loads(strip_markdown(turn1_raw))
        result_entry["turn1_sqls"] = sqls
        steps.append({"step": "turn1", "raw": turn1_raw, "sqls": sqls, "error": None})

    except json.JSONDecodeError as e:
        result_entry["error"] = f"turn1_parse_error: {e}"
        result_entry["token_usage"] = token_usage
        result_entry["cost_usd"] = compute_cost(token_usage, model)
        steps.append({"step": "turn1", "raw": turn1_raw, "sqls": None, "error": result_entry["error"]})
        return result_entry, steps
    except Exception as e:
        result_entry["error"] = f"turn1_error: {e}"
        result_entry["token_usage"] = token_usage
        result_entry["cost_usd"] = compute_cost(token_usage, model)
        steps.append({"step": "turn1", "raw": turn1_raw, "sqls": None, "error": result_entry["error"]})
        return result_entry, steps

    # ── SQL execution ─────────────────────────────────────────────────────────
    try:
        exec_results = execute_sqls(db_dir, sqls)
        result_entry["execution_results"] = exec_results
        log_exec = [
            {
                "db": er["db"],
                "sql": er["sql"],
                "result": er["result"][:3] if isinstance(er["result"], list) else er["result"],
            }
            for er in exec_results
        ]
        steps.append({"step": "execution", "results": log_exec, "error": None})
    except Exception as e:
        result_entry["error"] = f"execution_error: {e}"
        result_entry["token_usage"] = token_usage
        result_entry["cost_usd"] = compute_cost(token_usage, model)
        steps.append({"step": "execution", "results": None, "error": result_entry["error"]})
        return result_entry, steps

    # ── Intermediate SQL check ────────────────────────────────────────────────
    try:
        isql_check = check_intermediate_sqls(task, exec_results, db_dir)
        result_entry["intermediate_sqls_check"] = isql_check
        steps.append({"step": "intermediate_sql_check", "check": isql_check, "error": None})
    except Exception as e:
        steps.append({"step": "intermediate_sql_check", "check": None, "error": str(e)})

    # ── Turn 2 ────────────────────────────────────────────────────────────────
    try:
        turn2_user = build_turn2_message(exec_results, format_hint)
        messages.append({"role": "user", "content": turn2_user})

        answer_tool = build_answer_tool(gold)
        resp2 = client.messages.create(
            model=model,
            max_tokens=4096,
            messages=messages,
            tools=[answer_tool],
            tool_choice={"type": "any"},
        )
        token_usage = _add_usage(token_usage, _usage_to_dict(resp2.usage))

        tool_block = next((b for b in resp2.content if b.type == "tool_use"), None)
        if tool_block is None:
            raise ValueError("model did not call submit_answer tool")

        tool_input = tool_block.input
        if isinstance(gold, dict):
            model_answer = tool_input
        else:
            model_answer = tool_input["answer"]

        result_entry["model_answer"] = model_answer
        steps.append({"step": "turn2", "tool_input": tool_input, "model_answer": model_answer, "error": None})

    except Exception as e:
        result_entry["error"] = f"turn2_error: {e}"
        result_entry["token_usage"] = token_usage
        result_entry["cost_usd"] = compute_cost(token_usage, model)
        steps.append({"step": "turn2", "tool_input": None, "model_answer": None, "error": result_entry["error"]})
        return result_entry, steps

    # ── Scoring ───────────────────────────────────────────────────────────────
    try:
        result_entry["correct"] = score_task(model_answer, gold, task["tolerance"])
    except Exception as e:
        result_entry["error"] = f"scoring_error: {e}"

    result_entry["token_usage"] = token_usage
    result_entry["cost_usd"] = round(compute_cost(token_usage, model), 6)

    steps.append({
        "step": "final",
        "model_answer": result_entry["model_answer"],
        "gold_answer": result_entry["gold_answer"],
        "correct": result_entry["correct"],
        "error": result_entry["error"],
        "token_usage": token_usage,
        "cost_usd": result_entry["cost_usd"],
    })
    return result_entry, steps


def compute_summary(all_results: list, model: str = "") -> dict:
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

    # Intermediate SQL check stats
    isql_tasks = [r for r in all_results if r.get("intermediate_sqls_check", {}).get("available")]
    isql_correct = sum(1 for r in isql_tasks if r["intermediate_sqls_check"].get("all_matched"))

    # Token usage totals
    total_usage: dict = {"input_tokens": 0, "output_tokens": 0,
                         "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0}
    for r in all_results:
        if r.get("token_usage"):
            total_usage = _add_usage(total_usage, r["token_usage"])
    total_cost = round(compute_cost(total_usage, model), 4) if model else None

    return {
        "total": total,
        "correct": correct_total,
        "accuracy": round(correct_total / total, 4),
        "execution_success": exec_success,
        "execution_success_rate": round(exec_success / total, 4),
        "truncated_tasks": truncated_count,
        "truncated_rate": round(truncated_count / total, 4),
        "intermediate_sqls": {
            "tasks_with_gold": len(isql_tasks),
            "all_matched": isql_correct,
            "match_rate": round(isql_correct / len(isql_tasks), 4) if isql_tasks else None,
        },
        "token_usage": total_usage,
        "cost_usd": total_cost,
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
    isql      = summary.get("intermediate_sqls", {})
    usage     = summary.get("token_usage", {})
    cost      = summary.get("cost_usd")

    print(f"\n{'='*52}")
    print(f"  Total tasks            : {total}")
    print(f"  Overall accuracy       : {correct}/{total} ({100*summary['accuracy']:.1f}%)")
    print(f"  Execution success      : {exec_s}/{total} ({100*summary['execution_success_rate']:.1f}%)")
    print(f"  Truncated tasks        : {truncated}/{total} ({100*summary['truncated_rate']:.1f}%)")

    if isql.get("tasks_with_gold", 0) > 0:
        n = isql["tasks_with_gold"]
        m = isql["all_matched"]
        rate = isql["match_rate"]
        print(f"  Intermediate SQL match : {m}/{n} ({100*rate:.1f}%)")

    if usage:
        inp  = usage.get("input_tokens", 0)
        out  = usage.get("output_tokens", 0)
        cw   = usage.get("cache_creation_input_tokens", 0)
        cr   = usage.get("cache_read_input_tokens", 0)
        print(f"\n  Token usage:")
        print(f"    Input tokens         : {inp:,}")
        print(f"    Output tokens        : {out:,}")
        if cw:
            print(f"    Cache write tokens   : {cw:,}")
        if cr:
            print(f"    Cache read tokens    : {cr:,}")
        if cost is not None:
            print(f"    Total cost           : ${cost:.4f} USD")

    print(f"\n  By difficulty:")
    for diff, counts in summary["by_difficulty"].items():
        t, c = counts["total"], counts["correct"]
        print(f"    {diff:<15} {c}/{t} ({100*counts['accuracy']:.1f}%)")
    print(f"{'='*52}\n")


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
                "id":                      task["id"],
                "difficulty":              task.get("difficulty", ""),
                "db_id":                   task.get("db_id", []),
                "question":                task["question"],
                "turn1_sqls":              None,
                "execution_results":       None,
                "intermediate_sqls_check": None,
                "model_answer":            None,
                "gold_answer":             task["result"],
                "correct":                 False,
                "error":                   f"unexpected_error: {traceback.format_exc()}",
                "token_usage":             None,
                "cost_usd":                None,
            }
            steps = [{"step": "error", "error": entry["error"]}]

        isql_check = entry.get("intermediate_sqls_check") or {}
        isql_str = ""
        if isql_check.get("available"):
            isql_str = f" | intermediate_sqls: {'OK' if isql_check.get('all_matched') else 'MISMATCH'}"

        cost_str = f" | ${entry['cost_usd']:.5f}" if entry.get("cost_usd") is not None else ""
        status = "CORRECT" if entry["correct"] else f"WRONG ({entry.get('error') or 'scoring mismatch'})"
        print(f"{status}{isql_str}{cost_str}")

        all_results.append(entry)
        task_logs.append({"task_id": task["id"], "steps": steps})

        # Write incrementally so partial runs are recoverable
        with open(log_path, "w") as f:
            json.dump(task_logs, f, indent=4)

    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=4)

    summary = compute_summary(all_results, model=args.model)
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=4)

    print(f"Results  → {results_path}")
    print(f"Log      → {log_path}")
    print(f"Summary  → {summary_path}")
    print_summary(summary)


if __name__ == "__main__":
    main()
