import argparse
import json
import math
import os
import re
import sqlite3
import subprocess
import sys
import tempfile
import traceback
from datetime import datetime
from pathlib import Path

import anthropic
import openai
from dotenv import load_dotenv

load_dotenv()

MAX_ROWS = 500

PRICING = {
    "claude-sonnet-4-6": {"input": 3.0,  "cache_write": 3.75, "cache_read": 0.30, "output": 15.0},
    "claude-opus-4-7":   {"input": 5.0,  "cache_write": 6.25, "cache_read": 0.50, "output": 25.0},
    "claude-haiku-4-5":  {"input": 1.0,  "cache_write": 1.25, "cache_read": 0.10, "output": 5.0},
    "gpt-5.4":           {"input": 2.50, "cache_write": 0.0,  "cache_read": 0.25, "output": 15.0},
}

MODEL_METADATA   = "claude-haiku-4-5-20251001"
# MODEL_SELECTOR   = "claude-sonnet-4-6"
# MODEL_SQL        = "claude-sonnet-4-6"
# MODEL_RESULT     = "claude-sonnet-4-6"
MODEL_SELECTOR   = "claude-opus-4-7"
MODEL_SQL        = "claude-opus-4-7"
MODEL_RESULT     = "claude-opus-4-7"
MAX_RETRIES      = 3
METADATA_CACHE   = "data/metadata_cache.json"


def get_schema(db_path: str) -> str:
    with sqlite3.connect(db_path) as conn:
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

        return "\n".join(parts)


def get_db_table_names(db_dir: str) -> dict:
    """Returns {db_name: [table_names]} without loading full schemas or sample rows."""
    result = {}
    for entry in os.scandir(db_dir):
        if entry.is_dir():
            sqlite_path = os.path.join(entry.path, f"{entry.name}.sqlite")
            if os.path.isfile(sqlite_path):
                try:
                    with sqlite3.connect(sqlite_path) as conn:
                        cur = conn.cursor()
                        cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")
                        result[entry.name] = [r[0] for r in cur.fetchall()]
                except Exception:
                    pass
    return result


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
    if 'last_valid' in locals():
        return last_valid
    s, e = text.find('['), text.rfind(']')
    if s != -1 and e != -1:
        return text[s:e+1]
    return text


def strip_code_block(text: str) -> str:
    """Extract code from a markdown code block, or return text as-is."""
    text = text.strip()
    match = re.search(r"```(?:python)?\s*\n?([\s\S]*?)\n?```", text, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return text


def execute_python_code(code: str) -> dict:
    """Write code to a temp file, execute it, parse stdout as JSON."""
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(code)
            tmp_path = f.name
        proc = subprocess.run(
            [sys.executable, tmp_path],
            capture_output=True, text=True, timeout=30
        )
        if proc.returncode != 0:
            return {"error": f"non-zero exit ({proc.returncode}): {proc.stderr.strip()}"}
        stdout = proc.stdout.strip()
        if not stdout:
            return {"error": "no output from script"}
        try:
            return json.loads(stdout)
        except json.JSONDecodeError as e:
            return {"error": f"json parse failed: {e} | stdout was: {stdout[:200]}"}
    except subprocess.TimeoutExpired:
        return {"error": "execution timeout (30s)"}
    except Exception as e:
        return {"error": str(e)}
    finally:
        if tmp_path:
            try:
                os.unlink(tmp_path)
            except Exception:
                pass


def execute_sqls(db_dir: str, sqls: list) -> list:
    results = []
    for item in sqls:
        db_name = item.get("db", "")
        sql = item.get("sql", "")
        db_path = os.path.join(db_dir, db_name, f"{db_name}.sqlite")
        try:
            conn = sqlite3.connect(db_path)
            conn.row_factory = sqlite3.Row
            try:
                cur = conn.cursor()
                cur.execute(sql)
                rows = cur.fetchall()
                cols = [d[0] for d in cur.description]
                result_rows = [dict(zip(cols, row)) for row in rows]
            finally:
                conn.close()
            result_rows.sort(key=str)
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
        gold_res_truncated = gold_res[:MAX_ROWS] if isinstance(gold_res, list) else gold_res
        matched = any(results_match(gold_res_truncated, mr) for mr in model_results)
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


def _add_usage(a: dict, b: dict) -> dict:
    return {k: a.get(k, 0) + b.get(k, 0) for k in set(a) | set(b)}


def score_task(answer, gold, tolerance) -> bool:
    if isinstance(gold, dict):
        if not isinstance(answer, dict):
            return False
        for key, gold_val in gold.items():
            if key not in answer:
                # Key name mismatch: if both dicts are single-entry, compare values positionally
                if len(gold) == 1 and len(answer) == 1:
                    ans_val = next(iter(answer.values()))
                else:
                    return False
            else:
                ans_val = answer[key]
            if isinstance(tolerance, dict):
                tol = tolerance.get(key)
                if tol is None:
                    return False
            else:
                tol = float(tolerance)
            try:
                a, g = float(ans_val), float(gold_val)
                if math.isnan(a) or math.isnan(g):
                    return False
                if abs(a - g) > float(tol):
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
            a, g = float(answer), float(gold)
            if math.isnan(a) or math.isnan(g):
                return False
            return abs(a - g) <= tol
        except (TypeError, ValueError):
            return False


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
    isql_tasks = [r for r in all_results if (r.get("intermediate_sqls_check") or {}).get("available")]
    isql_correct = sum(1 for r in isql_tasks if r["intermediate_sqls_check"].get("all_matched"))

    # DB selection stats
    db_sel_correct = sum(1 for r in all_results if r.get("db_selection_correct"))

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
        "db_selection": {
            "correct": db_sel_correct,
            "total": total,
            "accuracy": round(db_sel_correct / total, 4),
        },
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
    db_sel    = summary.get("db_selection", {})
    isql      = summary.get("intermediate_sqls", {})
    usage     = summary.get("token_usage", {})
    cost      = summary.get("cost_usd")

    print(f"\n{'='*52}")
    print(f"  Total tasks            : {total}")
    print(f"  Overall accuracy       : {correct}/{total} ({100*summary['accuracy']:.1f}%)")
    if db_sel:
        ds_c = db_sel["correct"]
        print(f"  DB selection accuracy  : {ds_c}/{total} ({100*db_sel['accuracy']:.1f}%)")
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


def call_api(client, model: str, messages: list, max_tokens: int) -> tuple[str, dict]:
    """Call Anthropic or OpenAI and return (text, usage_dict)."""
    if isinstance(client, anthropic.Anthropic):
        resp = client.messages.create(model=model, max_tokens=max_tokens, messages=messages)
        text = resp.content[0].text.strip()
        usage = {
            "input_tokens":                getattr(resp.usage, "input_tokens", 0),
            "output_tokens":               getattr(resp.usage, "output_tokens", 0),
            "cache_creation_input_tokens": getattr(resp.usage, "cache_creation_input_tokens", 0),
            "cache_read_input_tokens":     getattr(resp.usage, "cache_read_input_tokens", 0),
        }
    else:
        resp = client.chat.completions.create(model=model, max_completion_tokens=max_tokens, messages=messages)
        text = resp.choices[0].message.content.strip()
        usage = {
            "input_tokens":                getattr(resp.usage, "prompt_tokens", 0),
            "output_tokens":               getattr(resp.usage, "completion_tokens", 0),
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens":     0,
        }
    return text, usage


def build_turn2_message(sqls: list, db_dir: str, question: str, evidence: str) -> str:
    """Build the result generator prompt using SQL queries and db paths, not pre-executed rows."""
    sql_entries = []
    for item in sqls:
        db_name = item.get("db", "")
        db_path = os.path.join(db_dir, db_name, f"{db_name}.sqlite")
        sql_entries.append({
            "db": db_name,
            "db_path": db_path,
            "sql": item.get("sql", ""),
        })

    evidence_line = f"\nEvidence: {evidence}" if evidence else ""

    return (
        "You are a data analyst. Given SQL queries and their database paths, "
        "write a self-contained Python script that executes the queries and "
        "computes the final numeric answer.\n\n"
        "The script must:\n"
        "- Use sqlite3 to execute each query against its database path directly\n"
        "- Fetch ALL rows — do not limit or truncate results\n"
        "- Import only from the standard library, scipy, numpy, and pandas\n"
        "- Compute the final answer from the full query results\n"
        "- Print exactly one JSON object to stdout with the final answer\n"
        "- Produce no other output — no debug prints, no warnings\n\n"
        f"Question: {question}{evidence_line}\n\n"
        f"Queries to execute:\n{json.dumps(sql_entries, indent=2)}\n\n"
        "Write the Python script now. Output only the script, no explanation."
    )


# ── Agent-specific functions ──────────────────────────────────────────────────

def extract_metadata(db_dir: str, anthropic_client: anthropic.Anthropic) -> dict:
    """Generate and cache semantic summaries for all databases using Haiku."""
    cache_path = Path(METADATA_CACHE)
    if cache_path.exists():
        with open(cache_path) as f:
            return json.load(f)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    db_table_names = get_db_table_names(db_dir)
    metadata = {}

    for db_name in sorted(db_table_names):
        db_path = os.path.join(db_dir, db_name, f"{db_name}.sqlite")
        schema = get_schema(db_path)
        prompt = (
            "You are a database analyst. Given a database schema, produce a structured JSON summary.\n\n"
            f"Database: {db_name}\n\n"
            f"Schema:\n{schema}\n\n"
            "Return ONLY a JSON object with these fields:\n"
            '{\n'
            '  "name": "<db_name>",\n'
            '  "domain": "<one phrase describing the domain>",\n'
            '  "tables": ["<table1>", "<table2>", ...],\n'
            '  "summary": "<2-3 sentences describing what questions this database can answer>"\n'
            '}\n'
            "No explanation, no markdown."
        )
        resp = anthropic_client.messages.create(
            model=MODEL_METADATA,
            max_tokens=512,
            messages=[{"role": "user", "content": prompt}],
        )
        text = resp.content[0].text.strip()
        try:
            s = text.find("{"); e = text.rfind("}")
            metadata[db_name] = json.loads(text[s:e+1])
        except Exception:
            metadata[db_name] = {
                "name": db_name,
                "domain": "unknown",
                "tables": db_table_names[db_name],
                "summary": "",
            }
        print(f"  Metadata: {db_name} ✓", flush=True)

    with open(cache_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Metadata cached → {cache_path}")
    return metadata


def build_selector_message(metadata: dict, question: str, evidence: str) -> str:
    db_lines = []
    for db_name in sorted(metadata):
        m = metadata[db_name]
        db_lines.append(
            f"- {db_name}: domain={m.get('domain','')}, "
            f"tables={', '.join(m.get('tables', []))}, "
            f"summary={m.get('summary','')}"
        )
    evidence_line = f"\nEvidence: {evidence}" if evidence else ""
    return (
        "You are a data analyst. Given database descriptions and a question, "
        "select only the databases needed to answer the question.\n"
        'Return ONLY a JSON array of database name strings, e.g. ["db1", "db2"]. '
        "No explanation, no markdown.\n\n"
        "Available databases:\n"
        + "\n".join(db_lines)
        + f"\n\nQuestion: {question}{evidence_line}"
    )


def build_sql_message(
    selected_dbs: list,
    db_dir: str,
    question: str,
    evidence: str,
    error_feedback: str = None,
) -> str:
    parts = []
    for db_id in selected_dbs:
        db_path = os.path.join(db_dir, db_id, f"{db_id}.sqlite")
        parts.append(f"=== Database: {db_id} ===")
        parts.append(get_schema(db_path))

    parts.append(f"Question: {question}")
    if evidence:
        parts.append(f"Evidence: {evidence}")

    base = (
        "You are a data analyst working with SQLite databases. Given database schemas and a question, "
        "return the SQLite compatible SQL SELECT queries needed to answer it. "
        "Write queries that return only aggregated or pre-joined results needed for the final "
        "computation — never return raw individual rows. "
        "For example, if the question asks for a correlation between two rates, "
        "each query should return one rate per group, not one row per record. "
        'Return ONLY a JSON array: [{"db": "<db_name>", "sql": "<SELECT ...>"}, ...]. '
        "No explanation, no markdown.\n\n"
        + "\n".join(parts)
    )

    if error_feedback:
        base += (
            f"\n\nThe previous attempt produced the following errors. "
            f"Fix the queries:\n{error_feedback}"
        )
    return base


def run_task(
    task: dict,
    db_dir: str,
    anthropic_client: anthropic.Anthropic,
    model: str,
    metadata: dict,
) -> tuple[dict, list]:
    steps = []
    token_usage = {"input_tokens": 0, "output_tokens": 0,
                   "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0}

    result_entry = {
        "id": task["id"],
        "difficulty": task.get("difficulty", ""),
        "db_id": task.get("db_id", []),
        "question": task["question"],
        "selected_dbs": None,
        "db_selection_correct": None,
        "turn1_sqls": None,
        "sql_retries": 0,
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
    question = task["question"]
    evidence = task.get("evidence", "")

    # ── Agent 1: Database Selector ────────────────────────────────────────────
    sel_raw = None
    try:
        sel_msg = build_selector_message(metadata, question, evidence)
        sel_raw, usage = call_api(
            anthropic_client, MODEL_SELECTOR,
            [{"role": "user", "content": sel_msg}], 1024
        )
        token_usage = _add_usage(token_usage, usage)

        selected_dbs = json.loads(strip_markdown(sel_raw))
        selected_dbs = list(dict.fromkeys(selected_dbs))

        if not selected_dbs:
            raise ValueError("empty db selection")
        unknown = [db for db in selected_dbs if db not in metadata]
        if unknown:
            raise ValueError(f"unknown databases: {unknown}")

        result_entry["selected_dbs"] = selected_dbs
        result_entry["db_selection_correct"] = set(selected_dbs) == set(task.get("db_id", []))
        steps.append({"step": "selector", "selected_dbs": selected_dbs, "error": None})

    except Exception as e:
        result_entry["error"] = f"selector_failed: {e}"
        result_entry["token_usage"] = token_usage
        result_entry["cost_usd"] = compute_cost(token_usage, MODEL_SELECTOR)
        steps.append({"step": "selector", "selected_dbs": None, "error": result_entry["error"]})
        return result_entry, steps

    # ── Agent 2: SQL Generator with retry loop ────────────────────────────────
    sqls = None
    exec_results = None
    error_feedback = None

    for attempt in range(MAX_RETRIES + 1):
        sql_raw = None
        try:
            sql_msg = build_sql_message(
                selected_dbs, db_dir, question, evidence, error_feedback
            )
            sql_raw, usage = call_api(
                anthropic_client, MODEL_SQL,
                [{"role": "user", "content": sql_msg}], 4096
            )
            token_usage = _add_usage(token_usage, usage)

            sqls = json.loads(strip_markdown(sql_raw))
            steps.append({"step": f"sql_gen_attempt_{attempt}", "sqls": sqls, "error": None})

        except Exception as e:
            steps.append({"step": f"sql_gen_attempt_{attempt}", "sqls": None, "error": str(e)})
            error_feedback = f"SQL parse error: {e}"
            result_entry["sql_retries"] = attempt + 1
            continue

        # Execute and check for errors
        exec_results = execute_sqls(db_dir, sqls)
        errors = [
            f"db={er['db']} sql={er['sql'][:80]} error={er['result']['error']}"
            for er in exec_results
            if isinstance(er.get("result"), dict) and "error" in er["result"]
        ]
        empty = [
            f"db={er['db']} returned 0 rows"
            for er in exec_results
            if isinstance(er.get("result"), list) and len(er["result"]) == 0
        ]

        if not errors and not empty:
            result_entry["sql_retries"] = attempt
            break

        issues = errors + empty
        error_feedback = "Issues with previous queries:\n" + "\n".join(issues)
        result_entry["sql_retries"] = attempt + 1
        steps.append({"step": f"sql_retry_{attempt}", "issues": issues})

        if attempt == MAX_RETRIES:
            result_entry["error"] = f"sql_failed_after_{MAX_RETRIES}_retries: {issues}"
            result_entry["token_usage"] = token_usage
            result_entry["cost_usd"] = compute_cost(token_usage, MODEL_SQL)
            return result_entry, steps

    result_entry["turn1_sqls"] = sqls
    result_entry["execution_results"] = exec_results

    # ── Intermediate SQL check ────────────────────────────────────────────────
    try:
        isql_check = check_intermediate_sqls(task, exec_results, db_dir)
        result_entry["intermediate_sqls_check"] = isql_check
        steps.append({"step": "intermediate_sql_check", "check": isql_check, "error": None})
    except Exception as e:
        steps.append({"step": "intermediate_sql_check", "check": None, "error": str(e)})

    # ── Agent 3: Result Generator ─────────────────────────────────────────────
    result_raw = None
    try:
        result_msg = build_turn2_message(sqls, db_dir, question, evidence)
        result_raw, usage = call_api(
            anthropic_client, MODEL_RESULT,
            [{"role": "user", "content": result_msg}], 4096
        )
        token_usage = _add_usage(token_usage, usage)

        code = strip_code_block(result_raw)
        steps.append({"step": "result_gen_code", "code": code, "error": None})

        exec_output = execute_python_code(code)
        if "error" in exec_output:
            raise ValueError(f"code execution failed: {exec_output['error']}")

        if isinstance(gold, dict):
            model_answer = exec_output
        else:
            if "answer" in exec_output:
                model_answer = exec_output["answer"]
            elif len(exec_output) == 1:
                model_answer = next(iter(exec_output.values()))
            else:
                model_answer = exec_output

        result_entry["model_answer"] = model_answer
        steps.append({"step": "result_gen", "model_answer": model_answer, "error": None})

    except Exception as e:
        result_entry["error"] = f"result_gen_error: {e}"
        result_entry["token_usage"] = token_usage
        result_entry["cost_usd"] = compute_cost(token_usage, MODEL_RESULT)
        steps.append({"step": "result_gen", "model_answer": None, "error": result_entry["error"]})
        return result_entry, steps

    # ── Scoring ───────────────────────────────────────────────────────────────
    try:
        result_entry["correct"] = score_task(model_answer, gold, task["tolerance"])
    except Exception as e:
        result_entry["error"] = f"scoring_error: {e}"

    result_entry["token_usage"] = token_usage
    result_entry["cost_usd"] = round(
        compute_cost(token_usage, MODEL_SQL), 6
    )

    steps.append({
        "step": "final",
        "model_answer": result_entry["model_answer"],
        "gold_answer": result_entry["gold_answer"],
        "correct": result_entry["correct"],
        "sql_retries": result_entry["sql_retries"],
        "error": result_entry["error"],
        "token_usage": token_usage,
        "cost_usd": result_entry["cost_usd"],
    })
    return result_entry, steps


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="dev/task.json")
    parser.add_argument("--db", default="dev/databases")
    parser.add_argument("--model", default="claude-sonnet-4-6")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--ids", nargs="+", type=int)
    parser.add_argument("--start", type=int, default=None, help="Skip tasks with id < this value")
    parser.add_argument("--summarize", action="store_true", help="Recompute summary.json from existing results.json and exit")
    parser.add_argument("--reprice", action="store_true", help="Recompute cost_usd in results.json using current PRICING, then rewrite summary.json")
    args = parser.parse_args()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) if args.output_dir else Path(f"data/agent_{ts}")
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "results.json"
    log_path     = output_dir / "log.json"
    summary_path = output_dir / "summary.json"
    print(f"Output folder: {output_dir}/")

    if args.summarize:
        with open(results_path) as f:
            all_results = json.load(f)
        summary = compute_summary(all_results, model=args.model)
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=4)
        print(f"Summary  → {summary_path}")
        print_summary(summary)
        return

    if args.reprice:
        with open(results_path) as f:
            all_results = json.load(f)
        for r in all_results:
            if r.get("token_usage"):
                r["cost_usd"] = round(compute_cost(r["token_usage"], args.model), 6)
        with open(results_path, "w") as f:
            json.dump(all_results, f, indent=4)
        summary = compute_summary(all_results, model=args.model)
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=4)
        print(f"Results  → {results_path}")
        print(f"Summary  → {summary_path}")
        print_summary(summary)
        return

    with open(args.task) as f:
        tasks = json.load(f)

    tasks = [t for t in tasks if "db_id" in t]
    if args.ids:
        tasks = [t for t in tasks if t["id"] in args.ids]
    if args.start is not None:
        tasks = [t for t in tasks if t["id"] >= args.start]

    print("Extracting database metadata (cached after first run)...")
    anthropic_client = anthropic.Anthropic()
    metadata = extract_metadata(args.db, anthropic_client)

    client = anthropic_client

    all_results = []
    task_logs   = []

    for i, task in enumerate(tasks):
        print(f"[{i+1}/{len(tasks)}] Task {task['id']} ({task.get('difficulty', '')}) ...", end=" ", flush=True)
        try:
            entry, steps = run_task(task, args.db, client, args.model, metadata)
        except Exception:
            entry = {
                "id":                      task["id"],
                "difficulty":              task.get("difficulty", ""),
                "db_id":                   task.get("db_id", []),
                "question":                task["question"],
                "selected_dbs":            None,
                "db_selection_correct":    None,
                "turn1_sqls":              None,
                "sql_retries":             None,
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

        retry_str = f" | retries={entry.get('sql_retries', 0)}"
        cost_str = f" | ${entry['cost_usd']:.5f}" if entry.get("cost_usd") is not None else ""
        status = "CORRECT" if entry["correct"] else f"WRONG ({entry.get('error') or 'scoring mismatch'})"
        print(f"{status}{isql_str}{retry_str}{cost_str}")

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
