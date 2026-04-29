#!/usr/bin/env python3
"""Run the cs498-dku-agent benchmark against dev/task.json and save run artifacts.

The runner imports the agent implementation from the sibling `cs498-dku-agent`
checkout, patches its structured LLM helper to track token usage, executes the
pipeline task-by-task, and writes a detailed JSON report into `cs498/runs/`.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sqlite3
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
AGENT_REPO = PROJECT_ROOT / "cs498-dku-agent"
DEFAULT_TASK_JSON = SCRIPT_DIR / "dev" / "task.json"
DEFAULT_DB_DIR = PROJECT_ROOT / "data" / "train" / "train_databases"
RUNS_DIR = SCRIPT_DIR / "runs"
WORKSPACE_DATA_ROOTS = [
    PROJECT_ROOT / "data",
    PROJECT_ROOT / "temp_train_data",
]

if str(AGENT_REPO) not in sys.path:
    sys.path.insert(0, str(AGENT_REPO))

import agent.agents.metadata as _meta_mod  # noqa: E402
import agent.agents.result_gen as _result_mod  # noqa: E402
import agent.agents.selector as _selector_mod  # noqa: E402
import agent.agents.sql_gen as _sql_mod  # noqa: E402
import agent.llm as _llm_mod  # noqa: E402
from agent.answer_schema import schema_from_example  # noqa: E402
from agent.main import is_empty_or_null, load_or_extract_metadata  # noqa: E402
from agent.models import DBSummary, QueryResult, SQLPlan  # noqa: E402
from agent.tools.db import execute_queries  # noqa: E402


logger = logging.getLogger(__name__)

_PRICING: dict[str, tuple[float, float]] = {
    "claude-haiku-4-5": (0.80, 4.00),
    "claude-haiku-4-5-20251001": (0.80, 4.00),
    "claude-sonnet-4-5": (3.00, 15.00),
    "claude-sonnet-4-5-20251001": (3.00, 15.00),
}

_task_usage: list[dict[str, Any]] = []


def _cost_usd(model: str, inp: int, out: int) -> float:
    in_rate, out_rate = _PRICING.get(model, (3.00, 15.00))
    return (inp * in_rate + out * out_rate) / 1_000_000


def _tracked_call_structured[T: Any](
    model: str,
    system: str,
    user: str,
    response_model: type[T],
    max_tokens: int = 2048,
) -> T:
    schema = response_model.model_json_schema()
    response = _llm_mod.get_client().messages.create(
        model=model,
        max_tokens=max_tokens,
        system=system,
        tools=[{
            "name": "submit_answer",
            "description": f"Submit a validated {response_model.__name__} object.",
            "input_schema": schema,
        }],
        tool_choice={"type": "tool", "name": "submit_answer"},
        messages=[{"role": "user", "content": user}],
    )
    _task_usage.append({
        "model": model,
        "input": response.usage.input_tokens,
        "output": response.usage.output_tokens,
    })
    tool_use = next(block for block in response.content if block.type == "tool_use")
    return response_model.model_validate(tool_use.input)


for _mod in (_meta_mod, _selector_mod, _sql_mod, _result_mod):
    _mod.call_structured = _tracked_call_structured  # type: ignore[attr-defined]


def _numeric_close(a: Any, b: Any, tol: float) -> bool:
    try:
        return abs(float(a) - float(b)) <= tol
    except (TypeError, ValueError):
        return False


def _values_match(agent_val: Any, gold_val: Any, tolerance: Any) -> bool:
    if agent_val is None and gold_val is None:
        return True
    if agent_val is None or gold_val is None:
        return False
    if isinstance(tolerance, (int, float)) and _numeric_close(agent_val, gold_val, float(tolerance)):
        return True
    return str(agent_val).strip().lower() == str(gold_val).strip().lower()


def _compare_answer(agent: Any, gold: Any, tolerance: Any) -> tuple[int, int]:
    if isinstance(gold, dict):
        if not isinstance(agent, dict):
            total = sum(
                _compare_answer(None, gold_val, tolerance.get(key) if isinstance(tolerance, dict) else tolerance)[1]
                for key, gold_val in gold.items()
            )
            return 0, total
        hits = 0
        total = 0
        for key, gold_val in gold.items():
            tol = tolerance.get(key) if isinstance(tolerance, dict) else tolerance
            sub_hits, sub_total = _compare_answer(agent.get(key), gold_val, tol)
            hits += sub_hits
            total += sub_total
        return hits, total
    if isinstance(gold, list):
        agent_list = agent if isinstance(agent, list) else []
        hits = 0
        total = 0
        for index, gold_val in enumerate(gold):
            agent_val = agent_list[index] if index < len(agent_list) else None
            tol = tolerance[index] if isinstance(tolerance, list) and index < len(tolerance) else tolerance
            sub_hits, sub_total = _compare_answer(agent_val, gold_val, tol)
            hits += sub_hits
            total += sub_total
        return hits, total
    return (1 if _values_match(agent, gold, tolerance) else 0), 1


def score_answer(agent_answer: Any, gold: Any, tolerance: Any) -> float:
    hits, total = _compare_answer(agent_answer, gold, tolerance)
    return hits / total if total else 0.0


def _load_tasks(path: Path) -> list[dict[str, Any]]:
    tasks = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(tasks, list):
        return tasks
    return [tasks]


def _task_answer(task: dict[str, Any]) -> Any:
    if "result" in task:
        return task["result"]
    return task.get("result/answer")


def _task_db_names(task: dict[str, Any]) -> list[str]:
    return list(task.get("db_id") or task.get("domains") or task.get("databases") or [])


def _task_db_paths(task: dict[str, Any], db_dir: Path) -> list[Path]:
    db_names = _task_db_names(task)
    if not db_names:
        raise ValueError(f"Task {task.get('id', '?')} does not define any database identifiers")
    return [_resolve_db_path(db_dir, db_name) for db_name in db_names]


def _resolve_db_path(db_root: Path, db_name: str) -> Path:
    search_roots = [db_root]
    for root in WORKSPACE_DATA_ROOTS:
        if root not in search_roots:
            search_roots.append(root)
    search_roots.extend([
        db_root / "dev_databases",
        db_root / "dev_databases" / "dev_databases",
    ])
    for root in search_roots:
        candidates = [
            root / f"{db_name}.sqlite",
            root / db_name / f"{db_name}.sqlite",
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        matches = list(root.rglob(f"{db_name}.sqlite"))
        if matches:
            return matches[0]
    raise FileNotFoundError(f"Could not locate SQLite database for {db_name!r} under {db_root}")


def _normalize_sql(sql: str) -> str:
    return " ".join(sql.strip().split())


def _run_sql(db_dir: Path, db_name: str, sql: str) -> list[dict[str, Any]] | dict[str, str]:
    db_path = _resolve_db_path(db_dir, db_name)
    try:
        with sqlite3.connect(db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute(sql)
            rows = cursor.fetchall()
            columns = [desc[0] for desc in cursor.description]
            return [dict(zip(columns, row)) for row in rows]
    except Exception as exc:  # noqa: BLE001
        return {"error": str(exc)}


def _numeric_values(rows: list[dict[str, Any]]) -> list[float]:
    values: list[float] = []
    for row in rows:
        for value in row.values():
            try:
                values.append(float(value))
            except (TypeError, ValueError):
                continue
    return sorted(values)


def _compare_numeric_sets(gold_vals: list[float], model_vals: list[float], tol: float = 1e-4) -> dict[str, Any]:
    if len(gold_vals) != len(model_vals):
        return {
            "match": False,
            "reason": f"value count differs: gold={len(gold_vals)}, model={len(model_vals)}",
        }
    mismatches = []
    for index, (gold_val, model_val) in enumerate(zip(gold_vals, model_vals)):
        diff = abs(gold_val - model_val)
        if diff > tol:
            mismatches.append({"index": index, "gold": gold_val, "model": model_val, "diff": diff})
    if mismatches:
        return {"match": False, "reason": f"{len(mismatches)} value(s) differ", "mismatches": mismatches[:5]}
    return {"match": True}


def _compare_sql_outputs(task: dict[str, Any], generated_results: list[QueryResult] | None, db_dir: Path) -> list[dict[str, Any]]:
    generated_by_db = {row.db: row for row in (generated_results or [])}
    comparisons: list[dict[str, Any]] = []
    for gold_entry in task.get("intermediate_sqls", []):
        db_name = gold_entry["db"]
        gold_sql = gold_entry["sql"]
        gold_rows = _run_sql(db_dir, db_name, gold_sql)
        generated_row = generated_by_db.get(db_name)

        entry: dict[str, Any] = {
            "db": db_name,
            "gold_sql": gold_sql,
            "generated_sql": generated_row.sql if generated_row else None,
            "gold_sql_normalized": _normalize_sql(gold_sql),
            "generated_sql_normalized": _normalize_sql(generated_row.sql) if generated_row else None,
            "sql_text_match": _normalize_sql(gold_sql) == _normalize_sql(generated_row.sql) if generated_row else False,
        }

        if isinstance(gold_rows, dict) and "error" in gold_rows:
            entry["gold_error"] = gold_rows["error"]
            comparisons.append(entry)
            continue

        entry["gold_row_count"] = len(gold_rows)
        entry["gold_rows_preview"] = gold_rows[:3]

        if generated_row is None:
            entry["generated_error"] = "no generated SQL for this database"
            comparisons.append(entry)
            continue

        entry["generated_row_count"] = generated_row.row_count
        entry["generated_rows_preview"] = generated_row.rows[:3]
        entry["row_count_match"] = len(gold_rows) == generated_row.row_count

        gold_values = _numeric_values(gold_rows)
        generated_values = _numeric_values(generated_row.rows)
        entry["gold_numeric_values"] = gold_values
        entry["generated_numeric_values"] = generated_values
        entry["value_comparison"] = _compare_numeric_sets(gold_values, generated_values)
        comparisons.append(entry)
    return comparisons


def _build_answer_schema(task: dict[str, Any]) -> type[Any] | None:
    gold = _task_answer(task)
    if isinstance(gold, dict):
        return schema_from_example(gold, name=f"Task{task.get('id', 'X')}Answer")
    return None


@dataclass
class TaskReport:
    task_id: int | str
    difficulty: str
    question: str
    db_id: list[str]
    selected_dbs: list[str]
    final_score: float
    correct: bool
    sql_success: bool
    sql_attempts: int
    agent_answer: Any
    gold_answer: Any
    sql_plan: list[dict[str, Any]]
    sql_comparison: list[dict[str, Any]]
    token_usage: dict[str, Any]
    elapsed_s: float
    error: str | None = None


def run_task(task: dict[str, Any], db_dir: Path) -> TaskReport:
    _task_usage.clear()
    t0 = time.time()

    task_id = task.get("id", "?")
    question = task["question"]
    db_names = _task_db_names(task)
    db_paths = _task_db_paths(task, db_dir)
    gold_answer = _task_answer(task)
    tolerance = task.get("tolerance", 0.01)
    answer_schema = _build_answer_schema(task)

    summaries = [load_or_extract_metadata(str(path)) for path in db_paths]
    selected = _selector_mod.database_selector(question, summaries, evidence=task.get("evidence", ""))
    selected_dbs = selected.databases
    selected_paths = [str(path) for path in db_paths if Path(path).stem in selected_dbs]
    selected_summaries = [summary for summary in summaries if summary.db_name in selected_dbs]
    if not selected_paths:
        selected_paths = [str(path) for path in db_paths]
        selected_summaries = summaries
        selected_dbs = [Path(path).stem for path in db_paths]

    feedback: str | None = None
    plan: SQLPlan | None = None
    results: list[QueryResult] | None = None
    sql_success = False
    sql_attempts = 0
    error: str | None = None

    for attempt in range(3):
        sql_attempts = attempt + 1
        plan = _sql_mod.sql_generator(
            question,
            selected_summaries,
            selected_paths,
            evidence=task.get("evidence", ""),
            feedback=feedback,
        )
        results, exec_error = execute_queries(plan, selected_paths)
        if exec_error:
            feedback = f"Execution error: {exec_error}"
            error = exec_error
            continue
        if is_empty_or_null(results):
            feedback = (
                "Queries executed but one or more returned empty or all-null rows. "
                "This is often a join-key mismatch or an overly strict filter."
            )
            error = feedback
            continue
        sql_success = True
        break

    answer_obj = _result_mod.result_generator(question, plan, results, answer_schema=answer_schema)
    dumped = answer_obj.model_dump()
    agent_answer = dumped.get("answer", dumped)
    final_score = score_answer(agent_answer, gold_answer, tolerance)
    correct = final_score >= 1.0

    usage_summary = {
        "calls": len(_task_usage),
        "input_tokens": sum(item["input"] for item in _task_usage),
        "output_tokens": sum(item["output"] for item in _task_usage),
        "by_call": list(_task_usage),
    }
    usage_summary["estimated_cost_usd"] = sum(
        _cost_usd(item["model"], item["input"], item["output"]) for item in _task_usage
    )

    sql_plan = [q.model_dump() for q in (plan.queries if plan else [])]
    sql_comparison = _compare_sql_outputs(task, results, db_dir)

    return TaskReport(
        task_id=task_id,
        difficulty=task.get("difficulty", ""),
        question=question,
        db_id=list(db_names),
        selected_dbs=selected_dbs,
        final_score=final_score,
        correct=correct,
        sql_success=sql_success,
        sql_attempts=sql_attempts,
        agent_answer=agent_answer,
        gold_answer=gold_answer,
        sql_plan=sql_plan,
        sql_comparison=sql_comparison,
        token_usage=usage_summary,
        elapsed_s=time.time() - t0,
        error=error,
    )


def _fmt(v: float) -> str:
    return "  N/A" if v != v else f"{v:.2f}"


def print_table(reports: list[TaskReport]) -> None:
    header = (
        f"{'Task':>4}  {'DBs':>3}  {'Acc.':>5}  {'SQL OK':>6}  {'Tokens':>8}  {'Time':>7}"
    )
    sep = "─" * len(header)
    print(sep)
    print(header)
    print(sep)
    for report in reports:
        print(
            f"{str(report.task_id):>4}  "
            f"{len(report.db_id):>3}  "
            f"{'✓' if report.correct else '✗':>5}  "
            f"{'✓' if report.sql_success else '✗':>6}  "
            f"{report.token_usage['input_tokens'] + report.token_usage['output_tokens']:>8}  "
            f"{report.elapsed_s:>6.1f}s"
        )
    print(sep)
    if reports:
        total = len(reports)
        correct = sum(1 for report in reports if report.correct)
        sql_ok = sum(1 for report in reports if report.sql_success)
        total_tokens = sum(
            report.token_usage["input_tokens"] + report.token_usage["output_tokens"] for report in reports
        )
        total_cost = sum(report.token_usage["estimated_cost_usd"] for report in reports)
        total_time = sum(report.elapsed_s for report in reports)
        print(
            f"{'AVG':>4}  {'—':>3}  "
            f"{_fmt(correct / total):>5}  "
            f"{_fmt(sql_ok / total):>6}  "
            f"{total_tokens:>8}  "
            f"{total_time:>6.1f}s"
        )
        print(f"Correct tasks: {correct}/{total} | Estimated cost: ${total_cost:.4f}")
        print(sep)


def _select_tasks(tasks: list[dict[str, Any]], task_ids: list[int] | None) -> list[dict[str, Any]]:
    selected = tasks
    if task_ids:
        wanted = set(task_ids)
        selected = [task for task in selected if task.get("id") in wanted]
    return selected


def _normalize_data_dir(data_dir: Path) -> Path:
    if data_dir.is_file():
        return data_dir.parent
    return data_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the cs498-dku-agent benchmark on dev/task.json.")
    parser.add_argument("benchmark_json", nargs="?", default=str(DEFAULT_TASK_JSON), help="Task JSON file")
    parser.add_argument(
        "--db-dir",
        default=str(DEFAULT_DB_DIR),
        help="Directory containing SQLite databases or a parent data directory to search",
    )
    parser.add_argument(
        "--data-dir",
        default=None,
        help="Alias for --db-dir; use this to point at a parent data folder if you prefer",
    )
    parser.add_argument("--agent-dir", default=str(AGENT_REPO), help="Path to the cs498-dku-agent checkout")
    parser.add_argument("--task-ids", help="Comma-separated task IDs to run")
    parser.add_argument("--limit", type=int, help="Maximum number of tasks to run after filtering")
    parser.add_argument("-o", "--output", help="Optional output file name inside runs/")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s [%(name)s] %(message)s",
    )

    benchmark_path = Path(args.benchmark_json)
    db_dir = Path(args.data_dir) if args.data_dir else Path(args.db_dir)
    db_dir = _normalize_data_dir(db_dir)
    agent_dir = Path(args.agent_dir)
    if str(agent_dir) not in sys.path:
        sys.path.insert(0, str(agent_dir))

    if not benchmark_path.exists():
        raise SystemExit(f"Benchmark file not found: {benchmark_path}")
    if not db_dir.exists():
        raise SystemExit(f"Database directory not found: {db_dir}")

    task_ids = None
    if args.task_ids:
        task_ids = [int(part.strip()) for part in args.task_ids.split(",") if part.strip()]

    tasks = _load_tasks(benchmark_path)
    tasks = _select_tasks(tasks, task_ids)
    if not tasks:
        raise SystemExit("No tasks matched the requested filters.")

    skipped_tasks: list[dict[str, Any]] = []
    runnable_tasks: list[dict[str, Any]] = []
    for task in tasks:
        try:
            _task_db_paths(task, db_dir)
        except Exception as exc:  # noqa: BLE001
            skipped_tasks.append({"task_id": task.get("id", "?"), "error": str(exc)})
            continue
        runnable_tasks.append(task)
    if args.limit is not None:
        runnable_tasks = runnable_tasks[: args.limit]
    tasks = runnable_tasks
    if not tasks:
        raise SystemExit("No runnable tasks remain after filtering unavailable databases.")
    if skipped_tasks:
        print(f"Skipping {len(skipped_tasks)} task(s) with missing databases before execution.", flush=True)

    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit("ANTHROPIC_API_KEY is not set.")

    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_name = args.output or f"benchmark_{benchmark_path.stem}_{timestamp}.json"
    output_path = RUNS_DIR / output_name

    print(f"Running {len(tasks)} task(s) from {benchmark_path.name}...\n", flush=True)

    reports: list[TaskReport] = []
    for index, task in enumerate(tasks, 1):
        task_id = task.get("id", "?")
        difficulty = task.get("difficulty", "—")
        preview = task["question"][:80]
        print(f"[{index}/{len(tasks)}] Task {task_id} ({difficulty}): {preview}...", flush=True)

        try:
            report = run_task(task, db_dir)
            reports.append(report)
            print(
                f"         {'✓' if report.correct else '✗'} Final={_fmt(report.final_score)} "
                f"SQL={'✓' if report.sql_success else '✗'} Tokens={report.token_usage['input_tokens'] + report.token_usage['output_tokens']} "
                f"{report.elapsed_s:.1f}s",
                flush=True,
            )
            for comparison in report.sql_comparison:
                db_name = comparison["db"]
                if comparison.get("gold_error"):
                    print(f"         {db_name}: gold SQL error: {comparison['gold_error']}", flush=True)
                elif comparison.get("generated_error"):
                    print(f"         {db_name}: generated SQL missing: {comparison['generated_error']}", flush=True)
                else:
                    text_ok = "✓" if comparison.get("sql_text_match") else "✗"
                    row_ok = "✓" if comparison.get("row_count_match") else "✗"
                    val_ok = "✓" if comparison.get("value_comparison", {}).get("match") else "✗"
                    print(
                        f"         {db_name}: text {text_ok} rows {row_ok} values {val_ok}",
                        flush=True,
                    )
        except Exception as exc:  # noqa: BLE001
            print(f"         ERROR: {exc}", flush=True)
            logger.exception("Task %s failed", task_id)
            reports.append(
                TaskReport(
                    task_id=task_id,
                    difficulty=difficulty,
                    question=task["question"],
                    db_id=list(task.get("db_id") or task.get("domains") or task.get("databases") or []),
                    selected_dbs=[],
                    final_score=0.0,
                    correct=False,
                    sql_success=False,
                    sql_attempts=0,
                    agent_answer=None,
                    gold_answer=_task_answer(task),
                    sql_plan=[],
                    sql_comparison=[],
                    token_usage={"calls": 0, "input_tokens": 0, "output_tokens": 0, "by_call": [], "estimated_cost_usd": 0.0},
                    elapsed_s=0.0,
                    error=str(exc),
                )
            )

    print()
    print_table(reports)

    payload = {
        "benchmark_json": str(benchmark_path),
        "db_dir": str(db_dir),
        "data_dir": str(db_dir),
        "agent_dir": str(agent_dir),
        "run_at": datetime.now().isoformat(timespec="seconds"),
        "task_count": len(reports),
        "correct_tasks": sum(1 for report in reports if report.correct),
        "accuracy": sum(1 for report in reports if report.correct) / len(reports) if reports else 0.0,
        "sql_success_tasks": sum(1 for report in reports if report.sql_success),
        "skipped_tasks": skipped_tasks,
        "input_tokens": sum(report.token_usage["input_tokens"] for report in reports),
        "output_tokens": sum(report.token_usage["output_tokens"] for report in reports),
        "estimated_cost_usd": sum(report.token_usage["estimated_cost_usd"] for report in reports),
        "elapsed_s": sum(report.elapsed_s for report in reports),
        "tasks": [asdict(report) for report in reports],
    }

    output_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()