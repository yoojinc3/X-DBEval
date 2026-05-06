"""
Print a formatted summary for each results/*_trial1/results.json.

Usage:
    python summarize_results.py [dir1 dir2 ...]

If no arguments are given, scans results/*_trial1 relative to the project root
(one level above the src/ directory containing this script).
"""
import json
import sys
from pathlib import Path

PRICING = {
    "claude-sonnet-4-6": {"input": 3.0,  "cache_write": 3.75, "cache_read": 0.30, "output": 15.0},
    "claude-opus-4-7":   {"input": 5.0,  "cache_write": 6.25, "cache_read": 0.50, "output": 25.0},
    "claude-haiku-4-5":  {"input": 1.0,  "cache_write": 1.25, "cache_read": 0.10, "output": 5.0},
    "gpt-5.4":           {"input": 2.50, "cache_write": 0.0,  "cache_read": 0.25, "output": 15.0},
}


def compute_cost(usage: dict, model: str) -> float:
    prices = next((v for k, v in PRICING.items() if k in model), PRICING["claude-sonnet-4-6"])
    return (
        usage.get("input_tokens", 0)                  / 1_000_000 * prices["input"]
        + usage.get("cache_creation_input_tokens", 0) / 1_000_000 * prices["cache_write"]
        + usage.get("cache_read_input_tokens", 0)     / 1_000_000 * prices["cache_read"]
        + usage.get("output_tokens", 0)               / 1_000_000 * prices["output"]
    )


def summarize(results: list, label: str) -> None:
    total = len(results)
    if total == 0:
        print(f"  [{label}] No results.")
        return

    correct = sum(1 for r in results if r.get("correct"))
    db_sel  = sum(1 for r in results if r.get("db_selection_correct"))

    exec_success = sum(
        1 for r in results
        if r.get("execution_results")
        and len(r["execution_results"]) > 0
        and all(
            not isinstance(er.get("result"), dict) or "error" not in er["result"]
            for er in r["execution_results"]
        )
    )

    truncated = sum(
        1 for r in results
        if any(er.get("truncated") for er in (r.get("execution_results") or []))
    )

    isql_tasks   = [r for r in results if r.get("intermediate_sqls_check", {}).get("available")]
    isql_correct = sum(1 for r in isql_tasks if r["intermediate_sqls_check"].get("all_matched"))

    total_usage = {"input_tokens": 0, "output_tokens": 0,
                   "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0}
    for r in results:
        for k in total_usage:
            total_usage[k] += (r.get("token_usage") or {}).get(k, 0)

    # Infer model from first entry that has token_usage and a cost
    model = ""
    for r in results:
        if r.get("token_usage") and r.get("cost_usd"):
            tu = r["token_usage"]
            inferred = compute_cost(tu, "claude-sonnet-4-6")
            # pick whichever key whose price best matches stored cost_usd
            best_key, best_diff = "claude-sonnet-4-6", float("inf")
            for k in PRICING:
                diff = abs(compute_cost(tu, k) - r["cost_usd"])
                if diff < best_diff:
                    best_key, best_diff = k, diff
            model = best_key
            break
    total_cost = compute_cost(total_usage, model)

    difficulties: dict = {}
    for r in results:
        d = r.get("difficulty", "unknown")
        difficulties.setdefault(d, {"total": 0, "correct": 0})
        difficulties[d]["total"] += 1
        if r.get("correct"):
            difficulties[d]["correct"] += 1

    inp = total_usage["input_tokens"]
    out = total_usage["output_tokens"]
    cw  = total_usage["cache_creation_input_tokens"]
    cr  = total_usage["cache_read_input_tokens"]

    print(f"\n  [{label}]")
    print(f"{'='*52}")
    print(f"  Total tasks            : {total}")
    print(f"  Overall accuracy       : {correct}/{total} ({100*correct/total:.1f}%)")
    print(f"  DB selection accuracy  : {db_sel}/{total} ({100*db_sel/total:.1f}%)")
    print(f"  Execution success      : {exec_success}/{total} ({100*exec_success/total:.1f}%)")
    print(f"  Truncated tasks        : {truncated}/{total} ({100*truncated/total:.1f}%)")

    if isql_tasks:
        n, m = len(isql_tasks), isql_correct
        print(f"  Intermediate SQL match : {m}/{n} ({100*m/n:.1f}%)")

    print(f"\n  Token usage:")
    print(f"    Input tokens         : {inp:,}")
    print(f"    Output tokens        : {out:,}")
    if cw:
        print(f"    Cache write tokens   : {cw:,}")
    if cr:
        print(f"    Cache read tokens    : {cr:,}")
    print(f"    Total cost           : ${total_cost:.4f} USD")

    print(f"\n  By difficulty:")
    for diff, counts in sorted(difficulties.items()):
        t, c = counts["total"], counts["correct"]
        print(f"    {diff:<15} {c}/{t} ({100*c/t:.1f}%)")
    print(f"{'='*52}\n")


def main():
    if len(sys.argv) > 1:
        dirs = [Path(d) for d in sys.argv[1:]]
    else:
        root = Path(__file__).parent.parent
        dirs = sorted((root / "results").glob("*_trial1"))

    if not dirs:
        print("No trial directories found.")
        return

    for d in dirs:
        results_path = d / "results.json"
        if not results_path.exists():
            print(f"  [{d.name}] results.json not found, skipping.")
            continue
        with open(results_path) as f:
            results = json.load(f)
        summarize(results, d.name)


if __name__ == "__main__":
    main()
