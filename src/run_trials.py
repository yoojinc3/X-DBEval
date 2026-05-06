"""
Aggregate best-of-N trial results for X-DBEval baseline evaluation.

Usage:
    python aggregate_trials.py <trial_dir_1> <trial_dir_2> <trial_dir_3> [--output <path>]

Each trial dir must contain a results.json produced by run_baseline.py.
For each task, the best result across trials is selected (correct if any trial
was correct, otherwise the first trial's result). Writes aggregated results to
stdout as JSON, and prints a summary to stderr.
"""
import argparse
import json
from pathlib import Path


def load_results(trial_dir: str) -> list:
    path = Path(trial_dir) / "results.json"
    if not path.exists():
        raise FileNotFoundError(f"results.json not found in {trial_dir}")
    with open(path) as f:
        return json.load(f)


def aggregate(trial_dirs: list[str]) -> list:
    all_trials = [load_results(d) for d in trial_dirs]

    # Group results by task id, preserving order from first trial
    by_id: dict = {}
    for trial in all_trials:
        for r in trial:
            by_id.setdefault(r["id"], []).append(r)

    best = []
    for task_id, results in sorted(by_id.items()):
        # Prefer any correct result; fall back to first trial
        winner = next((r for r in results if r.get("correct")), results[0])
        best.append(winner)

    return best


def print_summary(best: list) -> None:
    total = len(best)
    correct = sum(1 for r in best if r.get("correct"))

    difficulties: dict = {}
    for r in best:
        d = r.get("difficulty", "unknown")
        difficulties.setdefault(d, {"total": 0, "correct": 0})
        difficulties[d]["total"] += 1
        if r.get("correct"):
            difficulties[d]["correct"] += 1

    db_sel_correct = sum(1 for r in best if r.get("db_selection_correct"))

    print(f"\n{'='*52}", flush=True)
    print(f"  Tasks aggregated       : {total}")
    print(f"  Overall accuracy       : {correct}/{total} ({100*correct/total:.1f}%)")
    print(f"  DB selection accuracy  : {db_sel_correct}/{total} ({100*db_sel_correct/total:.1f}%)")
    print(f"\n  By difficulty:")
    for diff, counts in sorted(difficulties.items()):
        t, c = counts["total"], counts["correct"]
        acc = 100 * c / t if t else 0
        print(f"    {diff:<15} {c}/{t} ({acc:.1f}%)")
    print(f"{'='*52}\n")


def main():
    parser = argparse.ArgumentParser(description="Aggregate best-of-N trial results.")
    parser.add_argument("trial_dirs", nargs="+", help="Paths to trial output directories")
    parser.add_argument("--output", default=None, help="Write aggregated JSON to this file (default: stdout)")
    args = parser.parse_args()

    if len(args.trial_dirs) < 2:
        print("Warning: only one trial directory provided. Aggregation has no effect.", flush=True)

    best = aggregate(args.trial_dirs)

    output_json = json.dumps(best, indent=2)
    if args.output:
        Path(args.output).write_text(output_json)
        print(f"Aggregated results written to {args.output}")
    else:
        print(output_json)

    print_summary(best)


if __name__ == "__main__":
    main()
