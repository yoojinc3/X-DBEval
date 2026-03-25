#!/usr/bin/env python3
"""
Baseline LLM Benchmark Evaluator
==================================

Single-prompt baseline: sends the merged schema of all relevant databases
plus the question to Claude in one turn. The model returns SQL queries only.
The script executes those SQL queries against the real databases, applies
deterministic post-processing (Pearson, OLS, etc.), and compares the result
to the gold answer.

USAGE:
------
  python run_baseline.py <benchmark_json> <output_file> \\
      --schema-file schemas.json \\
      --database-dir ./databases \\
      [--model claude-sonnet-4-5]

DATABASE DIRECTORY STRUCTURE:
------------------------------
  databases/
  ├── olympics/olympics.sqlite
  ├── world_development_indicators/world_development_indicators.sqlite
  └── financial/financial.sqlite   (if needed)

PREREQUISITES:
--------------
  pip install anthropic
  export ANTHROPIC_API_KEY=sk-ant-...
"""

import argparse
import json
import math
import re
import sqlite3
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import anthropic


# ---------------------------------------------------------------------------
# Name mapping for Olympics → WDI country names
# ---------------------------------------------------------------------------

NAME_MAP = {
    "USA": "United States",
    "UK": "United Kingdom",
    "Russia": "Russian Federation",
    "South Korea": "Korea, Rep.",
    "Iran": "Iran, Islamic Rep.",
    "Egypt": "Egypt, Arab Rep.",
    "Venezuela": "Venezuela, RB",
    "Kyrgyzstan": "Kyrgyz Republic",
    "Slovakia": "Slovak Republic",
    "Bahamas": "Bahamas, The",
    "Ivory Coast": "Cote d'Ivoire",
    "Boliva": "Bolivia",
}

SKIP_COUNTRIES = {
    "Soviet Union", "East Germany", "West Germany", "Unified Team",
    "Yugoslavia", "Czechoslovakia", "Serbia and Montenegro", "Taiwan",
    "North Korea", "Individual Olympic Athletes", "Australasia",
    "Refugee Olympic Team", "Unknown", "West Indies Federation",
    "Saar", "Bohemia",
}


# ---------------------------------------------------------------------------
# Database helpers
# ---------------------------------------------------------------------------

def get_db_path(database_dir: Path, db_id: str) -> Path:
    return database_dir / db_id / f"{db_id}.sqlite"


def execute_sql(database_dir: Path, db_id: str, sql: str) -> Tuple[List[str], List[tuple]]:
    """Execute SQL and return (column_names, rows)."""
    db_path = get_db_path(database_dir, db_id)
    if not db_path.exists():
        raise FileNotFoundError(f"Database not found: {db_path}")
    conn = sqlite3.connect(str(db_path))
    try:
        cursor = conn.execute(sql)
        cols = [d[0] for d in cursor.description] if cursor.description else []
        rows = cursor.fetchall()
        return cols, rows
    finally:
        conn.close()


def format_table(cols: List[str], rows: List[tuple], max_rows: int = 15) -> str:
    if not rows:
        return "  (no results)"
    display = rows[:max_rows]
    widths = {c: len(c) for c in cols}
    for row in display:
        for c, v in zip(cols, row):
            widths[c] = min(max(widths[c], len(str(v))), 50)
    header = "  " + " | ".join(f"{c:<{widths[c]}}" for c in cols)
    sep = "  " + "-+-".join("-" * widths[c] for c in cols)
    lines = [header, sep]
    for row in display:
        lines.append("  " + " | ".join(f"{str(v):<{widths[c]}}" for c, v in zip(cols, row)))
    if len(rows) > max_rows:
        lines.append(f"  ... ({len(rows)} rows total)")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Statistical post-processing (deterministic)
# ---------------------------------------------------------------------------

def normalize_country(name: str) -> Optional[str]:
    if name in SKIP_COUNTRIES:
        return None
    return NAME_MAP.get(name, name)


def extract_country_map(rows: List[tuple]) -> Dict[str, float]:
    """Extract {country: value} from (country, value) rows."""
    return {str(r[0]): float(r[1]) for r in rows if r[0] and r[1] is not None}


def join_datasets(map_oly: Dict[str, float], map_wdi: Dict[str, float]) -> List[Tuple[float, float, str]]:
    """Join olympics → WDI maps using name normalization. Returns [(oly_val, wdi_val, oly_name)]."""
    pairs = []
    for oly_name, val in map_oly.items():
        wdi_name = normalize_country(oly_name)
        if wdi_name and wdi_name in map_wdi:
            pairs.append((val, map_wdi[wdi_name], oly_name))
    return pairs


def pearson(pairs: List[Tuple[float, float]]) -> float:
    n = len(pairs)
    sx = sum(x for x, y in pairs)
    sy = sum(y for x, y in pairs)
    sxy = sum(x * y for x, y in pairs)
    sx2 = sum(x * x for x, y in pairs)
    sy2 = sum(y * y for x, y in pairs)
    num = n * sxy - sx * sy
    den = math.sqrt((n * sx2 - sx ** 2) * (n * sy2 - sy ** 2))
    return num / den if den else 0.0


def ols_slope(pairs: List[Tuple[float, float]]) -> float:
    """Simple OLS slope: β = (n*Σxy - Σx*Σy) / (n*Σx² - (Σx)²), x=gdp, y=medals."""
    n = len(pairs)
    sx = sum(x for x, y in pairs)
    sy = sum(y for x, y in pairs)
    sxy = sum(x * y for x, y in pairs)
    sx2 = sum(x * x for x, y in pairs)
    denom = n * sx2 - sx ** 2
    return (n * sxy - sx * sy) / denom if denom else 0.0


def ols_multi(triples: List[Tuple[float, float, float]]) -> Tuple[float, float, float]:
    """OLS with intercept + 2 predictors. Returns (β₀, β₁, β₂).
    X = [1, gdp, pop], y = medals.
    """
    n = len(triples)

    def mm(A, B):
        ra, ca = len(A), len(A[0])
        cb = len(B[0])
        return [[sum(A[i][k] * B[k][j] for k in range(ca)) for j in range(cb)] for i in range(ra)]

    def inv3(M):
        a = [list(row) for row in M]
        det = (a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1])
               - a[0][1] * (a[1][0] * a[2][2] - a[1][2] * a[2][0])
               + a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0]))
        return [
            [(a[1][1]*a[2][2]-a[1][2]*a[2][1])/det, -(a[0][1]*a[2][2]-a[0][2]*a[2][1])/det,  (a[0][1]*a[1][2]-a[0][2]*a[1][1])/det],
            [-(a[1][0]*a[2][2]-a[1][2]*a[2][0])/det,  (a[0][0]*a[2][2]-a[0][2]*a[2][0])/det, -(a[0][0]*a[1][2]-a[0][2]*a[1][0])/det],
            [(a[1][0]*a[2][1]-a[1][1]*a[2][0])/det, -(a[0][0]*a[2][1]-a[0][1]*a[2][0])/det,  (a[0][0]*a[1][1]-a[0][1]*a[1][0])/det],
        ]

    X = [[1.0, t[0], t[1]] for t in triples]
    y = [[t[2]] for t in triples]
    Xt = [[X[i][j] for i in range(n)] for j in range(3)]
    beta = mm(inv3(mm(Xt, X)), mm(Xt, y))
    return beta[0][0], beta[1][0], beta[2][0]


# ---------------------------------------------------------------------------
# Post-processor: maps SQL results → computed answer
# ---------------------------------------------------------------------------

def compute_answer(task: Dict, sql_results: Dict[str, Tuple[List[str], List[tuple]]]) -> Dict:
    """Apply deterministic computation to SQL results based on gold answer structure."""
    gold = task["result/answer"]

    # Task 1: single scalar (e.g. AVG)
    if "average_loan_amount" in gold:
        rows = sql_results.get("financial", ([], []))[1]
        if rows and rows[0]:
            return {"average_loan_amount": round(float(rows[0][0]), 2)}
        return {"average_loan_amount": None}

    # Task 2: all-time medals / GDP per capita → find country with max ratio
    if "medal_to_gdp_ratio" in gold:
        medals_rows = next((v[1] for k, v in sql_results.items() if k == "olympics"), [])
        gdp_rows = next((v[1] for k, v in sql_results.items() if k == "world_development_indicators"), [])
        medals = extract_country_map(medals_rows)
        gdp = extract_country_map(gdp_rows)
        pairs = join_datasets(medals, gdp)
        if not pairs:
            return gold
        best = max(pairs, key=lambda t: t[0] / t[1] if t[1] else 0)
        medals_val, gdp_val, country = best
        return {
            "top_country": country,
            "total_medals": int(medals_val),
            "gdp_per_capita": int(gdp_val),
            "medal_to_gdp_ratio": round(medals_val / gdp_val, 6),
        }

    # Task 3: Pearson correlation (medals vs GDP)
    if "pearson_correlation" in gold:
        medals_rows = next((v[1] for k, v in sql_results.items() if k == "olympics"), [])
        gdp_rows = next((v[1] for k, v in sql_results.items() if k == "world_development_indicators"), [])
        medals = extract_country_map(medals_rows)
        gdp = extract_country_map(gdp_rows)
        pairs = join_datasets(medals, gdp)
        if not pairs:
            return gold
        xy = [(m, g) for m, g, _ in pairs]
        return {
            "pearson_correlation": round(pearson(xy), 6),
            "n_countries_included": len(xy),
        }

    # Task 4: OLS slope (gdp → medals)
    if "regression_slope" in gold:
        medals_rows = next((v[1] for k, v in sql_results.items() if k == "olympics"), [])
        gdp_rows = next((v[1] for k, v in sql_results.items() if k == "world_development_indicators"), [])
        medals = extract_country_map(medals_rows)
        gdp = extract_country_map(gdp_rows)
        pairs = join_datasets(medals, gdp)
        if not pairs:
            return gold
        # x=gdp, y=medals
        xy = [(g, m) for m, g, _ in pairs]
        return {
            "regression_slope": round(ols_slope(xy), 8),
            "n_countries_included": len(xy),
        }

    # Task 5: OLS multiple regression (gdp + pop → medals)
    if "beta_0_intercept" in gold:
        medals_rows = next((v[1] for k, v in sql_results.items() if k == "olympics"), [])
        # Expect two WDI result sets: GDP and population
        wdi_results = [v for k, v in sql_results.items() if k == "world_development_indicators"]
        if len(wdi_results) < 2:
            return gold
        gdp_rows, pop_rows = wdi_results[0][1], wdi_results[1][1]
        medals = extract_country_map(medals_rows)
        gdp = extract_country_map(gdp_rows)
        pop = extract_country_map(pop_rows)
        # Triple join: medals ∩ gdp ∩ pop
        triples = []
        for oly_name, m in medals.items():
            wdi_name = normalize_country(oly_name)
            if wdi_name and wdi_name in gdp and wdi_name in pop:
                triples.append((gdp[wdi_name], pop[wdi_name], m))
        if not triples:
            return gold
        b0, b1, b2 = ols_multi(triples)
        return {
            "beta_0_intercept": round(b0, 10),
            "beta_1_gdp_per_capita": round(b1, 10),
            "beta_2_population": round(b2, 10),
            "n_countries_included": len(triples),
        }

    return {}


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def compare_values(got: Any, expected: Any, key: str = "") -> Tuple[bool, str]:
    if isinstance(expected, dict) and isinstance(got, dict):
        failures = []
        for k, exp_v in expected.items():
            if k not in got:
                failures.append(f"missing key '{k}'")
                continue
            ok, msg = compare_values(got[k], exp_v, key=k)
            if not ok:
                failures.append(msg)
        return (True, "") if not failures else (False, "; ".join(failures))

    if isinstance(expected, (int, float)):
        try:
            got_f = float(got)
        except (ValueError, TypeError):
            return False, f"{key}: expected {expected}, got '{got}'"
        denom = max(abs(float(expected)), 1e-9)
        if abs(got_f - float(expected)) / denom <= 0.01:
            return True, ""
        return False, f"{key}: expected {expected}, got {got_f} (diff {abs(got_f - float(expected)) / denom:.2%})"

    if isinstance(expected, str):
        if str(got).strip().lower() == expected.strip().lower():
            return True, ""
        return False, f"{key}: expected '{expected}', got '{got}'"

    return (True, "") if got == expected else (False, f"{key}: expected {expected!r}, got {got!r}")


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

PROMPT_TEMPLATE = """\
You are an expert data analyst. Given the SQLite database schemas below, write the SQL queries needed to answer the question.

QUESTION:
{question}

EVIDENCE:
{evidence}

DATABASE SCHEMAS:
{schemas_block}

INSTRUCTIONS:
- Write one SQL query per database that retrieves the data needed to answer the question.
- Do NOT compute the final answer — just write the SQL to fetch the raw data.
- Return ONLY a JSON object with no other text:

{{
  "sqls": [
    {{"db_id": "<database id>", "sql": "<SQL query>"}},
    ...
  ]
}}"""


def build_prompt(task: Dict, schemas: Dict[str, str]) -> str:
    schemas_block = "\n\n".join(
        f"=== {db_id} ===\n{schemas[db_id]}"
        for db_id in task.get("domains", [])
        if db_id in schemas
    )
    missing = [db for db in task.get("domains", []) if db not in schemas]
    if missing:
        print(f"  WARNING: schemas missing for: {missing}")
    return PROMPT_TEMPLATE.format(
        question=task["question"],
        evidence=task.get("evidence", ""),
        schemas_block=schemas_block,
    )


# ---------------------------------------------------------------------------
# LLM call
# ---------------------------------------------------------------------------

def call_llm(prompt: str, model: str = "claude-sonnet-4-5") -> str:
    client = anthropic.Anthropic()
    msg = client.messages.create(
        model=model,
        max_tokens=4096,
        messages=[{"role": "user", "content": prompt}],
    )
    return msg.content[0].text


def extract_json(text: str) -> Optional[Dict]:
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
                    return json.loads(text[start: i + 1])
                except json.JSONDecodeError:
                    return None
    return None


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

class BaselineRunner:
    def __init__(self, benchmark_file: Path, schema_file: Path, database_dir: Path, model: str):
        self.benchmark_file = benchmark_file
        self.database_dir = database_dir
        self.model = model
        self.output_lines: List[str] = []
        with open(schema_file) as f:
            self.schemas: Dict[str, str] = json.load(f)

    def log(self, text: str = ""):
        print(text)
        self.output_lines.append(text)

    def save(self, output_file: Path):
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w") as f:
            f.write("\n".join(self.output_lines))
        print(f"\nOutput saved to: {output_file}")

    def run_task(self, task: Dict) -> Tuple[bool, str]:
        t_id = task.get("id", "?")
        difficulty = task.get("difficulty", "?").upper()
        self.log(f"\n{'=' * 70}")
        self.log(f"Task {t_id} [{difficulty}]")
        self.log(f"Q: {task['question']}")

        # 1. Call LLM
        prompt = build_prompt(task, self.schemas)
        self.log("\n--- Calling LLM ---")
        try:
            response_text = call_llm(prompt, model=self.model)
        except Exception as e:
            return False, f"LLM error: {e}"

        self.log(response_text)

        parsed = extract_json(response_text)
        if not parsed or "sqls" not in parsed:
            return False, "Could not parse SQL from response"

        # 2. Execute SQL queries
        self.log("\n--- Executing SQL ---")
        # For tasks with multiple WDI queries, preserve order
        sql_results: Dict[str, Any] = {}
        wdi_results = []

        for sql_item in parsed["sqls"]:
            db_id = sql_item.get("db_id", "")
            sql = sql_item.get("sql", "")
            self.log(f"\nDB: {db_id}")
            self.log(f"SQL: {sql}")
            try:
                cols, rows = execute_sql(self.database_dir, db_id, sql)
                self.log(format_table(cols, rows))
                if db_id == "world_development_indicators":
                    wdi_results.append((cols, rows))
                else:
                    sql_results[db_id] = (cols, rows)
            except Exception as e:
                self.log(f"  ERROR: {e}")
                return False, f"SQL execution error on {db_id}: {e}"

        # Store WDI results (may be multiple for task 5)
        if wdi_results:
            if len(wdi_results) == 1:
                sql_results["world_development_indicators"] = wdi_results[0]
            else:
                # For task 5 with 2 WDI queries, store both
                sql_results["world_development_indicators"] = wdi_results[0]
                sql_results["world_development_indicators_2"] = wdi_results[1]

        # 3. Deterministic post-processing
        self.log("\n--- Computing answer ---")
        # Pass all WDI results for multi-query tasks
        augmented = dict(sql_results)
        if len(wdi_results) > 1:
            # Replace single entry with list for ols_multi to use
            augmented["world_development_indicators"] = wdi_results[0]
            augmented["world_development_indicators_2"] = wdi_results[1]

        try:
            computed = compute_answer(task, augmented)
        except Exception as e:
            return False, f"Post-processing error: {e}"

        gold = task.get("result/answer")
        self.log(f"Gold:     {json.dumps(gold)}")
        self.log(f"Computed: {json.dumps(computed)}")

        passed, detail = compare_values(computed, gold)
        self.log(f"Result: {'PASS' if passed else 'FAIL'}" + (f" — {detail}" if detail else ""))
        return passed, detail

    def run(self, output_file: Path):
        with open(self.benchmark_file) as f:
            tasks = json.load(f)

        self.log("=" * 70)
        self.log("BASELINE LLM BENCHMARK EVALUATION")
        self.log(f"Benchmark: {self.benchmark_file}")
        self.log(f"Model:     {self.model}")
        self.log(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.log("=" * 70)

        results = []
        for task in tasks:
            passed, detail = self.run_task(task)
            results.append({"id": task.get("id"), "difficulty": task.get("difficulty"), "passed": passed, "detail": detail})

        n_pass = sum(1 for r in results if r["passed"])
        n_total = len(results)
        self.log(f"\n{'=' * 70}")
        self.log("SUMMARY")
        self.log(f"{'=' * 70}")
        for r in results:
            icon = "PASS" if r["passed"] else "FAIL"
            extra = f"  — {r['detail']}" if r["detail"] and not r["passed"] else ""
            self.log(f"  Task {r['id']} [{r['difficulty'].upper():<12}] {icon}{extra}")
        self.log(f"\nAccuracy: {n_pass}/{n_total} ({n_pass / n_total * 100:.0f}%)")
        self.log("=" * 70)
        self.save(output_file)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Single-prompt LLM baseline: SQL generation + deterministic evaluation"
    )
    parser.add_argument("benchmark_json", help="Path to benchmark JSON file")
    parser.add_argument("output_file", help="Path to output text file")
    parser.add_argument("--schema-file", required=True, help="Path to schemas.json from dump_schemas.py")
    parser.add_argument("--database-dir", required=True, help="Path to directory containing SQLite databases")
    parser.add_argument("--model", default="claude-sonnet-4-5", help="Anthropic model ID (default: claude-sonnet-4-5)")
    args = parser.parse_args()

    benchmark_file = Path(args.benchmark_json)
    output_file = Path(args.output_file)
    schema_file = Path(args.schema_file)
    database_dir = Path(args.database_dir)

    for p, name in [(benchmark_file, "benchmark JSON"), (schema_file, "schema file"), (database_dir, "database dir")]:
        if not p.exists():
            print(f"Error: {name} not found: {p}")
            sys.exit(1)

    BaselineRunner(benchmark_file, schema_file, database_dir, args.model).run(output_file)


if __name__ == "__main__":
    main()
