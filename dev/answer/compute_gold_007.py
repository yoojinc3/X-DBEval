"""
Gold answer computation for Task 7
Pearson correlation between county-level Standard ZIP code rate (student_club)
and county-level average SAT Critical Reading score (california_schools).

Pipeline:
  student_club       → Standard ZIP rate per CA county
       ↓  (county name join)
  california_schools → high achiever rate per county
       ↓
  Pearson r

Usage:
    python compute_gold_007.py \
        --club    path/to/student_club.sqlite \
        --schools path/to/california_schools.sqlite
"""

import argparse
import sqlite3
import math


# ── SQL queries ────────────────────────────────────────────────────────────────

# Step 1: Standard ZIP rate per county in CA (student_club)
SQL_ZIP = """
SELECT
    REPLACE(county, ' County', '') AS county,
    CAST(SUM(CASE WHEN type = 'Standard' THEN 1 ELSE 0 END) AS REAL)
        / COUNT(*) AS standard_rate
FROM zip_code
WHERE state = 'California'
GROUP BY county
"""

# Step 2: high achiever rate per county (california_schools)
SQL_SAT = """
SELECT
    s.County,
    CAST(SUM(sat.NumGE1500) AS REAL) / SUM(sat.NumTstTakr) AS top_rate
FROM schools s
JOIN satscores sat ON s.CDSCode = sat.cds
WHERE sat.NumTstTakr > 0
  AND sat.NumGE1500 IS NOT NULL
GROUP BY s.County
"""


# ── Pearson correlation ────────────────────────────────────────────────────────

def pearson_r(xs, ys):
    n = len(xs)
    if n < 2:
        raise ValueError(f"Need at least 2 data points, got {n}")
    sx  = sum(xs)
    sy  = sum(ys)
    sxy = sum(x * y for x, y in zip(xs, ys))
    sx2 = sum(x ** 2 for x in xs)
    sy2 = sum(y ** 2 for y in ys)
    num = n * sxy - sx * sy
    den = math.sqrt((n * sx2 - sx**2) * (n * sy2 - sy**2))
    if den == 0:
        raise ValueError("Zero variance in one of the variables")
    return num / den


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--club",    required=True, help="Path to student_club.sqlite")
    parser.add_argument("--schools", required=True, help="Path to california_schools.sqlite")
    args = parser.parse_args()

    # ── Step 1: Standard ZIP rate per county (student_club) ───────────────────
    con = sqlite3.connect(args.club)
    rows = con.execute(SQL_ZIP).fetchall()
    con.close()

    zip_by_county = {
        row[0]: row[1]
        for row in rows
        if row[0] is not None and row[1] is not None
    }

    # ── Step 2: avg SAT reading per county (california_schools) ───────────────
    con = sqlite3.connect(args.schools)
    rows = con.execute(SQL_SAT).fetchall()
    con.close()

    sat_by_county = {
        row[0]: row[1]
        for row in rows
        if row[0] is not None and row[1] is not None
    }

    # ── Step 3: intersect and compute Pearson r ───────────────────────────────
    common = sorted(set(zip_by_county) & set(sat_by_county))

    print(f"Counties in student_club (CA)    : {len(zip_by_county)}")
    print(f"Counties in california_schools   : {len(sat_by_county)}")
    print(f"Overlapping counties (n)         : {len(common)}")
    print()

    if len(common) < 2:
        raise ValueError(
            f"Only {len(common)} overlapping county/ies — cannot compute correlation."
        )

    xs = [zip_by_county[c] for c in common]   # standard_rate
    ys = [sat_by_county[c] for c in common]   # top_rate

    r = pearson_r(xs, ys)

    print("── Per-county detail ────────────────────────────────────────────────")
    print(f"{'county':<25}  {'std_rate':>9}  {'top_rate':>9}")
    for c, x, y in zip(common, xs, ys):
        print(f"{c:<25}  {x:>9.4f}  {y:>9.2f}")

    print()
    print(f"Pearson r (raw)     : {r}")
    print(f"Pearson r (round 4) : {round(r, 4)}")
    print()
    print("── Gold answer ──────────────────────────────────────────────────────")
    print(f'{{"r": {round(r, 4)}}}')


if __name__ == "__main__":
    main()