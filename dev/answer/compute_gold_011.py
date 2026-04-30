"""
Gold answer computation for Task 12
Pearson correlation between country-level average F1 circuit latitude
and average away team goals per match in domestic football league.

Pipeline:
  formula_1            → avg circuit latitude per country
       ↓  (direct country name join)
  european_football_2  → avg away team goals per match per country
       ↓
  Pearson r

Note: circuits.country uses 'UK' while Country.name uses 'England'/'Scotland'.
      Only countries with exact string match are included.

Usage:
    python compute_gold_012.py \
        --f1       path/to/formula_1.sqlite \
        --football path/to/european_football_2.sqlite
"""

import argparse
import sqlite3
import math


# ── SQL queries ────────────────────────────────────────────────────────────────

SQL_F1 = """
SELECT
    ci.country,
    AVG(ci.lat) AS avg_lat
FROM circuits ci
WHERE ci.lat IS NOT NULL
GROUP BY ci.country
"""

SQL_FOOTBALL = """
SELECT
    co.name,
    AVG(m.away_team_goal) AS avg_away_goals
FROM Match m
JOIN Country co ON m.country_id = co.id
WHERE m.away_team_goal IS NOT NULL
GROUP BY co.name
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
    parser.add_argument("--f1",       required=True, help="Path to formula_1.sqlite")
    parser.add_argument("--football", required=True, help="Path to european_football_2.sqlite")
    args = parser.parse_args()

    # ── Step 1: avg circuit latitude per country (formula_1) ──────────────────
    con = sqlite3.connect(args.f1)
    rows = con.execute(SQL_F1).fetchall()
    con.close()

    f1_by_country = {
        row[0]: row[1]
        for row in rows
        if row[0] is not None and row[1] is not None
    }

    # ── Step 2: avg away goals per country (european_football_2) ──────────────
    con = sqlite3.connect(args.football)
    rows = con.execute(SQL_FOOTBALL).fetchall()
    con.close()

    football_by_country = {
        row[0]: row[1]
        for row in rows
        if row[0] is not None and row[1] is not None
    }

    # ── Step 3: direct string intersection ────────────────────────────────────
    common = sorted(set(f1_by_country) & set(football_by_country))
    unmatched_f1 = sorted(set(f1_by_country) - set(football_by_country))

    print(f"F1 countries total             : {len(f1_by_country)}")
    print(f"Football countries total       : {len(football_by_country)}")
    print(f"Overlapping (exact match, n)   : {len(common)}")
    print(f"F1-only (no football match)    : {unmatched_f1}")
    print()

    if len(common) < 2:
        raise ValueError(f"Only {len(common)} overlapping countries.")

    xs = [f1_by_country[c]       for c in common]   # avg_lat
    ys = [football_by_country[c] for c in common]   # avg_away_goals

    r = pearson_r(xs, ys)

    print("── Per-country detail ───────────────────────────────────────────────")
    print(f"{'country':<15}  {'avg_lat':>9}  {'avg_away_goals':>15}")
    for c, x, y in zip(common, xs, ys):
        print(f"{c:<15}  {x:>9.4f}  {y:>15.4f}")

    print()
    print(f"Pearson r (raw)     : {r}")
    print(f"Pearson r (round 4) : {round(r, 4)}")
    print()
    print("── Gold answer ──────────────────────────────────────────────────────")
    print(f'{{"r": {round(r, 4)}}}')


if __name__ == "__main__":
    main()