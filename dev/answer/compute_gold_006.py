"""
Gold answer computation for Task 6
Pearson correlation between country-level average constructor points per race (F1)
and average total goals per match (European football).

Pipeline:
  formula_1            → avg constructor points per race per nationality
       ↓  (adjective → noun mapping)
  european_football_2  → avg total goals per match per country
       ↓  (join on country)
  Pearson r

Usage:
    python compute_gold_006.py \
        --f1       path/to/formula_1.sqlite \
        --football path/to/european_football_2.sqlite
"""

import argparse
import sqlite3
import math


# ── Nationality adjective → country noun mapping ──────────────────────────────
# Agent must discover this independently; hardcoded here for gold computation only.

NATIONALITY_MAP = {
    "British":     "England",
    "French":      "France",
    "German":      "Germany",
    "Italian":     "Italy",
    "Dutch":       "Netherlands",
    "Spanish":     "Spain",
    "Swiss":       "Switzerland",
    "Belgium":     "Belgium",   # stored as noun in F1 DB, not adjective
}


# ── SQL queries ────────────────────────────────────────────────────────────────

# formula_1: avg championship position per nationality (lower = better)
SQL_F1 = """
SELECT
    c.nationality,
    AVG(cs.position) AS avg_position
FROM constructorStandings cs
JOIN constructors c ON cs.constructorId = c.constructorId
GROUP BY c.nationality
"""

# european_football_2: avg total goals per match per country
SQL_FOOTBALL = """
SELECT
    co.name,
    AVG(m.home_team_goal + m.away_team_goal) AS avg_goals
FROM Match m
JOIN Country co ON m.country_id = co.id
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

    # ── Step 1: avg points per race per nationality (formula_1) ───────────────
    con = sqlite3.connect(args.f1)
    rows = con.execute(SQL_F1).fetchall()
    con.close()

    # map nationality adjective → country noun
    f1_by_country = {}
    unmapped = []
    for nationality, avg_pos in rows:
        if nationality is None or avg_pos is None:
            continue
        country = NATIONALITY_MAP.get(nationality)
        if country:
            f1_by_country[country] = avg_pos
        else:
            unmapped.append(nationality)

    print(f"F1 nationalities total          : {len(rows)}")
    print(f"  → mapped to country           : {len(f1_by_country)}")
    if unmapped:
        print(f"  → unmapped (excluded)         : {unmapped}")

    # ── Step 2: avg goals per match per country (european_football_2) ─────────
    con = sqlite3.connect(args.football)
    rows = con.execute(SQL_FOOTBALL).fetchall()
    con.close()

    football_by_country = {
        row[0]: row[1]
        for row in rows
        if row[0] is not None and row[1] is not None
    }

    print(f"Football countries total         : {len(football_by_country)}")

    # ── Step 3: intersect and compute Pearson r ───────────────────────────────
    common = sorted(set(f1_by_country) & set(football_by_country))

    print(f"Overlapping countries (n)        : {len(common)}")
    print()

    if len(common) < 2:
        raise ValueError(
            f"Only {len(common)} overlapping country/ies — cannot compute correlation. "
            "Check NATIONALITY_MAP for missing entries."
        )

    xs = [f1_by_country[c]       for c in common]  # avg F1 position
    ys = [football_by_country[c] for c in common]  # avg goals per match

    r = pearson_r(xs, ys)

    print("── Per-country detail ───────────────────────────────────────────────")
    print(f"{'country':<15}  {'avg_f1_pts':>12}  {'avg_goals':>10}")
    for c, x, y in zip(common, xs, ys):
        print(f"{c:<15}  {x:>12.4f}  {y:>10.4f}")

    print()
    print(f"Pearson r (raw)     : {r}")
    print(f"Pearson r (round 4) : {round(r, 4)}")
    print()
    print("── Gold answer ──────────────────────────────────────────────────────")
    print(f'{{"r": {round(r, 4)}}}')


if __name__ == "__main__":
    main()