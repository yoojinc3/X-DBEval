"""
Gold answer computation for Task 14
Pearson correlation between country-level total constructor championship points (F1)
and average home team goals per match (European football).

Usage:
    python compute_gold_014.py \
        --f1       path/to/formula_1.sqlite \
        --football path/to/european_football_2.sqlite
"""

import argparse
import sqlite3
import math

NATIONALITY_MAP = {
    "British":  "England",
    "French":   "France",
    "German":   "Germany",
    "Italian":  "Italy",
    "Dutch":    "Netherlands",
    "Spanish":  "Spain",
    "Swiss":    "Switzerland",
    "Belgium":  "Belgium",
}

SQL_F1 = """
SELECT c.nationality, SUM(cs.points) AS total_points
FROM constructorStandings cs
JOIN constructors c ON cs.constructorId = c.constructorId
GROUP BY c.nationality
"""

SQL_FOOTBALL = """
SELECT co.name, AVG(m.home_team_goal) AS avg_home_goals
FROM Match m
JOIN Country co ON m.country_id = co.id
GROUP BY co.name
"""

def pearson_r(xs, ys):
    n = len(xs)
    if n < 2:
        raise ValueError(f"Need at least 2 data points, got {n}")
    sx, sy = sum(xs), sum(ys)
    sxy = sum(x*y for x,y in zip(xs,ys))
    sx2 = sum(x**2 for x in xs)
    sy2 = sum(y**2 for y in ys)
    num = n*sxy - sx*sy
    den = math.sqrt((n*sx2 - sx**2) * (n*sy2 - sy**2))
    if den == 0:
        raise ValueError("Zero variance")
    return num / den

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--f1",       required=True)
    parser.add_argument("--football", required=True)
    args = parser.parse_args()

    con = sqlite3.connect(args.f1)
    rows = con.execute(SQL_F1).fetchall()
    con.close()
    f1 = {}
    for nat, pts in rows:
        country = NATIONALITY_MAP.get(nat)
        if country:
            f1[country] = pts

    con = sqlite3.connect(args.football)
    rows = con.execute(SQL_FOOTBALL).fetchall()
    con.close()
    football = {r[0]: r[1] for r in rows if r[0] and r[1] is not None}

    common = sorted(set(f1) & set(football))
    print(f"Overlapping countries (n) : {len(common)}")
    print()
    print(f"{'country':<15}  {'total_pts':>12}  {'avg_home_goals':>15}")
    xs = [f1[c] for c in common]
    ys = [football[c] for c in common]
    for c, x, y in zip(common, xs, ys):
        print(f"{c:<15}  {x:>12,.0f}  {y:>15.4f}")

    r = pearson_r(xs, ys)
    print(f"\nPearson r (raw)     : {r}")
    print(f"Pearson r (round 4) : {round(r, 4)}")
    print(f'\nGold answer: {{"r": {round(r, 4)}}}')

if __name__ == "__main__":
    main()