"""
Gold answer computation for Task 8
Pearson correlation between annual total points of British F1 constructors
and average home team goals per match in the English Premier League (2008-2015).

Pipeline:
  formula_1            → total British constructor points per year (2008-2015)
       ↓  (year join)
  european_football_2  → avg home goals per match in England per season (2008-2015)
       ↓
  Pearson r + two-tailed p-value

Usage:
    python compute_gold_008.py \
        --f1       path/to/formula_1.sqlite \
        --football path/to/european_football_2.sqlite
"""

import argparse
import sqlite3
import math


# ── SQL queries ────────────────────────────────────────────────────────────────

SQL_F1 = """
SELECT
    r.year,
    SUM(cs.points) AS brit_pts
FROM constructorStandings cs
JOIN constructors c ON cs.constructorId = c.constructorId
JOIN races r        ON cs.raceId        = r.raceId
WHERE c.nationality = 'British'
  AND r.year BETWEEN 2008 AND 2015
GROUP BY r.year
ORDER BY r.year
"""

SQL_FOOTBALL = """
SELECT
    CAST(SUBSTR(m.season, 1, 4) AS INTEGER) AS season_year,
    AVG(m.home_team_goal)                   AS avg_home_goals
FROM Match m
JOIN League  l  ON m.league_id   = l.id
JOIN Country ct ON l.country_id  = ct.id
WHERE ct.name = 'England'
  AND CAST(SUBSTR(m.season, 1, 4) AS INTEGER) BETWEEN 2008 AND 2015
GROUP BY m.season
ORDER BY m.season
"""


# ── Statistics ─────────────────────────────────────────────────────────────────

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


def pearson_p(r, n):
    """Two-tailed p-value for Pearson r with n observations."""
    if n <= 2:
        raise ValueError("Need n > 2 for p-value")
    if abs(r) == 1.0:
        return 0.0
    t = r * math.sqrt((n - 2) / (1 - r ** 2))
    return _t_two_tailed_p(abs(t), df=n - 2)


def _t_two_tailed_p(t_abs, df):
    x = df / (df + t_abs ** 2)
    return _ibeta(x, df / 2.0, 0.5)


def _ibeta(x, a, b, max_iter=200, tol=1e-10):
    if x <= 0: return 0.0
    if x >= 1: return 1.0
    lbeta = math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)
    front = math.exp(math.log(x) * a + math.log(1 - x) * b - lbeta) / a
    if x < (a + 1) / (a + b + 2):
        return front * _cf(a, b, x, max_iter, tol)
    return 1.0 - (
        math.exp(math.log(1 - x) * b + math.log(x) * a - lbeta) / b
        * _cf(b, a, 1 - x, max_iter, tol)
    )


def _cf(a, b, x, max_iter=200, tol=1e-10):
    qab, qap, qam = a + b, a + 1, a - 1
    c = 1.0
    d = 1.0 - qab * x / qap
    d = 1 / d if abs(d) > 1e-30 else 1e30
    h = d
    for m in range(1, max_iter + 1):
        m2  = 2 * m
        aa  = m * (b - m) * x / ((qam + m2) * (a + m2))
        d   = 1 + aa * d;  d = 1 / d if abs(d) > 1e-30 else 1e30
        c   = 1 + aa / c  if abs(c) > 1e-30 else 1 + aa * 1e30
        h  *= d * c
        aa  = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d   = 1 + aa * d;  d = 1 / d if abs(d) > 1e-30 else 1e30
        c   = 1 + aa / c  if abs(c) > 1e-30 else 1 + aa * 1e30
        delta = d * c;     h *= delta
        if abs(delta - 1) < tol:
            break
    return h


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--f1",       required=True, help="Path to formula_1.sqlite")
    parser.add_argument("--football", required=True, help="Path to european_football_2.sqlite")
    args = parser.parse_args()

    # ── Step 1: British constructor points per year (formula_1) ───────────────
    con = sqlite3.connect(args.f1)
    rows = con.execute(SQL_F1).fetchall()
    con.close()

    f1 = {row[0]: row[1] for row in rows if row[0] and row[1] is not None}

    # ── Step 2: avg home goals per season (european_football_2) ───────────────
    con = sqlite3.connect(args.football)
    rows = con.execute(SQL_FOOTBALL).fetchall()
    con.close()

    football = {row[0]: row[1] for row in rows if row[0] and row[1] is not None}

    # ── Step 3: intersect on year ─────────────────────────────────────────────
    common = sorted(set(f1) & set(football))

    print(f"F1 years (2008-2015)       : {sorted(f1.keys())}")
    print(f"Football years (2008-2015) : {sorted(football.keys())}")
    print(f"Overlapping years (n)      : {len(common)}")
    print()

    if len(common) < 3:
        raise ValueError(
            f"Only {len(common)} overlapping year(s) — cannot compute p-value (need n > 2)."
        )

    xs = [f1[y]       for y in common]   # brit_pts
    ys = [football[y] for y in common]   # avg_home_goals

    r = pearson_r(xs, ys)
    p = pearson_p(r, len(common))

    print("── Per-year detail ──────────────────────────────────────────────────")
    print(f"{'year':>6}  {'brit_pts':>10}  {'avg_home_goals':>15}")
    for year, x, y in zip(common, xs, ys):
        print(f"{year:>6}  {x:>10.1f}  {y:>15.4f}")

    print()
    print(f"Pearson r (raw)     : {r}")
    print(f"Pearson r (round 4) : {round(r, 4)}")
    print(f"p-value   (raw)     : {p}")
    print(f"p-value   (round 4) : {round(p, 4)}")
    print(f"Significant at 0.05 : {p < 0.05}")
    print()
    print("── Gold answer ──────────────────────────────────────────────────────")
    print(f'{{"r": {round(r, 4)}, "p": {round(p, 4)}}}')


if __name__ == "__main__":
    main()