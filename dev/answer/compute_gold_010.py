"""
Gold answer computation for Task 10
Spearman correlation between country-level F1 constructor win rate
and average total goals per match in domestic football league.

Pipeline:
  formula_1            → win rate per constructor nationality
       ↓  (adjective → noun mapping)
  european_football_2  → avg total goals per match per country
       ↓
  Spearman r + two-tailed p-value

Usage:
    python compute_gold_010.py \
        --f1       path/to/formula_1.sqlite \
        --football path/to/european_football_2.sqlite
"""

import argparse
import sqlite3
import math


# ── Nationality adjective → country noun mapping ──────────────────────────────

NATIONALITY_MAP = {
    "British":  "England",
    "French":   "France",
    "German":   "Germany",
    "Italian":  "Italy",
    "Dutch":    "Netherlands",
    "Spanish":  "Spain",
    "Swiss":    "Switzerland",
    "Belgium":  "Belgium",   # stored as noun in F1 DB
}


# ── SQL queries ────────────────────────────────────────────────────────────────

# formula_1: win rate per nationality
# win rate = total wins / total race appearances
SQL_F1 = """
SELECT
    c.nationality,
    CAST(SUM(cs.wins) AS REAL) / COUNT(*) AS win_rate
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


# ── Statistics ─────────────────────────────────────────────────────────────────

def rank_data(xs):
    """Average ranks (1-based, ties get average rank)."""
    sorted_vals = sorted(enumerate(xs), key=lambda iv: iv[1])
    ranks = [0.0] * len(xs)
    i = 0
    while i < len(sorted_vals):
        j = i
        while j < len(sorted_vals) - 1 and sorted_vals[j+1][1] == sorted_vals[i][1]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1
        for k in range(i, j + 1):
            ranks[sorted_vals[k][0]] = avg_rank
        i = j + 1
    return ranks


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


def spearman_r(xs, ys):
    return pearson_r(rank_data(xs), rank_data(ys))


def spearman_p(r, n):
    """Two-tailed p-value for Spearman r."""
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

    # ── Step 1: win rate per nationality (formula_1) ──────────────────────────
    con = sqlite3.connect(args.f1)
    rows = con.execute(SQL_F1).fetchall()
    con.close()

    f1_by_country = {}
    unmapped = []
    for nationality, win_rate in rows:
        if nationality is None or win_rate is None:
            continue
        country = NATIONALITY_MAP.get(nationality)
        if country:
            f1_by_country[country] = win_rate
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

    # ── Step 3: intersect and compute Spearman r ──────────────────────────────
    common = sorted(set(f1_by_country) & set(football_by_country))
    print(f"Overlapping countries (n)        : {len(common)}")
    print()


    # filter to countries with non-zero win_rate only
    common_nonzero = [c for c in common if f1_by_country[c] > 0]
    excluded       = [c for c in common if f1_by_country[c] == 0]

    print(f"Countries with win_rate > 0      : {common_nonzero}")
    print(f"Excluded (win_rate = 0)          : {excluded}")
    print()

    if len(common_nonzero) < 2:
        raise ValueError(f"Only {len(common_nonzero)} non-zero countries.")

    xs = [f1_by_country[c]       for c in common_nonzero]
    ys = [football_by_country[c] for c in common_nonzero]

    r = spearman_r(xs, ys)

    print("── Per-country detail (win_rate > 0 only) ───────────────────────────")
    print(f"{'country':<15}  {'win_rate':>10}  {'avg_goals':>10}")
    for c, x, y in zip(common_nonzero, xs, ys):
        print(f"{c:<15}  {x:>10.4f}  {y:>10.4f}")

    print()
    print(f"Spearman r (raw)     : {r}")
    print(f"Spearman r (round 4) : {round(r, 4)}")
    print()
    print("── Gold answer ──────────────────────────────────────────────────────")
    print('{"spearman_r": ' + str(round(r, 4)) + '}')


if __name__ == "__main__":
    main()