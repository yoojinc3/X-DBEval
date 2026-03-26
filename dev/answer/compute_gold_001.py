"""
Gold answer computation for Task cross_001
Pearson correlation between ward-level food inspection failure rate
and ward-level crime arrest rate.

Usage:
    python compute_gold_001.py \
        --crime path/to/chicago_crime.sqlite \
        --food  path/to/food_inspection_2.sqlite
"""

import argparse
import sqlite3
import math


# ── SQL queries ────────────────────────────────────────────────────────────────

SQL_FAIL_RATE = """
SELECT
    e.ward,
    CAST(SUM(CASE WHEN i.results = 'Fail' THEN 1 ELSE 0 END) AS REAL)
        / COUNT(*) AS fail_rate
FROM establishment AS e
INNER JOIN inspection AS i ON e.license_no = i.license_no
GROUP BY e.ward
"""

SQL_ARREST_RATE = """
SELECT
    ward_no,
    CAST(SUM(CASE WHEN arrest = 'TRUE' THEN 1 ELSE 0 END) AS REAL)
        / COUNT(*) AS arrest_rate
FROM Crime
GROUP BY ward_no
"""


# ── Pearson correlation (no external deps) ────────────────────────────────────

def pearson_r(xs, ys):
    n = len(xs)
    if n < 2:
        raise ValueError(f"Need at least 2 data points, got {n}")

    sum_x  = sum(xs)
    sum_y  = sum(ys)
    sum_xy = sum(x * y for x, y in zip(xs, ys))
    sum_x2 = sum(x ** 2 for x in xs)
    sum_y2 = sum(y ** 2 for y in ys)

    numerator   = n * sum_xy - sum_x * sum_y
    denominator = math.sqrt(
        (n * sum_x2 - sum_x ** 2) * (n * sum_y2 - sum_y ** 2)
    )

    if denominator == 0:
        raise ValueError("Zero variance in one of the variables")

    return numerator / denominator


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--crime", required=True, help="Path to chicago_crime.sqlite")
    parser.add_argument("--food",  required=True, help="Path to food_inspection_2.sqlite")
    args = parser.parse_args()

    # --- food_inspection_2: fail_rate per ward
    con_food = sqlite3.connect(args.food)
    rows_food = con_food.execute(SQL_FAIL_RATE).fetchall()
    con_food.close()
    fail_rate = {int(row[0]): row[1] for row in rows_food if row[0] is not None}

    # --- chicago_crime: arrest_rate per ward
    con_crime = sqlite3.connect(args.crime)
    rows_crime = con_crime.execute(SQL_ARREST_RATE).fetchall()
    con_crime.close()
    arrest_rate = {int(row[0]): row[1] for row in rows_crime if row[0] is not None}

    # --- intersect wards
    common_wards = sorted(set(fail_rate.keys()) & set(arrest_rate.keys()))

    print(f"Wards in food_inspection_2 : {len(fail_rate)}")
    print(f"Wards in chicago_crime     : {len(arrest_rate)}")
    print(f"Overlapping wards (n)      : {len(common_wards)}")
    print()

    xs = [fail_rate[w]   for w in common_wards]
    ys = [arrest_rate[w] for w in common_wards]

    r = pearson_r(xs, ys)

    print(f"Pearson r (raw)     : {r}")
    print(f"Pearson r (round 4) : {round(r, 4)}")
    print()
    print("── Per-ward detail ──────────────────────────────")
    print(f"{'ward':>6}  {'fail_rate':>10}  {'arrest_rate':>12}")
    for w, x, y in zip(common_wards, xs, ys):
        print(f"{w:>6}  {x:>10.4f}  {y:>12.4f}")


if __name__ == "__main__":
    main()