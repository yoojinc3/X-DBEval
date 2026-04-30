"""
Gold answer computation for Task 003 with temporary date remapping.
Remaps 2013 → 2018 in sales_in_weather.weather, computes result, then rolls back.

Usage:
    python compute_gold_003.py \
        --crime path/to/chicago_crime.sqlite \
        --weather path/to/sales_in_weather.sqlite
"""

import argparse
import sqlite3
import math


# ── SQL queries ────────────────────────────────────────────────────────────────

SQL_PRECIP = """
SELECT yr_month, AVG(station_avg) AS avg_precip
FROM (
    SELECT station_nbr,
           SUBSTR(date, 1, 7) AS yr_month,
           AVG(preciptotal)   AS station_avg
    FROM weather
    WHERE preciptotal IS NOT NULL
    GROUP BY station_nbr, SUBSTR(date, 1, 7)
)
GROUP BY yr_month
"""

SQL_DV_RATE = """
SELECT
    PRINTF('%04d-%02d',
        CAST(SUBSTR(date, INSTR(date,'/')+INSTR(SUBSTR(date,INSTR(date,'/')+1),'/') +1, 4) AS INT),
        CAST(SUBSTR(date, 1, INSTR(date,'/')-1) AS INT)
    ) AS yr_month,
    CAST(SUM(CASE WHEN domestic = 'TRUE' THEN 1 ELSE 0 END) AS REAL) / COUNT(*) AS dv_rate
FROM Crime
GROUP BY yr_month
"""


# ── Statistics (no external deps) ─────────────────────────────────────────────

def percentile(values, p):
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    idx = (p / 100) * (n - 1)
    lo = int(idx)
    hi = lo + 1
    if hi >= n:
        return sorted_vals[-1]
    frac = idx - lo
    return sorted_vals[lo] + frac * (sorted_vals[hi] - sorted_vals[lo])


def mean(values):
    return sum(values) / len(values)


def variance(values):
    m = mean(values)
    return sum((x - m) ** 2 for x in values) / (len(values) - 1)


def welch_t_test(a, b):
    n_a, n_b = len(a), len(b)
    mean_a, mean_b = mean(a), mean(b)
    var_a, var_b = variance(a), variance(b)

    se = math.sqrt(var_a / n_a + var_b / n_b)
    t = (mean_a - mean_b) / se

    df = (var_a / n_a + var_b / n_b) ** 2 / (
        (var_a / n_a) ** 2 / (n_a - 1) +
        (var_b / n_b) ** 2 / (n_b - 1)
    )

    p = 2 * _t_cdf_upper(abs(t), df)
    return t, p


def _t_cdf_upper(t, df):
    x = df / (df + t * t)
    return 0.5 * _betainc(df / 2, 0.5, x)


def _betainc(a, b, x, iterations=200):
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    lbeta = math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)
    front = math.exp(math.log(x) * a + math.log(1 - x) * b - lbeta) / a
    cf = _continued_fraction(a, b, x, iterations)
    return front * cf


def _continued_fraction(a, b, x, iterations):
    tiny = 1e-30
    f = tiny
    C = f
    D = 0.0
    for m in range(iterations):
        for step in (0, 1):
            if m == 0 and step == 0:
                num = 1.0
            elif step == 0:
                num = m * (b - m) * x / ((a + 2*m - 1) * (a + 2*m))
            else:
                num = -(a + m) * (a + b + m) * x / ((a + 2*m) * (a + 2*m + 1))
            D = 1.0 + num * D
            if abs(D) < tiny:
                D = tiny
            D = 1.0 / D
            C = 1.0 + num / C
            if abs(C) < tiny:
                C = tiny
            delta = C * D
            f *= delta
            if abs(delta - 1.0) < 1e-10:
                return f
    return f


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--crime",   required=True, help="Path to chicago_crime.sqlite")
    parser.add_argument("--weather", required=True, help="Path to sales_in_weather.sqlite")
    args = parser.parse_args()

    # ── Step 1: remap 2013 → 2018 inside a transaction, then rollback ─────────
    con_w = sqlite3.connect(args.weather)
    con_w.execute("BEGIN")

    rows_before = con_w.execute(
        "SELECT COUNT(*) FROM weather WHERE date LIKE '2013%'"
    ).fetchone()[0]
    print(f"Rows with 2013 dates (before update) : {rows_before}")

    con_w.execute("""
        UPDATE weather
        SET date = '2018' || SUBSTR(date, 5)
        WHERE date LIKE '2013%'
    """)

    rows_after = con_w.execute(
        "SELECT COUNT(*) FROM weather WHERE date LIKE '2018%'"
    ).fetchone()[0]
    print(f"Rows with 2018 dates (after update)  : {rows_after}")
    print()

    # ── Step 2: run analysis on remapped data ─────────────────────────────────
    precip = {row[0]: row[1] for row in con_w.execute(SQL_PRECIP).fetchall()}

    con_c = sqlite3.connect(args.crime)
    dv = {row[0]: row[1] for row in con_c.execute(SQL_DV_RATE).fetchall()}
    con_c.close()

    common_months = sorted(set(precip.keys()) & set(dv.keys()))
    print(f"Months in sales_in_weather (remapped) : {len(precip)}")
    print(f"Months in chicago_crime               : {len(dv)}")
    print(f"Overlapping months                    : {len(common_months)}")
    if common_months:
        print(f"Overlapping range                     : {common_months[0]} ~ {common_months[-1]}")
    print()

    precip_vals = [precip[m] for m in common_months]
    dv_vals     = {m: dv[m]  for m in common_months}

    p75 = percentile(precip_vals, 75)
    print(f"75th percentile of avg_precip : {p75:.6f}")

    harsh     = [m for m in common_months if precip[m] >  p75]
    non_harsh = [m for m in common_months if precip[m] <= p75]
    print(f"Harsh months                  : {len(harsh)}")
    print(f"Non-harsh months              : {len(non_harsh)}")
    print()

    harsh_dv     = [dv_vals[m] for m in harsh]
    non_harsh_dv = [dv_vals[m] for m in non_harsh]

    t, p = welch_t_test(harsh_dv, non_harsh_dv)

    print(f"t (raw) : {t}")
    print(f"p (raw) : {p}")
    print()
    print("── Gold Answer ───────────────────────────────────")
    print(f"  t : {round(t, 2)}")
    print(f"  p : {round(p, 4)}")

    # ── Step 3: rollback — database is unchanged ───────────────────────────────
    con_w.execute("ROLLBACK")
    verify = con_w.execute(
        "SELECT COUNT(*) FROM weather WHERE date LIKE '2013%'"
    ).fetchone()[0]
    print()
    print(f"Rollback complete. 2013 rows restored : {verify}")
    con_w.close()


if __name__ == "__main__":
    main()