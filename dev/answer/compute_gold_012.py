"""
Gold answer computation for Task 13
Two-proportion z-test between:
  p1 = delinquency rate among active loans (financial)
  p2 = thrombosis rate among hospitalized patients (thrombosis_prediction)

Pipeline:
  financial              → p1: count(status='D') / count(status IN ('C','D'))
  thrombosis_prediction  → p2: count(Thrombosis>=1) / count(Admission='+')
       ↓
  pooled two-proportion z-test

Usage:
    python compute_gold_013.py \
        --financial path/to/financial.sqlite \
        --thro      path/to/thrombosis_prediction.sqlite
"""

import argparse
import sqlite3
import math


# ── SQL queries ────────────────────────────────────────────────────────────────

# Step 1: active loan status counts (financial)
SQL_LOAN = """
SELECT status, COUNT(*) AS cnt
FROM loan
WHERE status IN ('C', 'D')
GROUP BY status
"""

# Step 2: thrombosis counts among hospitalized patients (thrombosis_prediction)
SQL_THRO = """
SELECT e.Thrombosis, COUNT(*) AS cnt
FROM Examination e
JOIN Patient p ON e.ID = p.ID
WHERE p.Admission = '+'
  AND e.Thrombosis IS NOT NULL
GROUP BY e.Thrombosis
"""


# ── Normal distribution (for z-test p-value) ──────────────────────────────────

def _erf(x):
    sign = 1 if x >= 0 else -1
    x = abs(x)
    t = 1.0 / (1.0 + 0.3275911 * x)
    y = 1.0 - (((((1.061405429 * t
                   - 1.453152027) * t)
                  + 1.421413741) * t
                 - 0.284496736) * t
                + 0.254829592) * t * math.exp(-x * x)
    return sign * y


def norm_cdf(x):
    return 0.5 * (1.0 + _erf(x / math.sqrt(2.0)))


def two_prop_z_test(x1, n1, x2, n2):
    p1 = x1 / n1
    p2 = x2 / n2
    p_pool = (x1 + x2) / (n1 + n2)
    se = math.sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n2))
    if se == 0:
        raise ValueError("Zero standard error")
    z = (p1 - p2) / se
    p_value = 2 * (1 - norm_cdf(abs(z)))
    return p1, p2, z, p_value


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--financial", required=True, help="Path to financial.sqlite")
    parser.add_argument("--thro",      required=True, help="Path to thrombosis_prediction.sqlite")
    args = parser.parse_args()

    # ── Step 1: p1 from financial ─────────────────────────────────────────────
    con = sqlite3.connect(args.financial)
    rows = con.execute(SQL_LOAN).fetchall()
    con.close()

    loan_counts = {row[0]: row[1] for row in rows}
    x1 = loan_counts.get('D', 0)
    n1 = sum(loan_counts.values())

    print(f"── Financial ────────────────────────────────────────────────────────")
    print(f"Active loans (C + D)     : {n1}")
    print(f"  → status C (on track)  : {loan_counts.get('C', 0)}")
    print(f"  → status D (delinquent): {x1}")
    print(f"p1 (delinquency rate)    : {x1}/{n1} = {x1/n1:.4f}")

    # ── Step 2: p2 from thrombosis_prediction ─────────────────────────────────
    con = sqlite3.connect(args.thro)
    rows = con.execute(SQL_THRO).fetchall()
    con.close()

    x2 = sum(cnt for thrombosis, cnt in rows if thrombosis >= 1)
    n2 = sum(cnt for _, cnt in rows)

    print(f"\n── Thrombosis Prediction ────────────────────────────────────────────")
    print(f"Hospitalized patients    : {n2}")
    print(f"  → confirmed thrombosis : {x2}")
    print(f"  → no thrombosis        : {n2 - x2}")
    print(f"p2 (thrombosis rate)     : {x2}/{n2} = {x2/n2:.4f}")

    # ── Step 3: two-proportion z-test ─────────────────────────────────────────
    p1, p2, z, p_value = two_prop_z_test(x1, n1, x2, n2)

    print(f"\n── Two-proportion z-test ────────────────────────────────────────────")
    print(f"p_pool = ({x1}+{x2}) / ({n1}+{n2}) = {(x1+x2)/(n1+n2):.4f}")
    print(f"z (raw)           : {z}")
    print(f"z (round 4)       : {round(z, 4)}")
    print(f"p-value (raw)     : {p_value}")
    print(f"p-value (round 4) : {round(p_value, 4)}")
    print(f"Significant at 0.05: {p_value < 0.05}")
    print()
    print("── Gold answer ──────────────────────────────────────────────────────")
    print(f'{{"p1": {round(p1, 4)}, "p2": {round(p2, 4)}, "z": {round(z, 4)}}}')


if __name__ == "__main__":
    main()