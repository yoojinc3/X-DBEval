"""
Gold answer computation for Task 9
Two-proportion z-test between:
  p1 = proportion of nitrogen-containing molecules that are carcinogenic (toxicology)
  p2 = proportion of examined patients with confirmed thrombosis (thrombosis_prediction)

Pipeline:
  toxicology           → p1: carcinogenic rate among nitrogen-containing molecules
  thrombosis_prediction → p2: confirmed thrombosis rate among examined patients
       ↓
  pooled two-proportion z-test

Usage:
    python compute_gold_009.py \
        --tox  path/to/toxicology.sqlite \
        --thro path/to/thrombosis_prediction.sqlite
"""

import argparse
import sqlite3
import math


# ── SQL queries ────────────────────────────────────────────────────────────────

# Step 1: per-molecule nitrogen flag + carcinogenicity label (toxicology)
SQL_TOX = """
SELECT
    m.label,
    MAX(CASE WHEN a.element = 'n' THEN 1 ELSE 0 END) AS has_nitrogen
FROM molecule m
JOIN atom a ON m.molecule_id = a.molecule_id
GROUP BY m.molecule_id, m.label
"""

# Step 2: thrombosis counts by level (thrombosis_prediction)
SQL_THRO = """
SELECT
    Thrombosis,
    COUNT(*) AS cnt
FROM Examination
WHERE Thrombosis IS NOT NULL
GROUP BY Thrombosis
"""


# ── Normal distribution CDF (no external deps) ────────────────────────────────

def _erf(x):
    """Approximation of erf(x) using Horner's method (Abramowitz & Stegun 7.1.26)."""
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
    """Standard normal CDF."""
    return 0.5 * (1.0 + _erf(x / math.sqrt(2.0)))


def two_prop_z_test(x1, n1, x2, n2):
    """
    Two-sided pooled two-proportion z-test.
    Returns (p1, p2, z, p_value).
    """
    p1 = x1 / n1
    p2 = x2 / n2
    p_pool = (x1 + x2) / (n1 + n2)
    se = math.sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n2))
    if se == 0:
        raise ValueError("Zero standard error — check counts")
    z = (p1 - p2) / se
    p_value = 2 * (1 - norm_cdf(abs(z)))
    return p1, p2, z, p_value


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tox",  required=True, help="Path to toxicology.sqlite")
    parser.add_argument("--thro", required=True, help="Path to thrombosis_prediction.sqlite")
    args = parser.parse_args()

    # ── Step 1: p1 from toxicology ────────────────────────────────────────────
    con = sqlite3.connect(args.tox)
    rows_tox = con.execute(SQL_TOX).fetchall()
    con.close()

    # filter to nitrogen-containing molecules only
    nitrogen_mols = [(label, has_n) for label, has_n in rows_tox if has_n == 1]
    x1 = sum(1 for label, _ in nitrogen_mols if label == '+')
    n1 = len(nitrogen_mols)

    print(f"── Toxicology ───────────────────────────────────────────────────────")
    print(f"Total molecules with nitrogen     : {n1}")
    print(f"  → carcinogenic (label='+')      : {x1}")
    print(f"  → non-carcinogenic (label='-')  : {n1 - x1}")
    print(f"p1 (carcinogenic rate)            : {x1}/{n1} = {x1/n1:.4f}")

    # ── Step 2: p2 from thrombosis_prediction ─────────────────────────────────
    con = sqlite3.connect(args.thro)
    rows_thro = con.execute(SQL_THRO).fetchall()
    con.close()

    x2 = sum(cnt for thrombosis, cnt in rows_thro if thrombosis >= 1)
    n2 = sum(cnt for _, cnt in rows_thro)

    print(f"\n── Thrombosis Prediction ────────────────────────────────────────────")
    print(f"Total examined patients           : {n2}")
    print(f"  → confirmed thrombosis (>=1)    : {x2}")
    print(f"  → no thrombosis (=0)            : {n2 - x2}")
    print(f"p2 (thrombosis rate)              : {x2}/{n2} = {x2/n2:.4f}")

    # ── Step 3: two-proportion z-test ─────────────────────────────────────────
    p1, p2, z, p_value = two_prop_z_test(x1, n1, x2, n2)

    print(f"\n── Two-proportion z-test ────────────────────────────────────────────")
    print(f"p_pool = ({x1}+{x2}) / ({n1}+{n2}) = {(x1+x2)/(n1+n2):.4f}")
    print(f"z (raw)       : {z}")
    print(f"z (round 4)   : {round(z, 4)}")
    print(f"p-value (raw) : {p_value}")
    print(f"p-value (round 4) : {round(p_value, 4)}")
    print(f"Significant at 0.05 : {p_value < 0.05}")
    print()
    print("── Gold answer ──────────────────────────────────────────────────────")
    print(f'{{"p1": {round(p1, 4)}, "p2": {round(p2, 4)}, "z": {round(z, 4)}}}')


if __name__ == "__main__":
    main()