"""
Gold answer computation for Task 16
OLS multiple linear regression predicting total medal count (y)
from GDP per capita (x1) and total population (x2)
for the 2012 Summer Olympics.

Pipeline:
  olympics                    → medal count per country (2012 Summer)
       ↓  (country name mapping + intersection)
  world_development_indicators → GDP per capita + population (2012)
       ↓
  OLS: β = (XᵀX)⁻¹ Xᵀy
  X = [1, gdp_per_capita, population]  (design matrix with intercept)

Usage:
    python compute_gold_016.py \
        --olympics path/to/olympics.sqlite \
        --wdi      path/to/world_development_indicators.sqlite
"""

import argparse
import sqlite3


# ── Country name mapping ───────────────────────────────────────────────────────

NAME_MAP = {
    "USA":          "United States",
    "UK":           "United Kingdom",
    "Russia":       "Russian Federation",
    "South Korea":  "Korea, Rep.",
    "Iran":         "Iran, Islamic Rep.",
    "Egypt":        "Egypt, Arab Rep.",
    "Venezuela":    "Venezuela, RB",
    "Kyrgyzstan":   "Kyrgyz Republic",
    "Slovakia":     "Slovak Republic",
    "Bahamas":      "Bahamas, The",
    "Ivory Coast":  "Cote d'Ivoire",
}

EXCLUDE = {
    "Soviet Union", "East Germany", "West Germany", "Unified Team",
    "Yugoslavia", "Czechoslovakia", "Serbia and Montenegro",
    "Taiwan", "North Korea", "Individual Olympic Athletes", "Australasia",
}


# ── SQL queries ────────────────────────────────────────────────────────────────

SQL_MEDALS = """
SELECT nr.region_name AS country, COUNT(*) AS total_medals
FROM competitor_event ce
JOIN games_competitor gc ON ce.competitor_id = gc.id
JOIN games            g  ON gc.games_id      = g.id
JOIN person_region    pr ON gc.person_id     = pr.person_id
JOIN noc_region       nr ON pr.region_id     = nr.id
WHERE ce.medal_id IN (1, 2, 3)
  AND g.games_year = 2012
  AND g.season     = 'Summer'
GROUP BY nr.region_name
"""

SQL_GDP = """
SELECT CountryName, Value AS gdp_per_capita
FROM Indicators
WHERE IndicatorCode = 'NY.GDP.PCAP.CD' AND Year = 2012
"""

SQL_POP = """
SELECT CountryName, Value AS population
FROM Indicators
WHERE IndicatorCode = 'SP.POP.TOTL' AND Year = 2012
"""


# ── 3×3 matrix operations (no external deps) ──────────────────────────────────

def mat_mul(A, B):
    """Multiply two matrices (list of lists)."""
    rows_A, cols_A = len(A), len(A[0])
    rows_B, cols_B = len(B), len(B[0])
    assert cols_A == rows_B
    C = [[0.0] * cols_B for _ in range(rows_A)]
    for i in range(rows_A):
        for j in range(cols_B):
            for k in range(cols_A):
                C[i][j] += A[i][k] * B[k][j]
    return C


def transpose(A):
    return [[A[j][i] for j in range(len(A))] for i in range(len(A[0]))]


def mat_inv_3x3(M):
    """Invert a 3×3 matrix using cofactor expansion."""
    a = [[M[i][j] for j in range(3)] for i in range(3)]

    def det2(r1, c1, r2, c2):
        return a[r1][c1] * a[r2][c2] - a[r1][c2] * a[r2][c1]

    cofactors = [
        [ det2(1,1,2,2), -det2(1,0,2,2),  det2(1,0,2,1)],
        [-det2(0,1,2,2),  det2(0,0,2,2), -det2(0,0,2,1)],
        [ det2(0,1,1,2), -det2(0,0,1,2),  det2(0,0,1,1)],
    ]
    det = sum(a[0][j] * cofactors[0][j] for j in range(3))
    if abs(det) < 1e-30:
        raise ValueError("Matrix is singular — cannot invert")
    return [[cofactors[j][i] / det for j in range(3)] for i in range(3)]


def ols_multiple(X_rows, y):
    """
    X_rows: list of [1, x1, x2] rows
    y:      list of target values
    Returns β = (XᵀX)⁻¹ Xᵀy as [β0, β1, β2]
    """
    X  = X_rows                           # n×3
    Xt = transpose(X)                     # 3×n
    XtX   = mat_mul(Xt, X)               # 3×3
    XtX_i = mat_inv_3x3(XtX)             # 3×3
    Xty   = [[sum(Xt[i][k] * y[k] for k in range(len(y)))] for i in range(3)]  # 3×1
    beta  = mat_mul(XtX_i, Xty)          # 3×1
    return [beta[i][0] for i in range(3)]


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--olympics", required=True)
    parser.add_argument("--wdi",      required=True)
    args = parser.parse_args()

    # ── Step 1: medal counts (olympics) ───────────────────────────────────────
    con = sqlite3.connect(args.olympics)
    medals_raw = {}
    for country, count in con.execute(SQL_MEDALS).fetchall():
        if country in EXCLUDE:
            continue
        medals_raw[NAME_MAP.get(country, country)] = count
    con.close()

    # ── Step 2: GDP per capita (WDI) ──────────────────────────────────────────
    con = sqlite3.connect(args.wdi)
    gdp = {r[0]: r[1] for r in con.execute(SQL_GDP).fetchall() if r[1] is not None}
    pop = {r[0]: r[1] for r in con.execute(SQL_POP).fetchall() if r[1] is not None}
    con.close()

    # ── Step 3: three-way intersection ────────────────────────────────────────
    common = sorted(set(medals_raw) & set(gdp) & set(pop))
    print(f"Overlapping countries (medals ∩ GDP ∩ population) : {len(common)}")
    print()

    if len(common) < 4:
        raise ValueError("Too few countries for multiple regression.")

    X_rows = [[1.0, gdp[c], pop[c]] for c in common]
    y      = [float(medals_raw[c])  for c in common]

    # ── Step 4: OLS ───────────────────────────────────────────────────────────
    beta = ols_multiple(X_rows, y)
    b0, b1, b2 = beta

    print(f"β0 (intercept)       : {b0:.10f}")
    print(f"β1 (gdp_per_capita)  : {b1:.10f}")
    print(f"β2 (population)      : {b2:.10f}")
    print()
    print("── Gold answer ──────────────────────────────────────────────────────")
    print(f'{{"beta_0": {round(b0, 6)}, "beta_1": {round(b1, 6)}, "beta_2": {round(b2, 10)}}}')


if __name__ == "__main__":
    main()