"""
Gold answer computation for Task 20
Welch's t-test comparing all-time Olympic medal counts between
High income and Low income countries.

Pipeline:
  world_development_indicators → IncomeGroup per country (Country table)
       ↓  (country name join)
  olympics                     → all-time medal count per country
       ↓
  Welch t-test: High income vs Low income groups

Usage:
    python compute_gold_020.py \
        --olympics path/to/olympics.sqlite \
        --wdi      path/to/world_development_indicators.sqlite
"""

import argparse
import sqlite3
import math


# ── Country name mapping: olympics region_name → WDI ShortName/TableName ──────
# WDI Country table uses ShortName/LongName, not CountryName from Indicators

NAME_MAP = {
    "USA":          "United States",
    "UK":           "United Kingdom",
    "South Korea":  "Korea, Rep.",
    "Bahamas":      "Bahamas, The",
    "Ivory Coast":  "Cote d'Ivoire",
    "Syria":        "Syrian Arab Republic",
    "Laos":         "Lao PDR",
    "Macedonia":    "North Macedonia",
    "Venezuela":    "Venezuela, RB",
    "Kyrgyzstan":   "Kyrgyz Republic",
    "Slovakia":     "Slovak Republic",
    # Russia, Iran, Egypt etc. use same ShortName in WDI Country table
}

EXCLUDE = {
    "Soviet Union", "East Germany", "West Germany", "Unified Team",
    "Yugoslavia", "Czechoslovakia", "Serbia and Montenegro",
    "Taiwan", "North Korea", "Individual Olympic Athletes", "Australasia",
    "Bohemia", "Mixed team", "Unknown",
}


# ── SQL queries ────────────────────────────────────────────────────────────────

# All-time medal count per country (olympics)
SQL_MEDALS = """
SELECT nr.region_name AS country, COUNT(*) AS total_medals
FROM competitor_event ce
JOIN games_competitor gc ON ce.competitor_id = gc.id
JOIN person_region    pr ON gc.person_id     = pr.person_id
JOIN noc_region       nr ON pr.region_id     = nr.id
WHERE ce.medal_id IN (1, 2, 3)
GROUP BY nr.region_name
"""

# IncomeGroup per country (WDI Country table)
SQL_INCOME = """
SELECT ShortName, IncomeGroup
FROM Country
WHERE IncomeGroup IS NOT NULL
  AND IncomeGroup != ''
  AND IncomeGroup LIKE '%income%'
"""


# ── Welch's t-test ────────────────────────────────────────────────────────────

def welch_t(a, b):
    n1, n2 = len(a), len(b)
    if n1 < 2 or n2 < 2:
        raise ValueError(f"Need at least 2 per group, got {n1} and {n2}")
    m1, m2 = sum(a)/n1, sum(b)/n2
    v1 = sum((x-m1)**2 for x in a) / (n1-1)
    v2 = sum((x-m2)**2 for x in b) / (n2-1)
    se = math.sqrt(v1/n1 + v2/n2)
    if se == 0:
        raise ValueError("Zero standard error")
    t  = (m1 - m2) / se
    df = (v1/n1 + v2/n2)**2 / (
        (v1/n1)**2/(n1-1) + (v2/n2)**2/(n2-1)
    )
    p = _t_two_tailed_p(abs(t), df)
    return t, p

def _t_two_tailed_p(t_abs, df):
    x = df / (df + t_abs**2)
    return _ibeta(x, df/2.0, 0.5)

def _ibeta(x, a, b, max_iter=200, tol=1e-10):
    if x <= 0: return 0.0
    if x >= 1: return 1.0
    lbeta = math.lgamma(a) + math.lgamma(b) - math.lgamma(a+b)
    front = math.exp(math.log(x)*a + math.log(1-x)*b - lbeta) / a
    if x < (a+1)/(a+b+2):
        return front * _cf(a, b, x, max_iter, tol)
    return 1.0 - (
        math.exp(math.log(1-x)*b + math.log(x)*a - lbeta) / b
        * _cf(b, a, 1-x, max_iter, tol)
    )

def _cf(a, b, x, max_iter=200, tol=1e-10):
    qab, qap, qam = a+b, a+1, a-1
    c = 1.0
    d = 1.0 - qab*x/qap
    d = 1/d if abs(d) > 1e-30 else 1e30
    h = d
    for m in range(1, max_iter+1):
        m2  = 2*m
        aa  = m*(b-m)*x / ((qam+m2)*(a+m2))
        d   = 1+aa*d;  d = 1/d if abs(d) > 1e-30 else 1e30
        c   = 1+aa/c  if abs(c) > 1e-30 else 1+aa*1e30
        h  *= d*c
        aa  = -(a+m)*(qab+m)*x / ((a+m2)*(qap+m2))
        d   = 1+aa*d;  d = 1/d if abs(d) > 1e-30 else 1e30
        c   = 1+aa/c  if abs(c) > 1e-30 else 1+aa*1e30
        delta = d*c;   h *= delta
        if abs(delta-1) < tol:
            break
    return h


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--olympics", required=True)
    parser.add_argument("--wdi",      required=True)
    args = parser.parse_args()

    # Step 1: all-time medal counts (olympics)
    con = sqlite3.connect(args.olympics)
    rows = con.execute(SQL_MEDALS).fetchall()
    con.close()

    medals = {}
    for country, count in rows:
        if country in EXCLUDE:
            continue
        wdi_name = NAME_MAP.get(country, country)
        medals[wdi_name] = count

    print(f"── Olympics ─────────────────────────────────────────────────────────")
    print(f"Countries with medals : {len(medals)}")

    # Step 2: IncomeGroup (WDI Country table)
    con = sqlite3.connect(args.wdi)
    rows = con.execute(SQL_INCOME).fetchall()
    con.close()

    income = {r[0]: r[1] for r in rows}

    print(f"\n── WDI IncomeGroup ──────────────────────────────────────────────────")
    from collections import Counter
    group_counts = Counter(income.values())
    for g, n in sorted(group_counts.items()):
        print(f"  {g:<30}: {n} countries")

    # Step 3: intersect and split into High vs Low income
    high, low = [], []
    unmatched = []
    for country, medal_count in medals.items():
        grp = income.get(country)
        if grp is None:
            unmatched.append(country)
            continue
        if grp.startswith("High income"):
            high.append(medal_count)
        elif grp in ("Low income", "Lower middle income"):
            low.append(medal_count)
        # Upper middle income excluded from both groups

    print(f"\nHigh income countries with medals  : {len(high)}")
    print(f"Low/Lower-middle income with medals: {len(low)}")
    print(f"Unmatched (name mismatch)          : {len(unmatched)}")
    if unmatched:
        print(f"  → {unmatched[:10]}")

    # Step 4: Welch t-test
    t, p = welch_t(high, low)

    print(f"\n── Welch t-test (High income vs Low/Lower-middle income) ────────────")
    print(f"High income  — n={len(high)}, mean={sum(high)/len(high):.1f}")
    print(f"Low income   — n={len(low)}, mean={sum(low)/len(low):.1f}")
    print(f"t (raw)      : {t}")
    print(f"t (round 4)  : {round(t, 4)}")
    print(f"p (raw)      : {p}")
    print(f"p (round 4)  : {round(p, 4)}")
    print(f"Significant at 0.05: {p < 0.05}")
    print()
    print("── Gold answer ──────────────────────────────────────────────────────")
    print(f'{{"t": {round(t, 4)}, "p": {round(p, 4)}}}')


if __name__ == "__main__":
    main()