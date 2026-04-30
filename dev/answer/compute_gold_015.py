"""
Gold answer computation for Task 15
OLS linear regression slope predicting total medal count (y)
from GDP per capita (x) for the 2012 Summer Olympics.

Pipeline:
  olympics                    → medal count per country (2012 Summer)
       ↓  (country name mapping + intersection)
  world_development_indicators → GDP per capita per country (2012)
       ↓
  OLS slope β = (n*Σxy - Σx*Σy) / (n*Σx² - (Σx)²)

Usage:
    python compute_gold_015.py \
        --olympics path/to/olympics.sqlite \
        --wdi      path/to/world_development_indicators.sqlite
"""

import argparse
import sqlite3
import math


# ── Country name mapping: olympics region_name → WDI CountryName ──────────────

NAME_MAP = {
    "USA":                          "United States",
    "UK":                           "United Kingdom",
    "Russia":                       "Russian Federation",
    "South Korea":                  "Korea, Rep.",
    "Iran":                         "Iran, Islamic Rep.",
    "Egypt":                        "Egypt, Arab Rep.",
    "Venezuela":                    "Venezuela, RB",
    "Kyrgyzstan":                   "Kyrgyz Republic",
    "Slovakia":                     "Slovak Republic",
    "Bahamas":                      "Bahamas, The",
    "Ivory Coast":                  "Cote d'Ivoire",
}

EXCLUDE = {
    "Soviet Union", "East Germany", "West Germany", "Unified Team",
    "Yugoslavia", "Czechoslovakia", "Serbia and Montenegro",
    "Taiwan", "North Korea", "Individual Olympic Athletes", "Australasia",
}


# ── SQL queries ────────────────────────────────────────────────────────────────

SQL_MEDALS = """
SELECT
    nr.region_name AS country,
    COUNT(*)       AS total_medals
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
WHERE IndicatorCode = 'NY.GDP.PCAP.CD'
  AND Year = 2012
"""


# ── OLS simple linear regression ──────────────────────────────────────────────

def ols_slope(xs, ys):
    """
    β = (n*Σxy - Σx*Σy) / (n*Σx² - (Σx)²)
    intercept = (Σy - β*Σx) / n
    """
    n   = len(xs)
    sx  = sum(xs)
    sy  = sum(ys)
    sxy = sum(x * y for x, y in zip(xs, ys))
    sx2 = sum(x ** 2 for x in xs)
    denom = n * sx2 - sx ** 2
    if denom == 0:
        raise ValueError("Zero variance in x — cannot compute slope")
    slope     = (n * sxy - sx * sy) / denom
    intercept = (sy - slope * sx) / n
    return slope, intercept


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--olympics", required=True, help="Path to olympics.sqlite")
    parser.add_argument("--wdi",      required=True, help="Path to world_development_indicators.sqlite")
    args = parser.parse_args()

    # ── Step 1: medal counts per country (olympics, 2012 Summer) ─────────────
    con = sqlite3.connect(args.olympics)
    rows = con.execute(SQL_MEDALS).fetchall()
    con.close()

    medals_raw = {}
    excluded, mapped = [], []
    for country, count in rows:
        if country in EXCLUDE:
            excluded.append(country)
            continue
        wdi_name = NAME_MAP.get(country, country)
        medals_raw[wdi_name] = count
        if country != wdi_name:
            mapped.append(f"{country} → {wdi_name}")

    print(f"── Olympics ─────────────────────────────────────────────────────────")
    print(f"Countries with medals (2012 Summer) : {len(rows)}")
    print(f"  → excluded (dissolved nations)    : {len(excluded)}")
    print(f"  → name-mapped                     : {len(mapped)}")
    for m in mapped:
        print(f"       {m}")
    print(f"  → remaining after cleanup         : {len(medals_raw)}")

    # ── Step 2: GDP per capita (world_development_indicators, 2012) ───────────
    con = sqlite3.connect(args.wdi)
    rows = con.execute(SQL_GDP).fetchall()
    con.close()

    gdp = {
        row[0]: row[1]
        for row in rows
        if row[0] is not None and row[1] is not None
    }

    print(f"\n── WDI ──────────────────────────────────────────────────────────────")
    print(f"Countries with GDP per capita (2012) : {len(gdp)}")

    # ── Step 3: intersect ─────────────────────────────────────────────────────
    common = sorted(set(medals_raw) & set(gdp))
    print(f"\nOverlapping countries (n)            : {len(common)}")
    print()

    if len(common) < 2:
        raise ValueError(f"Only {len(common)} overlapping countries.")

    xs = [gdp[c]        for c in common]   # GDP per capita (x)
    ys = [medals_raw[c] for c in common]   # total medals (y)

    slope, intercept = ols_slope(xs, ys)

    print("── Top 10 by medal count ─────────────────────────────────────────────")
    top10 = sorted(common, key=lambda c: medals_raw[c], reverse=True)[:10]
    print(f"{'country':<30}  {'medals':>8}  {'gdp_per_cap':>12}")
    for c in top10:
        print(f"{c:<30}  {medals_raw[c]:>8}  {gdp[c]:>12,.0f}")

    print()
    print(f"OLS slope (raw)          : {slope}")
    print(f"OLS slope (round 8)      : {round(slope, 8)}")
    print(f"OLS intercept            : {intercept:.4f}")
    print()
    print("── Gold answer ──────────────────────────────────────────────────────")
    print(f'{{"slope": {round(slope, 8)}}}')


if __name__ == "__main__":
    main()