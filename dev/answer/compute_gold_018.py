"""
Gold answer computation for Task 18
Spearman correlation between NBA player count per birth country (professional_basketball)
and GDP per capita for year 2000 (world_development_indicators).

Pipeline:
  professional_basketball → NBA player count per birth country (IOC codes)
       ↓  (IOC → WDI country name mapping)
  world_development_indicators → GDP per capita per country (2000)
       ↓
  Spearman r

Usage:
    python compute_gold_018.py \
        --basketball path/to/professional_basketball.sqlite \
        --wdi        path/to/world_development_indicators.sqlite
"""

import argparse
import sqlite3
import math


# ── IOC 3-letter code → WDI CountryName mapping ───────────────────────────────

IOC_TO_WDI = {
    "USA": "United States",
    "GER": "Germany",
    "FRA": "France",
    "CAN": "Canada",
    "CRO": "Croatia",
    "ESP": "Spain",
    "HOL": "Netherlands",
    "LTU": "Lithuania",
    "ARG": "Argentina",
    "NGR": "Nigeria",
    "ENG": "United Kingdom",
    "BRA": "Brazil",
    "AUS": "Australia",
    "SEN": "Senegal",
    "TUR": "Turkey",
    "CHN": "China",
    "UKR": "Ukraine",
    "JAM": "Jamaica",
    "GEO": "Georgia",
    "HAI": "Haiti",
    "DOM": "Dominican Republic",
    "SUD": "Sudan",
    "RUS": "Russian Federation",
    "ITA": "Italy",
    "SLO": "Slovenia",
    "PAN": "Panama",
    "MEX": "Mexico",
    "SAF": "South Africa",
    "ROM": "Romania",
    "BEL": "Belgium",
    "BAH": "Bahamas, The",
    "TRI": "Trinidad and Tobago",
    "POL": "Poland",
    "NZL": "New Zealand",
    "SRB": "Serbia",
    "SUI": "Switzerland",
    "LAT": "Latvia",
    "EGY": "Egypt, Arab Rep.",
    "CMR": "Cameroon",
    "BOS": "Bosnia and Herzegovina",
    "TAN": "Tanzania",
    "HUN": "Hungary",
    "VEN": "Venezuela, RB",
    "ISL": "Iceland",
    "IRI": "Iran, Islamic Rep.",
    "GRE": "Greece",
    "SWE": "Sweden",
    "ISR": "Israel",
    "IRL": "Ireland",
    "GUY": "Guyana",
    "EST": "Estonia",
    "CUB": "Cuba",
    "MOR": "Morocco",
    "MLI": "Mali",
    "KOR": "Korea, Rep.",
    "GAB": "Gabon",
    "FIN": "Finland",
    "NOR": "Norway",
    "LUX": "Luxembourg",
    "JPN": "Japan",
    "DEN": "Denmark",
    "BUL": "Bulgaria",
}

# Dissolved / non-sovereign entities — exclude
EXCLUDE = {
    "YUG",  # Yugoslavia
    "TCH",  # Czechoslovakia
    "URS",  # Soviet Union
    "ZAI",  # Zaire
    "PCZ",  # Panama Canal Zone
    "TAI",  # Taiwan
    "SCO",  # Scotland
    "ISV",  # U.S. Virgin Islands
    "PUR",  # Puerto Rico
    "VIN",  # Saint Vincent
    "LIB",  # Lebanon/Liberia (ambiguous)
    "CON",  # Republic of Congo
    "COD",  # DR Congo
    "CGO",  # Republic of Congo
    "DMC",  # Dominica
    "MON",  # Monaco
    "SLV",  # El Salvador (very small n)
}


# ── SQL queries ────────────────────────────────────────────────────────────────

# NBA player count per birth country
SQL_BASKETBALL = """
SELECT p.birthCountry, COUNT(DISTINCT p.playerID) AS player_count
FROM players p
JOIN players_teams pt ON p.playerID = pt.playerID
WHERE pt.lgID = 'NBA'
  AND p.birthCountry IS NOT NULL
GROUP BY p.birthCountry
ORDER BY player_count DESC
"""

# GDP per capita year 2000
SQL_GDP = """
SELECT CountryName, Value AS gdp_per_capita
FROM Indicators
WHERE IndicatorCode = 'NY.GDP.PCAP.CD'
  AND Year = 2000
"""


# ── Spearman correlation ───────────────────────────────────────────────────────

def rank_data(xs):
    sorted_vals = sorted(enumerate(xs), key=lambda iv: iv[1])
    ranks = [0.0] * len(xs)
    i = 0
    while i < len(sorted_vals):
        j = i
        while j < len(sorted_vals)-1 and sorted_vals[j+1][1] == sorted_vals[i][1]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1
        for k in range(i, j+1):
            ranks[sorted_vals[k][0]] = avg_rank
        i = j + 1
    return ranks

def pearson_r(xs, ys):
    n = len(xs)
    sx, sy   = sum(xs), sum(ys)
    sxy      = sum(x*y for x,y in zip(xs,ys))
    sx2, sy2 = sum(x**2 for x in xs), sum(y**2 for y in ys)
    num = n*sxy - sx*sy
    den = math.sqrt((n*sx2 - sx**2) * (n*sy2 - sy**2))
    if den == 0:
        raise ValueError("Zero variance")
    return num / den

def spearman_r(xs, ys):
    return pearson_r(rank_data(xs), rank_data(ys))


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--basketball", required=True)
    parser.add_argument("--wdi",        required=True)
    args = parser.parse_args()

    # Step 1: NBA player count per country
    con = sqlite3.connect(args.basketball)
    rows = con.execute(SQL_BASKETBALL).fetchall()
    con.close()

    bball = {}
    skipped = []
    for ioc, count in rows:
        if ioc in EXCLUDE:
            skipped.append(ioc)
            continue
        wdi_name = IOC_TO_WDI.get(ioc)
        if wdi_name:
            bball[wdi_name] = count
        else:
            skipped.append(ioc)

    print(f"── Basketball ───────────────────────────────────────────────────────")
    print(f"Countries with NBA players    : {len(rows)}")
    print(f"  → mapped to WDI name        : {len(bball)}")
    print(f"  → excluded/unmapped         : {skipped}")

    # Step 2: GDP per capita
    con = sqlite3.connect(args.wdi)
    rows = con.execute(SQL_GDP).fetchall()
    con.close()
    gdp = {r[0]: r[1] for r in rows if r[1] is not None}

    print(f"\n── WDI ──────────────────────────────────────────────────────────────")
    print(f"Countries with GDP (2000)     : {len(gdp)}")

    # Step 3: intersect
    common = sorted(set(bball) & set(gdp))
    print(f"\nOverlapping countries (n)     : {len(common)}")
    print()

    xs = [bball[c] for c in common]
    ys = [gdp[c]   for c in common]

    r = spearman_r(xs, ys)

    print(f"{'country':<30}  {'nba_players':>12}  {'gdp_per_cap':>12}")
    top = sorted(common, key=lambda c: bball[c], reverse=True)[:10]
    for c in top:
        print(f"{c:<30}  {bball[c]:>12}  {gdp[c]:>12,.0f}")

    print(f"\nSpearman r (raw)     : {r}")
    print(f"Spearman r (round 4) : {round(r, 4)}")
    print()
    print("── Gold answer ──────────────────────────────────────────────────────")
    print(f'{{"spearman_r": {round(r, 4)}}}')

if __name__ == "__main__":
    main()