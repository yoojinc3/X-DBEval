"""
Gold answer computation for Task 19
Pearson correlation between NBA player count per birth country (professional_basketball)
and total medal count at the 2000 Summer Olympics (olympics).

Pipeline:
  professional_basketball → NBA player count per birth country (IOC codes)
       ↓  (IOC code → olympics region_name mapping)
  olympics                → medal count per country (2000 Summer)
       ↓
  Pearson r

Usage:
    python compute_gold_019.py \
        --basketball path/to/professional_basketball.sqlite \
        --olympics   path/to/olympics.sqlite
"""

import argparse
import sqlite3
import math


# ── IOC 3-letter code → Olympics region_name mapping ─────────────────────────

IOC_TO_REGION = {
    "USA": "USA",
    "GER": "Germany",
    "FRA": "France",
    "CAN": "Canada",
    "CRO": "Croatia",
    "ESP": "Spain",
    "HOL": "Netherlands",
    "LTU": "Lithuania",
    "ARG": "Argentina",
    "NGR": "Nigeria",
    "ENG": "UK",
    "BRA": "Brazil",
    "AUS": "Australia",
    "SEN": "Senegal",
    "TUR": "Turkey",
    "CHN": "China",
    "UKR": "Ukraine",
    "JAM": "Jamaica",
    "RUS": "Russia",
    "ITA": "Italy",
    "SLO": "Slovenia",
    "PAN": "Panama",
    "MEX": "Mexico",
    "SAF": "South Africa",
    "ROM": "Romania",
    "BEL": "Belgium",
    "BAH": "Bahamas",
    "TRI": "Trinidad and Tobago",
    "POL": "Poland",
    "NZL": "New Zealand",
    "SRB": "Serbia",
    "SUI": "Switzerland",
    "LAT": "Latvia",
    "EGY": "Egypt",
    "CMR": "Cameroon",
    "BOS": "Bosnia and Herzegovina",
    "TAN": "Tanzania",
    "HUN": "Hungary",
    "VEN": "Venezuela",
    "ISL": "Iceland",
    "IRI": "Iran",
    "GRE": "Greece",
    "SWE": "Sweden",
    "ISR": "Israel",
    "IRL": "Ireland",
    "GUY": "Guyana",
    "EST": "Estonia",
    "CUB": "Cuba",
    "MOR": "Morocco",
    "MLI": "Mali",
    "KOR": "South Korea",
    "GAB": "Gabon",
    "FIN": "Finland",
    "NOR": "Norway",
    "LUX": "Luxembourg",
    "JPN": "Japan",
    "DEN": "Denmark",
    "BUL": "Bulgaria",
    "URU": "Uruguay",
    "SLV": "El Salvador",
    "COD": "Democratic Republic of the Congo",
    "CGO": "Republic of Congo",
    "CON": "Republic of Congo",
}

EXCLUDE = {
    "YUG",   # Yugoslavia (dissolved)
    "TCH",   # Czechoslovakia (dissolved)
    "URS",   # Soviet Union (dissolved)
    "ZAI",   # Zaire (renamed to DRC)
    "PCZ",   # Panama Canal Zone
    "TAI",   # Taiwan
    "SCO",   # Scotland
    "ISV",   # U.S. Virgin Islands
    "PUR",   # Puerto Rico
    "VIN",   # Saint Vincent
    "LIB",   # ambiguous
    "DMC",   # Dominica
    "MON",   # Monaco
}


# ── SQL queries ────────────────────────────────────────────────────────────────

SQL_BASKETBALL = """
SELECT p.birthCountry, COUNT(DISTINCT p.playerID) AS player_count
FROM players p
JOIN players_teams pt ON p.playerID = pt.playerID
WHERE pt.lgID = 'NBA'
  AND p.birthCountry IS NOT NULL
GROUP BY p.birthCountry
ORDER BY player_count DESC
"""

SQL_MEDALS = """
SELECT nr.region_name AS country, COUNT(*) AS total_medals
FROM competitor_event ce
JOIN games_competitor gc ON ce.competitor_id = gc.id
JOIN games            g  ON gc.games_id      = g.id
JOIN person_region    pr ON gc.person_id     = pr.person_id
JOIN noc_region       nr ON pr.region_id     = nr.id
WHERE ce.medal_id IN (1, 2, 3)
  AND g.games_year = 2000
  AND g.season     = 'Summer'
GROUP BY nr.region_name
"""


# ── Pearson correlation ────────────────────────────────────────────────────────

def pearson_r(xs, ys):
    n = len(xs)
    if n < 2:
        raise ValueError(f"Need at least 2 data points, got {n}")
    sx, sy   = sum(xs), sum(ys)
    sxy      = sum(x*y for x,y in zip(xs,ys))
    sx2, sy2 = sum(x**2 for x in xs), sum(y**2 for y in ys)
    num = n*sxy - sx*sy
    den = math.sqrt((n*sx2 - sx**2) * (n*sy2 - sy**2))
    if den == 0:
        raise ValueError("Zero variance")
    return num / den


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--basketball", required=True)
    parser.add_argument("--olympics",   required=True)
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
        region = IOC_TO_REGION.get(ioc)
        if region:
            bball[region] = count
        else:
            skipped.append(ioc)

    print(f"── Basketball ───────────────────────────────────────────────────────")
    print(f"Countries total               : {len(rows)}")
    print(f"  → mapped to region name     : {len(bball)}")
    print(f"  → excluded/unmapped         : {skipped}")

    # Step 2: medal counts (Olympics 2000 Summer)
    con = sqlite3.connect(args.olympics)
    rows = con.execute(SQL_MEDALS).fetchall()
    con.close()

    medals = {r[0]: r[1] for r in rows}

    print(f"\n── Olympics (2000 Summer) ───────────────────────────────────────────")
    print(f"Countries with medals         : {len(medals)}")

    # Step 3: intersect and compute Pearson r
    common = sorted(set(bball) & set(medals))
    print(f"\nOverlapping countries (n)     : {len(common)}")
    print()

    if len(common) < 2:
        raise ValueError("Too few overlapping countries.")

    xs = [bball[c]  for c in common]
    ys = [medals[c] for c in common]

    r = pearson_r(xs, ys)

    print(f"{'country':<30}  {'nba_players':>12}  {'medals_2000':>12}")
    top = sorted(common, key=lambda c: bball[c], reverse=True)[:10]
    for c in top:
        print(f"{c:<30}  {bball[c]:>12}  {medals[c]:>12}")

    print(f"\nPearson r (raw)     : {r}")
    print(f"Pearson r (round 4) : {round(r, 4)}")
    print()
    print("── Gold answer ──────────────────────────────────────────────────────")
    print(f'{{"r": {round(r, 4)}}}')


if __name__ == "__main__":
    main()