"""
Gold answer computation for Task 17
Pearson correlation between weekday crime count (chicago_crime)
and average tweet sentiment per weekday (social_media).

Usage:
    python compute_gold_017.py \
        --crime  path/to/chicago_crime.sqlite \
        --social path/to/social_media.sqlite
"""

import argparse
import sqlite3
import math

DOW_MAP = {
    '0': 'Sunday', '1': 'Monday', '2': 'Tuesday', '3': 'Wednesday',
    '4': 'Thursday', '5': 'Friday', '6': 'Saturday'
}

SQL_CRIME = (
    "SELECT strftime('%w',"
    " SUBSTR(date, INSTR(date,'/')+INSTR(SUBSTR(date,INSTR(date,'/')+1),'/')+1, 4)"
    " || '-' || PRINTF('%02d', CAST(SUBSTR(date, 1, INSTR(date,'/')-1) AS INT))"
    " || '-' || PRINTF('%02d', CAST(SUBSTR(SUBSTR(date,INSTR(date,'/')+1), 1,"
    " INSTR(SUBSTR(date,INSTR(date,'/')+1),'/')-1) AS INT))"
    ") AS dow, COUNT(*) AS crime_count"
    " FROM Crime GROUP BY dow ORDER BY dow"
)

SQL_SENTIMENT = (
    "SELECT Weekday, AVG(Sentiment) AS avg_sentiment "
    "FROM twitter GROUP BY Weekday ORDER BY Weekday"
)

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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--crime",  required=True)
    parser.add_argument("--social", required=True)
    args = parser.parse_args()

    # Step 1: crime count per weekday
    con = sqlite3.connect(args.crime)
    rows = con.execute(SQL_CRIME).fetchall()
    con.close()
    crime_by_day = {}
    for r in rows:
        if r[0] is not None:
            name = DOW_MAP.get(str(r[0]))
            if name:
                crime_by_day[name] = r[1]

    # Step 2: avg sentiment per weekday
    con = sqlite3.connect(args.social)
    rows = con.execute(SQL_SENTIMENT).fetchall()
    con.close()
    sent_by_day = {r[0]: r[1] for r in rows if r[0] and r[1] is not None}

    # Step 3: intersect and compute Pearson r
    common = sorted(set(crime_by_day) & set(sent_by_day))
    print(f"Overlapping weekdays (n) : {len(common)}")
    print()
    print(f"{'weekday':<12}  {'crime_count':>12}  {'avg_sentiment':>14}")
    xs = [crime_by_day[d] for d in common]
    ys = [sent_by_day[d]  for d in common]
    for d, x, y in zip(common, xs, ys):
        print(f"{d:<12}  {x:>12}  {y:>14.4f}")

    r = pearson_r(xs, ys)
    print(f"\nPearson r (raw)     : {r}")
    print(f"Pearson r (round 4) : {round(r, 4)}")
    print()
    print("── Gold answer ──────────────────────────────────────────────────────")
    print(f'{{"r": {round(r, 4)}}}')

if __name__ == "__main__":
    main()