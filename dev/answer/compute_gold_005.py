"""
Gold answer computation for Task 9
Pearson correlation between a school's spending-to-labor-market ratio
and its student adverse outcome rate.

Pipeline:
  student_loan        → adverse_rate per school (abbreviation)
       ↓  (SCHOOL_NAME_MAP)
  college_completion  → exp_award_value per school (full name)
       ↓  (scalar join)
  human_resources     → avg maxsalary for bachelor's-level positions
                        (used as labor market benchmark scalar)

  ratio = exp_award_value / bachelor_avg_maxsalary  (per school)
  result = Pearson r(ratio, adverse_rate)

Usage:
    python compute_gold_009.py \
        --loan    path/to/student_loan.sqlite \
        --college path/to/college_completion.sqlite \
        --hr      path/to/human_resources.sqlite
"""

import argparse
import sqlite3
import math


# ── School name mapping (abbreviation → full chronname) ───────────────────────

SCHOOL_NAME_MAP = {
    "ucb":  "University of California at Berkeley",
    "ucla": "University of California at Los Angeles",
    "ucsd": "University of California at San Diego",
    "uci":  "University of California at Irvine",
    "smc":  "Santa Monica College",
    "occ":  "Orange Coast College",
}


# ── SQL queries ────────────────────────────────────────────────────────────────

# Step 1: adverse_rate per school (student_loan)
SQL_ADVERSE_PER_SCHOOL = """
SELECT
    e.school,
    CAST(
        COUNT(DISTINCT CASE
            WHEN u.name  IS NOT NULL
              OR fb.name IS NOT NULL
              OR np.bool = 'pos'
            THEN e.name
        END) AS REAL
    ) / COUNT(DISTINCT e.name) AS adverse_rate
FROM enrolled AS e
LEFT JOIN unemployed           AS u  ON e.name = u.name
LEFT JOIN filed_for_bankrupcy  AS fb ON e.name = fb.name
LEFT JOIN no_payment_due       AS np ON e.name = np.name
GROUP BY e.school
"""

# Step 2: exp_award_value per school (college_completion)
SQL_EXP_AWARD = """
SELECT chronname, exp_award_value
FROM institution_details
WHERE exp_award_value IS NOT NULL
"""

# Step 3: avg maxsalary for bachelor's-level positions (human_resources)
# maxsalary format: "US$25,000.00" → strip "US$" and "," before casting
SQL_BACHELOR_MAXSALARY = """
SELECT
    AVG(
        CAST(
            REPLACE(REPLACE(maxsalary, 'US$', ''), ',', '')
        AS REAL)
    ) AS avg_maxsalary
FROM position
WHERE educationrequired = '4 year degree'
"""


# ── Pearson correlation ────────────────────────────────────────────────────────

def pearson_r(xs, ys):
    n = len(xs)
    if n < 2:
        raise ValueError(f"Need at least 2 data points, got {n}")
    sum_x  = sum(xs)
    sum_y  = sum(ys)
    sum_xy = sum(x * y for x, y in zip(xs, ys))
    sum_x2 = sum(x ** 2 for x in xs)
    sum_y2 = sum(y ** 2 for y in ys)
    num = n * sum_xy - sum_x * sum_y
    den = math.sqrt((n * sum_x2 - sum_x**2) * (n * sum_y2 - sum_y**2))
    if den == 0:
        raise ValueError("Zero variance in one of the variables")
    return num / den


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--loan",    required=True, help="Path to student_loan.sqlite")
    parser.add_argument("--college", required=True, help="Path to college_completion.sqlite")
    parser.add_argument("--hr",      required=True, help="Path to human_resources.sqlite")
    args = parser.parse_args()

    # ── Step 1: adverse_rate per school (student_loan) ────────────────────────
    con_loan = sqlite3.connect(args.loan)
    rows_loan = con_loan.execute(SQL_ADVERSE_PER_SCHOOL).fetchall()
    con_loan.close()

    school_adverse = {}
    for abbr, rate in rows_loan:
        if abbr is None or rate is None:
            continue
        full_name = SCHOOL_NAME_MAP.get(abbr)
        if full_name:
            school_adverse[full_name] = rate
        else:
            print(f"  [WARN] No mapping for abbreviation: '{abbr}'")

    print(f"Schools from student_loan          : {len(rows_loan)}")
    print(f"  → mapped to full name            : {len(school_adverse)}")

    # ── Step 2: exp_award_value per school (college_completion) ───────────────
    con_college = sqlite3.connect(args.college)
    rows_college = con_college.execute(SQL_EXP_AWARD).fetchall()
    con_college.close()

    exp_award = {row[0]: row[1] for row in rows_college if row[0] and row[1]}

    # ── Step 3: bachelor's avg maxsalary scalar (human_resources) ─────────────
    con_hr = sqlite3.connect(args.hr)
    row_hr = con_hr.execute(SQL_BACHELOR_MAXSALARY).fetchone()
    con_hr.close()

    bachelor_maxsalary = row_hr[0]
    if bachelor_maxsalary is None:
        raise ValueError("No bachelor's-level positions found in human_resources")

    print(f"Bachelor avg maxsalary (HR scalar) : ${bachelor_maxsalary:,.2f}")

    # ── Step 4: join on school name, compute ratio ────────────────────────────
    common_schools = sorted(
        set(school_adverse.keys()) & set(exp_award.keys())
    )

    print(f"Schools with exp_award_value       : {len(exp_award)}")
    print(f"Overlapping schools (n)            : {len(common_schools)}")
    print()

    if len(common_schools) < 2:
        raise ValueError(
            f"Only {len(common_schools)} overlapping school(s) — cannot compute correlation. "
            "Check SCHOOL_NAME_MAP against chronname values."
        )

    ratios        = [exp_award[s] / bachelor_maxsalary for s in common_schools]
    adverse_rates = [school_adverse[s] for s in common_schools]

    r = pearson_r(ratios, adverse_rates)

    print("── Per-school detail ────────────────────────────────────────────────")
    print(f"{'school':<45}  {'exp_award':>10}  {'ratio':>8}  {'adverse':>8}")
    for s, ratio, adv in zip(common_schools, ratios, adverse_rates):
        print(f"{s:<45}  {exp_award[s]:>10,.0f}  {ratio:>8.4f}  {adv:>8.4f}")

    print()
    print(f"Pearson r (raw)     : {r}")
    print(f"Pearson r (round 4) : {round(r, 4)}")
    print()
    print("── Gold answer ──────────────────────────────────────────────────────")
    print(f'{{"r": {round(r, 4)}}}')


if __name__ == "__main__":
    main()