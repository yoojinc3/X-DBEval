"""
Gold answer computation for Task 11
Gender-stratified Welch's t-test comparing total claim amounts per patient
between high-burden and low-burden age groups, separately for Male and Female.

Pipeline:
  mental_health_survey → mh_rate per gender × age_group
                         → within each gender, classify age groups as
                           high/low burden (vs median within that gender)
       ↓  (gender × age_group label)
  synthea              → total claims per patient
                         → within each gender, assign age group high/low label
                         → Welch t-test per gender

Usage:
    python compute_gold_003.py \
        --synthea path/to/synthea.sqlite \
        --survey  path/to/mental_health_survey.sqlite
"""

import argparse
import sqlite3
import math


# ── SQL queries ────────────────────────────────────────────────────────────────

SQL_MH_RATE = """
SELECT
    a_gender.AnswerText                                        AS gender,
    CASE
        WHEN CAST(a_age.AnswerText AS INT) BETWEEN 18 AND 34 THEN '18-34'
        WHEN CAST(a_age.AnswerText AS INT) BETWEEN 35 AND 49 THEN '35-49'
        WHEN CAST(a_age.AnswerText AS INT) BETWEEN 50 AND 65 THEN '50-65'
    END                                                        AS age_group,
    CAST(SUM(CASE WHEN a_mh.AnswerText = 'Yes' THEN 1 ELSE 0 END) AS REAL)
        / COUNT(*)                                             AS mh_rate
FROM Answer a_mh
JOIN Answer a_gender ON a_mh.UserID    = a_gender.UserID
                     AND a_mh.SurveyID  = a_gender.SurveyID
JOIN Answer a_age    ON a_mh.UserID    = a_age.UserID
                     AND a_mh.SurveyID  = a_age.SurveyID
WHERE a_mh.QuestionID    = 33
  AND a_gender.QuestionID = 2
  AND a_gender.AnswerText IN ('Male', 'Female')
  AND a_age.QuestionID    = 1
  AND CAST(a_age.AnswerText AS INT) BETWEEN 18 AND 65
GROUP BY gender, age_group
"""

SQL_PATIENT_CLAIMS = """
SELECT
    p.patient,
    p.gender,
    CASE
        WHEN (strftime('%Y', 'now') - strftime('%Y', p.birthdate)) BETWEEN 18 AND 34 THEN '18-34'
        WHEN (strftime('%Y', 'now') - strftime('%Y', p.birthdate)) BETWEEN 35 AND 49 THEN '35-49'
        WHEN (strftime('%Y', 'now') - strftime('%Y', p.birthdate)) BETWEEN 50 AND 65 THEN '50-65'
    END AS age_group,
    SUM(cl.TOTAL) AS total_claims
FROM patients p
JOIN claims cl ON p.patient = cl.PATIENT
WHERE p.deathdate IS NULL
GROUP BY p.patient
"""

GENDER_MAP = {"Male": "M", "Female": "F"}


# ── Statistics ─────────────────────────────────────────────────────────────────

def median(vals):
    s = sorted(vals)
    n = len(s)
    return (s[n//2 - 1] + s[n//2]) / 2 if n % 2 == 0 else s[n//2]


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
    parser.add_argument("--synthea", required=True)
    parser.add_argument("--survey",  required=True)
    args = parser.parse_args()

    # ── Step 1: mh_rate per (gender, age_group) from survey ──────────────────
    con = sqlite3.connect(args.survey)
    rows = con.execute(SQL_MH_RATE).fetchall()
    con.close()

    # survey_rates: clinical_gender → {age_group: mh_rate}
    survey_rates = {}
    for survey_gender, age_group, mh_rate in rows:
        if age_group is None: continue
        cg = GENDER_MAP[survey_gender]
        survey_rates.setdefault(cg, {})[age_group] = mh_rate

    # ── Step 2: classify high/low within each gender (gender-stratified median)
    labels = {}   # (clinical_gender, age_group) → 'high' | 'low'
    print("── Gender-stratified classification ─────────────────────────────────")
    for cg in ("F", "M"):
        rates_for_gender = survey_rates.get(cg, {})
        med = median(list(rates_for_gender.values()))
        print(f"\n  Gender={cg}  median_mh_rate={med:.4f}")
        print(f"  {'age':>6}  {'mh_rate':>8}  {'burden':>8}")
        for age_group, rate in sorted(rates_for_gender.items()):
            label = "high" if rate > med else "low"
            labels[(cg, age_group)] = label
            print(f"  {age_group:>6}  {rate:>8.4f}  {label:>8}")

    # ── Step 3: total claims per patient from synthea ─────────────────────────
    con = sqlite3.connect(args.synthea)
    rows = con.execute(SQL_PATIENT_CLAIMS).fetchall()
    con.close()

    # claims_by_group: clinical_gender → {'high': [...], 'low': [...]}
    claims_by_group = {"F": {"high": [], "low": []},
                       "M": {"high": [], "low": []}}
    skipped = 0
    for patient, gender, age_group, total in rows:
        if age_group is None or total is None or gender not in ("F", "M"):
            skipped += 1
            continue
        label = labels.get((gender, age_group))
        if label is None:
            skipped += 1
            continue
        claims_by_group[gender][label].append(total)

    print(f"\nSkipped patients (out of age range / no match): {skipped}")

    # ── Step 4: Welch t-test per gender ──────────────────────────────────────
    print("\n── Welch t-test results ─────────────────────────────────────────────")
    results = {}
    for cg, label in (("F", "female"), ("M", "male")):
        high = claims_by_group[cg]["high"]
        low  = claims_by_group[cg]["low"]
        print(f"\n  Gender={cg}  high_n={len(high)}  low_n={len(low)}")
        t, p = welch_t(high, low)
        results[label] = {"t": round(t, 4), "p": round(p, 4)}
        print(f"  t={t:.6f}  p={p:.6f}")
        print(f"  t (round 4)={round(t,4)}  p (round 4)={round(p,4)}")

    print()
    print("── Gold answer ──────────────────────────────────────────────────────")
    print(f'{{"female_t": {results["female"]["t"]}, "female_p": {results["female"]["p"]}, '
          f'"male_t": {results["male"]["t"]}, "male_p": {results["male"]["p"]}}}')


if __name__ == "__main__":
    main()