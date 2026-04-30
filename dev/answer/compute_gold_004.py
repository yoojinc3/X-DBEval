# compute_gold_004.py
import argparse, sqlite3, math

SCHOOL_NAME_MAP = {
    "ucb":  "University of California at Berkeley",
    "ucla": "University of California at Los Angeles",
    "ucsd": "University of California at San Diego",
    "uci":  "University of California at Irvine",
    "smc":  "Santa Monica College",
    "occ":  "Orange Coast College",
}

SQL_ADVERSE = """
SELECT e.school,
    CAST(COUNT(DISTINCT CASE
        WHEN u.name IS NOT NULL OR fb.name IS NOT NULL OR np.bool = 'pos'
        THEN e.name END) AS REAL) / COUNT(DISTINCT e.name) AS adverse_rate
FROM enrolled e
LEFT JOIN unemployed          u  ON e.name = u.name
LEFT JOIN filed_for_bankrupcy fb ON e.name = fb.name
LEFT JOIN no_payment_due      np ON e.name = np.name
GROUP BY e.school
"""

SQL_GRAD = """
SELECT chronname, grad_150_value
FROM institution_details
WHERE grad_150_value IS NOT NULL
"""

def rank_data(xs):
    sorted_vals = sorted(enumerate(xs), key=lambda iv: iv[1])
    ranks = [0.0] * len(xs)
    i = 0
    while i < len(sorted_vals):
        j = i
        while j < len(sorted_vals) - 1 and sorted_vals[j+1][1] == sorted_vals[i][1]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1
        for k in range(i, j + 1):
            ranks[sorted_vals[k][0]] = avg_rank
        i = j + 1
    return ranks

def pearson_r(xs, ys):
    n = len(xs)
    sx, sy = sum(xs), sum(ys)
    sxy = sum(x*y for x,y in zip(xs,ys))
    sx2 = sum(x**2 for x in xs)
    sy2 = sum(y**2 for y in ys)
    num = n*sxy - sx*sy
    den = math.sqrt((n*sx2 - sx**2) * (n*sy2 - sy**2))
    if den == 0: raise ValueError("Zero variance")
    return num / den

def spearman_r(xs, ys):
    return pearson_r(rank_data(xs), rank_data(ys))

def spearman_p(r, n):
    if abs(r) == 1.0: return 0.0
    t = r * math.sqrt((n-2) / (1 - r**2))
    # two-tailed p via regularized incomplete beta
    x = (n-2) / ((n-2) + t**2)
    return _ibeta(x, (n-2)/2, 0.5)

def _ibeta(x, a, b):
    if x == 0: return 0.0
    if x == 1: return 1.0
    lbeta = math.lgamma(a) + math.lgamma(b) - math.lgamma(a+b)
    front = math.exp(math.log(x)*a + math.log(1-x)*b - lbeta) / a
    if x < (a+1)/(a+b+2):
        return front * _cf(a, b, x)
    return 1.0 - math.exp(math.log(1-x)*b + math.log(x)*a - lbeta)/b * _cf(b, a, 1-x)

def _cf(a, b, x, iters=200, tol=1e-10):
    qab, qap, qam = a+b, a+1, a-1
    c, d = 1.0, 1.0 - qab*x/qap
    d = 1/max(abs(d), 1e-30) * (1 if d >= 0 else -1)
    h = d
    for m in range(1, iters+1):
        m2 = 2*m
        aa = m*(b-m)*x / ((qam+m2)*(a+m2))
        d = 1 + aa*d; d = 1/max(abs(d),1e-30)*(1 if d>=0 else -1)
        c = 1 + aa/max(abs(c),1e-30)*(1 if c>=0 else -1); h *= d*c
        aa = -(a+m)*(qab+m)*x / ((a+m2)*(qap+m2))
        d = 1+aa*d; d = 1/max(abs(d),1e-30)*(1 if d>=0 else -1)
        c = 1+aa/max(abs(c),1e-30)*(1 if c>=0 else -1)
        delta = d*c; h *= delta
        if abs(delta-1) < tol: break
    return h

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--loan",    required=True)
    parser.add_argument("--college", required=True)
    args = parser.parse_args()

    con = sqlite3.connect(args.loan)
    rows = con.execute(SQL_ADVERSE).fetchall()
    con.close()
    school_adverse = {SCHOOL_NAME_MAP[r[0]]: r[1] for r in rows if r[0] in SCHOOL_NAME_MAP}

    con = sqlite3.connect(args.college)
    rows = con.execute(SQL_GRAD).fetchall()
    con.close()
    grad = {r[0]: float(r[1]) for r in rows if r[0] and r[1] is not None}

    common = sorted(set(school_adverse) & set(grad))
    print(f"Overlapping schools (n): {len(common)}")

    xs = [grad[s]           for s in common]
    ys = [school_adverse[s] for s in common]

    r = spearman_r(xs, ys)
    p = spearman_p(r, len(common))

    print(f"\n{'school':<45}  {'grad_150':>8}  {'adverse':>8}")
    for s, x, y in zip(common, xs, ys):
        print(f"{s:<45}  {x:>8.4f}  {y:>8.4f}")

    print(f"\nSpearman r : {round(r, 4)}")
    print(f"p-value    : {round(p, 4)}")
    print(f'\nGold answer: {{"spearman_r": {round(r, 4)}, "p_value": {round(p, 4)}}}')

if __name__ == "__main__":
    main()