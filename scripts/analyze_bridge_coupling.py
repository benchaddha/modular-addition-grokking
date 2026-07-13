"""Pre-registered batch-B coupling analysis (see results/reports/batch_b_preregistration.md)."""
import csv
import itertools
from itertools import product
from math import comb

# (grok99_t0, grok99_sgld, top3_t0, top3_sgld) at the testacc_99 milestone, pre_unembed
NEW = {10: (10000, 14300, 0.4586, 0.5372), 11: (3900, 6100, 0.9601, 0.9905),
       17: (3400, 9600, 0.6237, 0.5162), 18: (3000, 3800, 0.9984, 0.9849),
       19: (2100, 19200, 0.9947, 0.5139)}
OLD = {0: (12900, 8000, 0.6260, 0.9909), 1: (2700, 3400, 0.9936, 0.9903),
       2: (5300, 7200, 0.9902, 0.6892), 5: (2400, 15300, 0.9970, 0.9911),
       6: (5800, 2800, 0.7892, 0.9922), 7: (2100, 5300, 0.9952, 0.9913),
       9: (9500, 3100, 0.9249, 0.9970)}


def spearman_exact(dt, dc):
    n = len(dt)

    def ranks(v):
        order = sorted(range(n), key=lambda i: v[i])
        r = [0] * n
        for k, i in enumerate(order):
            r[i] = k + 1
        return r

    rt, rc = ranks(dt), ranks(dc)
    d2 = sum((a - b) ** 2 for a, b in zip(rt, rc))
    rho = 1 - 6 * d2 / (n * (n * n - 1))
    if n <= 8:  # exact
        cnt = tot = 0
        for perm in itertools.permutations(range(1, n + 1)):
            dd = sum((a - b) ** 2 for a, b in zip(rt, perm))
            r2 = 1 - 6 * dd / (n * (n * n - 1))
            tot += 1
            if abs(r2) >= abs(rho) - 1e-12:
                cnt += 1
        return rho, cnt / tot
    import random
    rng = random.Random(123)
    base = list(range(1, n + 1))
    cnt = 0
    reps = 500_000
    for _ in range(reps):
        rng.shuffle(base)
        dd = sum((a - b) ** 2 for a, b in zip(rt, base))
        r2 = 1 - 6 * dd / (n * (n * n - 1))
        if abs(r2) >= abs(rho) - 1e-12:
            cnt += 1
    return rho, (cnt + 1) / (reps + 1)  # Monte Carlo, 500k permutations, seed 123


def wilcoxon_exact(d):
    d = [x for x in d if abs(x) > 1e-12]
    m = len(d)
    absd = sorted(abs(x) for x in d)

    def rk(v):
        idx = [i for i, a in enumerate(absd) if abs(a - v) < 1e-15]
        return sum(i + 1 for i in idx) / len(idx)

    W = sum(rk(abs(x)) for x in d if x > 0)
    allr = [rk(abs(x)) for x in d]
    tot = le = ge = 0
    for signs in product([0, 1], repeat=m):
        w = sum(r for r, s in zip(allr, signs) if s)
        tot += 1
        if w <= W + 1e-9:
            le += 1
        if w >= W - 1e-9:
            ge += 1
    return min(1.0, 2 * min(le, ge) / tot)


def report(label, ps):
    ss = sorted(ps)
    dt = [ps[s][1] - ps[s][0] for s in ss]
    dc = [ps[s][3] - ps[s][2] for s in ss]
    rho, p = spearman_exact(dt, dc)
    print(f"{label}: Spearman rho={rho:.3f} exact two-sided p={p:.4f}")
    a = sum(1 for t, c in zip(dt, dc) if t < 0 and c > 0.02)
    b = sum(1 for t, c in zip(dt, dc) if t < 0 and c <= 0.02)
    c2 = sum(1 for t, c in zip(dt, dc) if t >= 0 and c > 0.02)
    d = sum(1 for t, c in zip(dt, dc) if t >= 0 and c <= 0.02)
    n = len(dt)
    row1, row2, col1 = a + b, c2 + d, a + c2
    if col1 > 0 and row1 > 0:
        def hyp(x):
            return comb(row1, x) * comb(row2, col1 - x) / comb(n, col1)
        p0 = hyp(a)
        pf = sum(hyp(x) for x in range(max(0, col1 - row2), min(row1, col1) + 1)
                 if hyp(x) <= p0 + 1e-12)
        print(f"   Fisher accel x advantage(>0.02): [[{a},{b}],[{c2},{d}]] p={pf:.4f}")
    conc = sum(1 for t, c in zip(dt, dc) if abs(c) > 0.02 and ((t < 0) == (c > 0)))
    disc = sum(1 for t, c in zip(dt, dc) if abs(c) > 0.02 and ((t < 0) != (c > 0)))
    ties = sum(1 for c in dc if abs(c) <= 0.02)
    print(f"   concordance (|dc|>0.02): {conc} concordant, {disc} discordant, {ties} ties")


if __name__ == "__main__":
    report("PRIMARY new pairs (n=5)", NEW)
    report("POOLED all pairs (n=12)", {**OLD, **NEW})
    allp = {**OLD, **NEW}
    ss = sorted(allp)
    dc = [allp[s][3] - allp[s][2] for s in ss]
    dt = [allp[s][1] - allp[s][0] for s in ss]
    print(f"Unconditional Wilcoxon pooled dc (n=12): p={wilcoxon_exact(dc):.4f} ; "
          f"dt: p={wilcoxon_exact(dt):.4f}")
    with open("results/metrics/bridge_compression_extended.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["seed", "batch", "grok99_t0", "grok99_sgld", "timing_delta",
                    "top3_t0", "top3_sgld", "compression_delta"])
        for s in ss:
            g0, g1, c0, c1 = allp[s]
            w.writerow([s, "B" if s in NEW else "A/orig", g0, g1, g1 - g0,
                        c0, c1, round(c1 - c0, 4)])
    print("csv updated")
