import json, math, collections, sys
R = [json.loads(l) for l in open(sys.argv[1])]
ok = [r for r in R if r.get("status") == "ok"]
print("max t", max(r["t"] for r in R), "ok", len(ok), "timeouts", sum(1 for r in R if r.get("status") == "timeout"))
for lo, hi in [(20, 100), (100, 200), (200, 300), (300, 400), (400, 600), (600, 1000)]:
    S = [r for r in ok if lo <= r["t"] < hi]
    nz = [r for r in S if r["r"] > 0]
    if not S:
        continue
    h = collections.Counter(int(math.log2(r["r"])) for r in nz)
    lam = 0.5 * math.log((lo + hi) / 2 * math.log(2))
    best = max([16 * r["r"] - 2 * r["t"] for r in nz] or [0])
    print("t in [%d,%d): samples %d, r>0: %d (%.3f); log2 r hist %s; lambda=%.2f; max(16r-2t)=%d"
          % (lo, hi, len(S), len(nz), len(nz) / len(S), dict(sorted(h.items())), lam, best))
