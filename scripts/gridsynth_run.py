"""Fig. 3(b) and Table I: Ross-Selinger T-counts at eps=1e-10 (chemistry scale).

Under canonical sharing the share u is uniform on Z_Q, and the frontier only uses
E_u tau(u), so we estimate that expectation from N random grid angles 2*pi*u/Q
rather than synthesizing all Q_eps ~ 7.9e9 of them.

Accuracy convention.  The accuracy of the manuscript is d_proj(A, B) = inf_phi ||A - e^{i phi} B||.
pygridsynth 2.0.0 (up_to_phase=True) accepts a candidate u when Re(conj(z) u) >= sqrt(1 - e^2/4)
(region.py / gridsynth.py), i.e. its parameter e is a diamond-norm-type distance and the word it
returns has d_proj <= e/2 (measured: worst d_proj / e = 0.499).  This script therefore takes the
target d_proj (--dproj) and calls pygridsynth with e = 2 * dproj; every run re-synthesizes 200 angles
and records the worst d_proj actually achieved, which must be <= dproj.

--eps is the accuracy eps of the tuned family: it fixes the grid Q = Q_eps (unless --Q is given).
--dproj is the accuracy of each synthesized word; the default eps/2 is the share accuracy that the
certified two-round point of Theorem 5 (r = 2) needs.  From the repository root:

  .venv/bin/python scripts/gridsynth_run.py
        # d_proj <= eps/2 = 5e-11 on the eps = 1e-10 grid: data/gridsynth_tau.json (E tau = 102.32)
  .venv/bin/python scripts/gridsynth_run.py --dproj 1e-10 --out gridsynth_tau_dproj_eps.json
        # d_proj <= eps: the single-rotation calibration

The paper uses N = 20000 (the default; the seed is fixed, so the angles are those of
data/u_20000.txt).  A smaller --n overwrites the output file with a smaller sample; use --out to
write elsewhere.  Results are cached by content (angle numerator, modulus, pygridsynth parameter e)
in data/gridsynth_cache.json, so reruns are free.
"""
import argparse
import json
import multiprocessing as mp
import os
import random
import time

import mpmath
from pygridsynth import GridsynthConfig, get_synthesized_unitary, gridsynth_gates

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, os.pardir, "data")

# up_to_phase=True: the phase is irrelevant, as for d_proj.  The pygridsynth parameter e is NOT d_proj:
# the returned word has d_proj <= e/2 (see the module docstring), so we pass e = 2 * dproj.
CFG = dict(up_to_phase=True)


def modulus(eps, dps=60):
    """Q_eps = floor(pi / (2 arcsin 2 eps)) as in Paper 1, Cor. 7."""
    with mpmath.workdps(dps):
        return int(mpmath.floor(mpmath.pi / (2 * mpmath.asin(2 * mpmath.mpf(eps)))))


def tcount(job):
    """T-count of the pygridsynth word at angle 2*pi*u/Q for pygridsynth parameter eps (= 2 d_proj)."""
    u, Q, eps, dps = job
    if u == 0:
        return u, 0, ""  # a zero share costs nothing
    with mpmath.workdps(dps):
        theta = 2 * mpmath.pi * mpmath.mpf(u) / mpmath.mpf(Q)
        gates = gridsynth_gates(theta, mpmath.mpf(eps), cfg=GridsynthConfig(**CFG))
    return u, gates.count("T"), gates


def dproj(gates, u, Q, dps=60):
    """inf_phi ||W - e^{i phi} R_z(2 pi u / Q)|| = sqrt(2 - |tr(V^dag W)|)."""
    with mpmath.workdps(dps):
        theta = 2 * mpmath.pi * mpmath.mpf(u) / mpmath.mpf(Q)
        W = get_synthesized_unitary(gates, dps=dps)
        V = mpmath.matrix([[mpmath.exp(-1j * theta / 2), 0],
                           [0, mpmath.exp(1j * theta / 2)]])
        M = V.H * W
        return float(mpmath.sqrt(abs(2 - abs(M[0, 0] + M[1, 1]))))


def run(jobs, cache, procs):
    todo = [j for j in jobs if str((j[0], j[1], j[2])) not in cache]
    if todo:
        t0 = time.time()
        with mp.Pool(procs) as pool:
            for i, (u, t, g) in enumerate(pool.imap_unordered(tcount, todo, 16), 1):
                cache[str((u, todo[0][1], todo[0][2]))] = t
                if i % 250 == 0:
                    print(f"    {i}/{len(todo)}  ({time.time() - t0:.0f}s)", flush=True)
    return [cache[str((j[0], j[1], j[2]))] for j in jobs]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eps", default="1e-10", help="accuracy of the tuned family (fixes Q = Q_eps)")
    ap.add_argument("--dproj", default=None, help="target d_proj of each word (default eps/2)")
    ap.add_argument("--n", type=int, default=20000)
    ap.add_argument("--lmax", type=int, default=8)
    ap.add_argument("--seed", type=int, default=20260903)
    ap.add_argument("--procs", type=int, default=os.cpu_count())
    ap.add_argument("--Q", type=int, default=None,
                    help="hold the modulus fixed instead of tuning it to --eps")
    ap.add_argument("--out", default="gridsynth_tau.json")
    args = ap.parse_args()

    eps_family = args.eps
    Q = args.Q if args.Q is not None else modulus(eps_family)
    dproj_target = float(args.dproj) if args.dproj is not None else float(eps_family) / 2
    eps = f"{2 * dproj_target:g}"          # pygridsynth parameter e = 2 d_proj (also the cache key)
    dps = 40 + 2 * int(-mpmath.log10(mpmath.mpf(eps)))
    with mpmath.workdps(60):
        sep_ok = mpmath.mpf(eps_family) < mpmath.sin(mpmath.pi / (2 * Q))
    print(f"family eps={eps_family}  Q={Q}  K={Q - 1}  log2 K={mpmath.log(Q - 1, 2)}  "
          f"separation ok={sep_ok};  target d_proj={dproj_target:g}  pygridsynth e={eps}")

    os.makedirs(DATA, exist_ok=True)
    cpath = os.path.join(DATA, "gridsynth_cache.json")
    cache = json.load(open(cpath)) if os.path.exists(cpath) else {}

    # (1) E_u tau(u) over uniform shares.
    rng = random.Random(args.seed)
    us = [rng.randrange(Q) for _ in range(args.n)]
    print(f"[1] {args.n} uniform shares")
    taus = run([(u, Q, eps, dps) for u in us], cache, args.procs)

    # (3) partial commitment: the low ell bits of a share, ell = 1..lmax.
    #     k < 2^ell is at most 2^lmax values, so enumerate them exactly.
    ks = list(range(2 ** args.lmax))
    print(f"[3] {len(ks)} small angles k < 2^{args.lmax}")
    ktaus = dict(zip(ks, run([(k, Q, eps, dps) for k in ks], cache, args.procs)))

    json.dump(cache, open(cpath, "w"))

    # verification: d_proj <= dproj_target on a random subsample
    print("[v] re-synthesizing 200 samples to check the achieved d_proj")
    check = rng.sample(us, 200)
    worst = 0.0
    with mp.Pool(args.procs) as pool:
        for u, t, g in pool.imap_unordered(tcount, [(u, Q, eps, dps) for u in check]):
            worst = max(worst, dproj(g, u, Q))
    print(f"    worst d_proj = {worst:.3e}  (target {dproj_target:.1e}, pygridsynth e = {eps})")
    assert worst <= dproj_target, "achieved d_proj exceeds the target"

    n = len(taus)
    mean = sum(taus) / n
    var = sum((t - mean) ** 2 for t in taus) / (n - 1)
    se = (var / n) ** 0.5
    L = float(-mpmath.log(mpmath.mpf(eps_family), 2))
    lK = float(mpmath.log(Q - 1, 2))
    out = dict(
        eps=eps_family, dproj=dproj_target, gridsynth_e=eps, Q=Q, K=Q - 1, L=L, log2K=lK, n=n,
        seed=args.seed,
        Q_tuned_to_eps=(args.Q is None), up_to_phase=True,
        tau_mean=mean, tau_se=se, tau_sd=var ** 0.5,
        tau_min=min(taus), tau_max=max(taus),
        slope=mean / lK, slope_se=se / lK,
        worst_dproj=worst,
        partial={str(l): sum(ktaus[k] for k in range(2 ** l)) / 2 ** l
                 for l in range(1, args.lmax + 1)},
        ktaus={str(k): v for k, v in ktaus.items()},
        taus=taus,
    )
    json.dump(out, open(os.path.join(DATA, args.out), "w"), indent=1)

    print(f"\ntau_bar = {mean:.3f} +- {se:.3f}   (sd {var ** 0.5:.2f}, "
          f"range {min(taus)}-{max(taus)}),  3L = {3 * L:.1f}")
    print(f"slope tau_bar/log2 K = {mean / lK:.4f} +- {se / lK:.4f}")
    for l in range(1, args.lmax + 1):
        Tk = out["partial"][str(l)]
        print(f"  ell={l}: E_k tau = {Tk:8.3f}   T per bit shed = {Tk / l:7.3f}")


if __name__ == "__main__":
    main()
