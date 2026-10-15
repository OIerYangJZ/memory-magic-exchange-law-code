#!/usr/bin/env python3
"""c_survey.py -- how large must the constant c of Conjecture 1 be at the identity coset (hence at every Clifford
coset), for large T-counts?  For every tau in a range, enumerate all Clifford+T words of minimal T-count <= tau whose
transverse distance to the torus {R_z(theta)} is at most eps_hi(tau) = sqrt(X0 / 2^tau), i.e. the words that matter
near the onset 2^tau eps^2 ~ 1, with torus_tube_enum.enum_torus_level (exact arithmetic, validated word for word
against the exhaustive enumeration), and evaluate

  tube requirement   max over eps <= eps_hi of (N_tube(tau, eps) - c0 - c1 tau) / (2^tau eps^2)   (exact sup; Q-free)
  grid requirement   (N_grid - c0 - c1 tau) / (2^tau eps^2) for the grid {R_z(2 pi d/Q + phi0)}, i.e. the frame
                     (G1, G2) = (I, R_z(phi0)) of Conjecture 1, maximized over the shift phi0 exactly (circular sweep)
                     for Q = Q_eps - j, j < NQ, at eps on a set of trial values; a valid lower bound on the c that
                     Conjecture 1 needs.

Levels: a word of level t within eps has 2^t eps^2 >= 1/4 unless it is one of the eight T-powers (Theorem 2:
a nonzero y2 in Z[omega] has |y2| |y2^bullet| >= 1), so levels t < tau - log2(4 X0) contribute only the T-powers.

The candidate enumeration of torus_tube_enum loops over O(2^(t/4)) values of one coordinate; here it is replaced
by the same enumeration after rescaling by a power of the unit 1 + sqrt2, which balances the two real places and
makes the loop O(sqrt(XY)) (see c_candidates_balanced; checked against the original for t <= 40).

Usage:  python3 c_survey.py validate                  (against data/typical_enum.json, L = 4..12, t <= 28)
        python3 c_survey.py run --tau0 30 --tau1 160 --X0 64 --procs 16 --out c_survey.jsonl
"""
import argparse, json, math, os, sys, time, types

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
try:
    import numba  # noqa: F401
except Exception:                                   # numba is only needed by the all-moduli kernel, unused here
    stub = types.ModuleType('numba')
    stub.njit = lambda *a, **k: (lambda f: f)
    stub.prange = range
    sys.modules['numba'] = stub
import numpy as np
import torus_tube_enum as T

EPS0 = 1.0 + math.sqrt(2.0)


def unit_pow(k):
    """(1 + sqrt2)^k as an element (x, y) of Z[sqrt2], k of either sign"""
    base = (1, 1) if k >= 0 else (-1, 1)
    r = (1, 0)
    for _ in range(abs(k)):
        r = T.zmul(r, base)
    return r


_orig_c_candidates = T._c_candidates


def c_candidates_balanced(X, Y):
    """all C in Z[sqrt2] with |s1(C)| <= X, |s2(C)| <= Y (same margin as the original)"""
    if X <= 0 or Y <= 0:
        return _orig_c_candidates(X, Y)
    k = int(round(math.log(X / Y) / (2 * math.log(EPS0))))
    if k == 0:
        return _orig_c_candidates(X, Y)
    Xp, Yp = X * EPS0 ** (-k), Y * EPS0 ** k          # C = u^k C' with u = 1 + sqrt2
    u = unit_pow(k)
    out = []
    for c in _orig_c_candidates(Xp, Yp):
        C = T.zmul(u, c)
        if abs(T.s1f(C)) <= X * (1 + 1e-9) + 1e-9 and abs(T.s2f(C)) <= Y * (1 + 1e-9) + 1e-9:
            out.append(C)
    return out


T._c_candidates = c_candidates_balanced
C0, C1 = 8, 2


def level_words(t, eps):
    w = T.enum_torus_level(t, eps)
    if not w:
        return np.zeros(0), np.zeros(0), []
    s, phi = T.features(w, t)
    return s, phi, w


def collect(tau, X0):
    eps_hi = min(0.125, math.sqrt(X0 / 2.0 ** tau))
    lo = int(math.floor(tau - math.log2(4 * X0))) - 1
    levels = sorted(set([0, 1] + list(range(max(2, lo), tau + 1))))
    S, P, TT, W = [], [], [], []
    for t in levels:
        s, phi, w = level_words(t, eps_hi)
        S.append(s); P.append(phi); TT.append(np.full(len(s), t)); W.extend(w)
    return eps_hi, np.concatenate(S), np.concatenate(P), np.concatenate(TT), W


def tube_requirement(tau, s):
    d = np.sort(T.tube_dist(s))
    best = (-1e9, None, None)
    # N(eps) = #{d <= eps}; sup of (N - c0 - c1 tau) / (2^tau eps^2) is attained at eps = some d_j > 0
    n = np.arange(1, len(d) + 1)
    # last index of each distinct value
    last = np.r_[d[1:] != d[:-1], True]
    for dj, nj in zip(d[last], n[last]):
        if dj <= 0:
            continue
        r = (nj - C0 - C1 * tau) / (2.0 ** tau * dj * dj)
        if r > best[0]:
            best = (r, float(dj), int(nj))
    return best


def max_shift_count(phi, half, period):
    """max over shifts x of #{i : the circle point x lies in [phi_i - half_i, phi_i + half_i]} (mod period)"""
    full = half * 2 >= period
    base = int(full.sum())
    ph = np.mod(phi[~full], period); hw = half[~full]
    if len(ph) == 0:
        return base, 0.0
    a = np.mod(ph - hw, period); b = np.mod(ph + hw, period)
    ev = np.concatenate([a, b]); typ = np.concatenate([np.ones(len(a)), -np.ones(len(b))])
    wrap = int((a > b).sum())                          # intervals that contain the point 0
    order = np.lexsort((-typ, ev))                     # at equal positions, open (+1) before close (-1)
    cur = wrap; best = cur; bestx = 0.0
    for idx in order:
        cur += int(typ[idx])
        if cur > best:
            best = cur; bestx = float(ev[idx])
    return base + best, bestx


def grid_requirement(tau, s, phi, eps_list, nq=16, nrand=0, seed=0):
    rng = np.random.default_rng(seed + tau)
    best = (-1e9, None)
    dt = T.tube_dist(s)
    sq = np.sqrt(1 - s)
    for eps in eps_list:
        e = eps * (1 - 1e-12)                          # safety margin for the lower bound
        ok = dt <= e
        if not ok.any():
            continue
        # |delta| <= half with 2 s/(1+sq) + 4 sq sin^2(delta/2) <= e^2
        val = (e * e - dt[ok] ** 2) / (4 * sq[ok])
        half = 2 * np.arcsin(np.minimum(1.0, np.sqrt(np.maximum(val, 0.0))))
        Qe = T.tuned_Q(eps)
        Qs = [Qe - j for j in range(nq) if Qe - j >= 3]
        if nrand and Qe > 2 * nq:
            lo = max(3, Qe // 2)
            Qs += [lo + int(rng.random() * (Qe + 1 - lo)) if Qe > 2 ** 52 else int(rng.integers(lo, Qe + 1)) for _ in range(nrand)]
        for Q in Qs:
            period = math.pi / Q                       # phi = arg W00 of the grid points: (pi/Q) Z + shift
            n, x = max_shift_count(phi[ok], half, period)
            r = (n - C0 - C1 * tau) / (2.0 ** tau * eps * eps)
            if r > best[0]:
                best = (r, dict(eps=float(eps), Q=int(Q), N=int(n), shift=x, x=float(2.0 ** tau * eps * eps)))
    return best


def survey_one(args):
    tau, X0, nrand = args
    t0 = time.time()
    eps_hi, s, phi, tt, W = collect(tau, X0)
    tr = tube_requirement(tau, s)
    d = np.sort(np.unique(T.tube_dist(s)))
    d = d[(d > 0) & (d <= eps_hi)]
    trial = set()
    for dj in d[:60]:                                  # the smallest transverse distances carry the steps
        for k in (1e-9, 0.02, 0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0, 1.5):
            e = dj * (1 + k)
            if e <= eps_hi:
                trial.add(e)
    gr = grid_requirement(tau, s, phi, sorted(trial), nrand=nrand)
    # orbits: words of the same level at the same transverse distance (the steps of the count)
    x = (2.0 ** tau) * T.tube_dist(s) ** 2
    key = np.round(np.log(np.maximum(T.tube_dist(s), 1e-300)), 9)
    groups = {}
    for i in np.nonzero((x > 0) & (x <= 8))[0]:
        g = groups.setdefault((int(tt[i]), float(key[i])), [0, float(x[i])]); g[0] += 1
    top = sorted(((v[0], k[0], v[1]) for k, v in groups.items()), reverse=True)[:3]
    # the largest orbit: words at the smallest few distances, with their level
    out = dict(tau=tau, X0=X0, eps_hi=eps_hi, nwords=int(len(s)),
               tube_req=tr[0], tube_eps=tr[1], tube_N=tr[2],
               tube_x=(2.0 ** tau * tr[1] ** 2) if tr[1] else None,
               grid_req=gr[0], grid=gr[1], top_orbits=[dict(size=a, level=b, x=c) for a, b, c in top],
               secs=round(time.time() - t0, 1))
    return out


def cmd_validate(a):
    d = json.load(open(os.path.join(HERE, os.pardir, 'data', 'typical_enum.json')))
    bad = 0
    for L in d['eps']:
        eps = float(d['eps'][L]); ref = d['tube'][L]
        for t in range(len(ref)):
            if ref[t] > 20000:
                continue
            n = len(T.enum_torus_level(t, eps))
            if n != ref[t]:
                bad += 1; print('MISMATCH', L, t, n, ref[t])
        print('L', L, 'ok' if not bad else 'bad', flush=True)
    # balanced vs original candidate lists
    for t in range(4, 41, 3):
        nt = T.n_t(t)
        for eps in (0.1, 0.01, 0.001):
            X = 2 * math.sqrt(T.s1f(nt)) * eps; Y = 2 * math.sqrt(T.s2f(nt))
            A = sorted(_orig_c_candidates(X, Y)); B = sorted(c_candidates_balanced(X, Y))
            if A != B:
                bad += 1; print('CAND MISMATCH', t, eps, len(A), len(B))
    print('validation', 'PASSED' if bad == 0 else f'FAILED ({bad})')


def cmd_run(a):
    from multiprocessing import Pool
    taus = list(range(a.tau0, a.tau1 + 1))
    done = set()
    if os.path.exists(a.out):
        for line in open(a.out):
            try:
                done.add(json.loads(line)['tau'])
            except Exception:
                pass
    todo = [(t, a.X0, a.nrand) for t in taus if t not in done]
    with Pool(a.procs) as pool, open(a.out, 'a') as f:
        for r in pool.imap_unordered(survey_one, todo):
            f.write(json.dumps(r) + '\n'); f.flush()
            g = r['grid'] or {}
            print(f"tau={r['tau']:4d} words={r['nwords']:6d} tube={r['tube_req']:8.1f} (x={r['tube_x']:.2f}) "
                  f"grid>={r['grid_req']:8.1f} (Q={g.get('Q')}, x={g.get('x', 0):.2f}, N={g.get('N')}) "
                  f"orbits={[(o['size'], o['level'], round(o['x'], 2)) for o in r['top_orbits']]} [{r['secs']}s]", flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    sub.add_parser('validate')
    r = sub.add_parser('run')
    r.add_argument('--tau0', type=int, default=30); r.add_argument('--tau1', type=int, default=120)
    r.add_argument('--X0', type=float, default=64.0); r.add_argument('--procs', type=int, default=16)
    r.add_argument('--out', default='c_survey.jsonl')
    r.add_argument('--nrand', type=int, default=0, help='extra moduli sampled uniformly from [Q_eps/2, Q_eps] per eps')
    a = ap.parse_args()
    {'validate': cmd_validate, 'run': cmd_run}[a.cmd](a)
