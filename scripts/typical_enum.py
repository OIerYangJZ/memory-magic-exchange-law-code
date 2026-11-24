"""typical_enum.py -- the data behind Fig. 1 (Theorems 1 and 2): words near the z-torus, exactly.

Enumerates every single-qubit Clifford+T unitary of minimal T-count t <= TMAX in the
Matsumoto-Amano normal form (T|e)(HT|SHT)^* C, C one of the 24 Cliffords, as in lowcost_counts.py
(same gates, same order, same Clifford list), and keeps the words within projective distance
eps_L = 2^-L of the torus {R_z(theta)}, L = 4..LMAX (default TMAX = 28, LMAX = 12).  Only the top-left entry a of the SU(2) lift
is needed: d_proj(W, torus)^2 = 2 - 2|a|, and d_proj(W, R_z(theta)) <= eps exactly for theta in the
arc |theta - theta_c| <= 2 arccos((1 - eps^2/2)/|a|) (mod 2 pi), theta_c = -2 arg a.

Writes data/typical_enum.json with
  tube[L][t]      number of words of minimal T-count exactly t in the eps_L-tube of the torus
                  (Theorem 2 at the identity coset); its Haar prediction is
                  36 * 2^t * eps^2 (1 - eps^2/4) for t >= 1 (24 * ... for t = 0), the Haar measure of
                  the tube being exactly eps^2 (1 - eps^2/4);
  meas[L][tau]    the fraction of theta in [0, 4 pi) for which R_z(theta) has an eps_L-approximation of
                  T-count <= tau (Theorem 1), the exact measure of a finite union of arcs;
  grid[L][tau]    the fraction of grid angles 2 pi d / Q_eps, d in Z_Q, with such an approximation;
  lam[L][tau]     the Haar expectation of the number of words of T-count <= tau within eps_L of a fixed
                  R_z(theta) (= the measure if the words were spread without overlap);
  taumin[L]       the exact minimal T-count of every grid angle (-1: above TMAX).
meas and grid are exact for tau <= TMAX; they are recorded for tau <= min(TMAX, 3L + 3).

Ties.  A word at distance exactly eps from the torus (it would need |a|^2 = (1 - eps^2/2)^2, denominator
2^(4L+2), hence t >= 8L - 2) or from a grid rotation must be counted ("<= eps").  Every word whose float distance is within TIE of eps is
rebuilt from its gate sequence and re-evaluated with mpmath at 50 digits, as in lowcost_counts.py; the
numbers of such words are recorded (zero for L = 4..12, t <= 28).  Ties do not affect meas (an arc of length zero).

Checks.  For L = 4..8 and t <= 22 the tube counts equal the identity-coset column of data/enum_res2.pkl
(enum2.py), and the grid tau_min equal data/taumin.pkl (L = 4, 5, 6) and data/taumin_exact.json (L = 5, 6).

Run from the repository root:  .venv/bin/python scripts/typical_enum.py [TMAX] [LMAX]
About 3.5 min and 4 GB peak at the default TMAX = 28 (single core, Apple-silicon laptop, numpy 2.5;
TMAX = 22 takes 3 s); levels above 20 are enumerated in chunks, and each extra t doubles the time.
"""
import json, os, sys, time
import numpy as np
import mpmath as mp

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import lowcost_counts as lc          # gates, Clifford list, gate sequences, mpmath evaluation

TMAX = int(sys.argv[1]) if len(sys.argv) > 1 else 28
LMAX = int(sys.argv[2]) if len(sys.argv) > 2 else 12
Ls = list(range(4, LMAX + 1))
TIE = 1e-9
mp.mp.dps = 50


def su2(M):
    return M / np.sqrt(np.linalg.det(M))


# SU(2) lifts of the normal-form letters; prefixes are tracked by their first row only
T1, HT1, SHT1 = su2(lc.T), su2(lc.HT), su2(lc.SHT)
C0 = lc.C24[:, :, 0]                  # first columns of the 24 Cliffords (already SU(2))


def prefix_rows(t):
    """first rows of lc.prefixes(t), same order: [T, HT, SHT], then repeatedly [P HT, P SHT]"""
    if t == 0:
        return np.array([[1, 0]], dtype=complex)
    R = np.array([T1[0], HT1[0], SHT1[0]])
    for _ in range(t - 1):
        R = np.concatenate([R @ HT1, R @ SHT1])
    return R


def mp_entry(t, i):
    """50-digit top-left entry of the SU(2) lift of words_of_level(t)[i]"""
    return lc.mp_word(lc.word_seq(t, i))[0, 0]


eps = {L: 2.0 ** -L for L in Ls}
Q = {L: lc.tuned_Q(eps[L]) for L in Ls}
keep_t = {L: min(TMAX, 3 * L + 3) for L in Ls}      # word lists kept for the lower tail
tube = {L: [0] * (TMAX + 1) for L in Ls}
words = {L: [] for L in Ls}                         # (t, |a|, theta_c, d, within-grid) per kept word
ties_tube = {L: 0 for L in Ls}
ties_grid = {L: 0 for L in Ls}
emin = min(eps.values()); emax = max(eps.values())

BASE = 20                                           # levels above BASE are enumerated in chunks


def suffix(j, k):
    """M_j = X_{b_1} ... X_{b_k}, b_1 the least significant bit of j, X_0 = HT, X_1 = SHT:
    lc.prefixes(B + k)[p0 + n_B j] = lc.prefixes(B)[p0] M_j, n_B = 3 2^(B-1)"""
    M = np.eye(2, dtype=complex)
    for i in range(k):
        M = M @ (SHT1 if (j >> i) & 1 else HT1)
    return M


def level(t):
    """yield (global word indices i = p * 24 + c, top-left entries a) of the words of T-count t
    within the largest tube, chunk by chunk"""
    B = min(t, BASE); k = t - B
    RB = prefix_rows(B); nB = len(RB)
    for j in range(2 ** k):
        Rj = RB @ suffix(j, k)
        for c in range(24):
            Ac = Rj @ C0[c]
            p = np.nonzero(np.abs(Ac) >= 1 - emax ** 2 / 2 - 1e-9)[0]
            yield (p + nB * j) * 24 + c, Ac[p]


t0 = time.time()
for t in range(TMAX + 1):
    cand, a = zip(*level(t))
    cand = np.concatenate(cand); a = np.concatenate(a)
    o = np.argsort(cand); cand, a = cand[o], a[o]
    aa = np.abs(a)
    th = np.mod(-2 * np.angle(a), 2 * np.pi)
    nw = 24 if t == 0 else 72 * 2 ** (t - 1)
    for L in Ls:
        e = eps[L]; c = 1 - e * e / 2
        inside = aa >= c + TIE * e                  # float distance safely below eps ... d^2 = 2 - 2|a|
        border = np.nonzero(np.abs(aa - c) < TIE * e)[0]
        add = []
        for j in border:                            # exact ties: re-evaluate at 50 digits
            ties_tube[L] += 1
            if abs(mp_entry(t, int(cand[j]))) >= 1 - mp.mpf(e) ** 2 / 2 - mp.mpf('1e-35'):
                add.append(j)
        sel = np.concatenate([np.nonzero(inside)[0], np.array(add, dtype=np.int64)])
        tube[L][t] = int(len(sel))
        if t <= keep_t[L] and len(sel):
            Qd = Q[L]
            d = np.round(th[sel] * Qd / (2 * np.pi)).astype(np.int64) % Qd
            re = np.abs(np.real(np.exp(1j * np.pi * d / Qd) * a[sel]))
            dist = np.sqrt(np.maximum(0.0, 2 - 2 * re))
            ok = dist < e - TIE
            for j in np.nonzero(np.abs(dist - e) <= TIE)[0]:
                ties_grid[L] += 1
                dd, dm = lc.mp_near_grid(lc.mp_word(lc.word_seq(t, int(cand[sel[j]]))),
                                         mp.eye(2), mp.eye(2), Qd)
                ok[j] = dm <= e + mp.mpf('1e-35')
                d[j] = dd
            words[L].append((np.full(len(sel), t), aa[sel], th[sel], d, ok))
    print(f"t={t:2d}  words={nw}  tube(eps=2^-4)={tube[4][t] if 4 in tube else '-'}  "
          f"{time.time() - t0:.0f}s", file=sys.stderr, flush=True)


def union_fraction(lo, hi):
    """measure of a union of arcs [lo, hi] (hi - lo < 2 pi) on R / 2 pi Z, as a fraction of 2 pi"""
    width = hi - lo
    lo = np.mod(lo, 2 * np.pi); hi = lo + width
    # split arcs that wrap past 2 pi
    w = hi > 2 * np.pi
    lo = np.concatenate([lo, np.zeros(w.sum())]); hi = np.concatenate([np.minimum(hi, 2 * np.pi), hi[w] - 2 * np.pi])
    o = np.argsort(lo); lo, hi = lo[o], hi[o]
    tot, cur_lo, cur_hi = 0.0, None, None
    for x, y in zip(lo, hi):
        if cur_hi is None or x > cur_hi:
            if cur_hi is not None:
                tot += cur_hi - cur_lo
            cur_lo, cur_hi = x, y
        else:
            cur_hi = max(cur_hi, y)
    if cur_hi is not None:
        tot += cur_hi - cur_lo
    return tot / (2 * np.pi)


def ball_measure(e):
    """Haar measure of {U in PU(2): d_proj(U, V) <= e}: two caps of angular radius rho on S^3"""
    rho = np.arccos(1 - e * e / 2)
    return 2 * (rho - np.sin(rho) * np.cos(rho)) / np.pi


def n_words(t):
    return 24 if t == 0 else 36 * 2 ** t


out = {'TMAX': TMAX, 'Ls': Ls, 'eps': {L: eps[L] for L in Ls}, 'Q': {L: Q[L] for L in Ls},
       'tube': {L: tube[L] for L in Ls},
       'tube_haar_measure': {L: eps[L] ** 2 * (1 - eps[L] ** 2 / 4) for L in Ls},
       'n_words': [n_words(t) for t in range(TMAX + 1)],
       'ties_tube': ties_tube, 'ties_grid': ties_grid,
       'meas': {}, 'grid': {}, 'lam': {}, 'taumin': {}, 'keep_t': keep_t}
for L in Ls:
    e = eps[L]
    tt = np.concatenate([w[0] for w in words[L]]); aa = np.concatenate([w[1] for w in words[L]])
    th = np.concatenate([w[2] for w in words[L]]); dd = np.concatenate([w[3] for w in words[L]])
    ok = np.concatenate([w[4] for w in words[L]])
    h = 2 * np.arccos(np.minimum(1.0, (1 - e * e / 2) / aa))
    taumin = np.full(Q[L], -1, dtype=np.int64)
    for t in range(keep_t[L], -1, -1):              # descending, so the smallest t wins
        taumin[dd[ok & (tt == t)]] = t
    meas, grid, lam = [], [], []
    cum = 0
    for tau in range(keep_t[L] + 1):
        m = tt <= tau
        meas.append(union_fraction(th[m] - h[m], th[m] + h[m]) if m.any() else 0.0)
        grid.append(float(np.mean((taumin >= 0) & (taumin <= tau))))
        cum += n_words(tau)
        lam.append(cum * ball_measure(e))
    out['meas'][L] = [float('%.10g' % x) for x in meas]; out['grid'][L] = grid
    out['lam'][L] = [float('%.10g' % x) for x in lam]
    out['taumin'][L] = taumin.tolist()
    print(f"L={L} Q={Q[L]} kept t<={keep_t[L]} words={len(tt)} ties tube/grid={ties_tube[L]}/{ties_grid[L]}"
          f"  grid covered {np.mean(taumin >= 0):.3f}", file=sys.stderr)

out['runtime_s'] = round(time.time() - t0)
json.dump(out, open(os.path.join(HERE, os.pardir, 'data', 'typical_enum.json'), 'w'), indent=0)
print('wrote data/typical_enum.json', file=sys.stderr)
