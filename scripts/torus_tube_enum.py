#!/usr/bin/env python3
"""torus_tube_enum.py -- all single-qubit Clifford+T unitaries of minimal T-count <= tau within projective
distance eps of the torus {R_z(theta)}, through the (c,d) parametrisation of the proof of Theorem 2
(Appendix "Proofs of Theorems 1 and 2" of the main text), with exact arithmetic.

Conventions (as in the main text).  A quaternion a+bi+cj+dk is the SU(2) matrix [[a+b*1j, c+d*1j], [-c+d*1j, a-b*1j]];
R_z(theta) is cos(theta/2) - sin(theta/2) i.  The maximal order is
    O = Z[r2] + Z[r2](1+i)/r2 + Z[r2](1+j)/r2 + Z[r2](1+i+j+k)/2,   r2 = sqrt(2),
and we store w in O through 2w = A + B i + C j + D k with A, B, C, D in Z[r2], each an integer pair (x, y) = x + y r2.
In these coordinates  w in O  <=>  A0 = B0 = C0 = D0 (mod 2)  and  A1 + B1 + C1 + D1 = 0 (mod 2).
A unitary of minimal T-count t is +-w/|w| for exactly two w in O with nrd(w) = n_t (n_t = 2^(t/2) for even t,
(2+r2) 2^((t-1)/2) for odd t), namely the two *primitive* ones (w/r2 not in O); a numerator with nrd(w) = n_t that is
divisible by r2 is r2^s times a primitive numerator of T-count t - 2s and is discarded at level t.

Tube.  d_proj(W, R_z(theta)) = sqrt(2 - |tr(R_z(theta)^dag W)|), so "some theta within eps" <=> |W_00| >= 1 - eps^2/2
<=> c^2 + d^2 <= (eps^2 - eps^4/4) nrd(w)  (at the first real place sigma_1), and at the second place
sigma_2(c)^2 + sigma_2(d)^2 < sigma_2(n_t).  For each admissible (C, D) the pair (A, B) gives alpha = A + B i in
Z[r2, i] (subset of Z[omega]) with alpha * conj(alpha) = M := 4 n_t - C^2 - D^2; all such alpha are found by factoring
N(M) in Z (python-flint) and splitting the primes in Z[omega] (units of relative norm 1: the eight omega^k).
The tube decision is exact (eps is a binary64 number, i.e. an exact dyadic rational; comparisons in Q(r2) are exact).

Grid.  "exists d in Z_Q: d_proj(W, R_z(2 pi d/Q)) <= eps": the nearest grid point is found in floating point
(numerically stable formulas, relative error ~1e-10 at eps = 2^-20); a word whose float distance is within TIE_REL*eps
of eps is re-evaluated from its exact coordinates with mpmath at 60 digits, and a distance within MP_TIE of eps is an
exact tie and counts as within, as d_proj <= eps requires (cf. TIE / MP_TIE in lowcost_counts.py).

Arithmetic frames.  For G1, G2 in Gamma (words of T-count t1, t2), W in the eps-tube (grid) of G1 R_z G2 iff
V = G1^-1 W G2^-1 is in the eps-tube (grid) of the torus, at the same distance, and t_min(V) <= t_min(W) + t1 + t2.
So the torus enumeration to T-count tau + t1 + t2 contains every W of T-count <= tau; t_min(W) is recomputed
exactly from w = g1 v g2 (v_P(nrd w) minus twice the r2-adic valuation of w in O).

Usage (from the repository root; python-flint is required):
  python scripts/torus_tube_enum.py validate  [--tmax 22]           exact comparison with exhaustive enumeration
  python scripts/torus_tube_enum.py band      --L 10 --K 12 ...      enumerate one accuracy band and analyse it
  (see --help of each subcommand and /tmp/conj1/REPORT.md for the exact commands used)
"""
import argparse, collections, hashlib, json, math, os, subprocess, sys, time
from array import array
from fractions import Fraction
from functools import lru_cache

import numpy as np
import flint
import mpmath as mp

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, os.pardir, 'data')
R2 = math.sqrt(2.0)
LOG_U = math.log(1.0 + R2)
mp.mp.dps = 60
TIE_REL = 1e-7                 # float distances within TIE_REL*eps of eps are re-evaluated with mpmath
MP_TIE = mp.mpf('1e-40')       # mpmath distances within MP_TIE of eps are exact ties (count as within)

def code_version():
    h = hashlib.sha256(open(os.path.abspath(__file__), 'rb').read()).hexdigest()[:16]
    try:
        g = subprocess.run(['git', '-C', HERE, 'rev-parse', '--short', 'HEAD'], capture_output=True, text=True).stdout.strip()
    except Exception:
        g = '?'
    return {'script_sha256_16': h, 'git_head': g, 'python_flint': flint.__version__, 'numpy': np.__version__}

# ============================================================== Z[r2]: (x, y) = x + y r2
def zmul(a, b): return (a[0] * b[0] + 2 * a[1] * b[1], a[0] * b[1] + a[1] * b[0])
def zadd(a, b): return (a[0] + b[0], a[1] + b[1])
def zsub(a, b): return (a[0] - b[0], a[1] - b[1])
def zconj(a): return (a[0], -a[1])
def znorm(a): return a[0] * a[0] - 2 * a[1] * a[1]
def zsq(a): return (a[0] * a[0] + 2 * a[1] * a[1], 2 * a[0] * a[1])

def zdiv(a, b):
    """a/b if it lies in Z[r2], else None"""
    n = znorm(b); x, y = zmul(a, zconj(b))
    if x % n or y % n:
        return None
    return (x // n, y // n)

def sign_qr2(x, y):
    """exact sign of x + y r2 for x, y integers or Fractions"""
    if x >= 0 and y >= 0: return 1 if (x > 0 or y > 0) else 0
    if x <= 0 and y <= 0: return -1
    d = x * x - 2 * y * y
    if x > 0: return (d > 0) - (d < 0)           # x > 0 > y
    return (d < 0) - (d > 0)                     # x < 0 < y

def s1f(a):
    """sigma_1(a) = a0 + a1 r2 in floating point, without cancellation"""
    x, y = a
    if (x >= 0) == (y >= 0) or x == 0 or y == 0:
        return x + y * R2
    return znorm(a) / (x - y * R2)

def s2f(a):
    x, y = a
    if (x >= 0) != (y >= 0) or x == 0 or y == 0:
        return x - y * R2
    return znorm(a) / (x + y * R2)

def zpow(a, n):
    r = (1, 0)
    for _ in range(n): r = zmul(r, a)
    return r

def n_t(t):
    return (2 ** (t // 2), 0) if t % 2 == 0 else (2 ** ((t + 1) // 2), 2 ** ((t - 1) // 2))

# ============================================================== Z[omega]: (x0,x1,x2,x3) = sum x_k omega^k, omega^4 = -1
def wmul(a, b):
    a0, a1, a2, a3 = a; b0, b1, b2, b3 = b
    return (a0 * b0 - a1 * b3 - a2 * b2 - a3 * b1,
            a0 * b1 + a1 * b0 - a2 * b3 - a3 * b2,
            a0 * b2 + a1 * b1 + a2 * b0 - a3 * b3,
            a0 * b3 + a1 * b2 + a2 * b1 + a3 * b0)
def wconj(a): return (a[0], -a[3], -a[2], -a[1])            # complex conjugation omega -> omega^7
def wtau5(a): return (a[0], -a[1], a[2], -a[3])             # omega -> omega^5 = -omega (r2 -> -r2, i -> i)
def wom(a): return (-a[3], a[0], a[1], a[2])                # times omega
def wemb(z): return (z[0], z[1], 0, -z[1])                  # Z[r2] -> Z[omega], r2 = omega - omega^3
def wrelnorm(a):
    r = wmul(a, wconj(a))
    assert r[2] == 0 and r[3] == -r[1], r
    return (r[0], r[1])
def wpow(a, n):
    r = (1, 0, 0, 0)
    for _ in range(n): r = wmul(r, a)
    return r
ONE_PLUS_OMEGA = (1, 1, 0, 0)                               # relative norm 2 + r2 = r2 (1 + r2)

def _gauss_reduce(u, v, k):
    """shortest vector of the lattice Z u + Z v for the form x^2 + k y^2 (Lagrange-Gauss reduction)"""
    B = lambda a, b: a[0] * b[0] + k * a[1] * b[1]
    nu, nv = B(u, u), B(v, v)
    if nu > nv: u, v, nu, nv = v, u, nv, nu
    while True:
        b = B(u, v); m = (2 * b + nu) // (2 * nu)
        v = (v[0] - m * u[0], v[1] - m * u[1]); nv = B(v, v)
        if nv >= nu: return u
        u, v, nu, nv = v, u, nv, nu

@lru_cache(maxsize=1 << 18)
def prime_data(p):
    """splitting data of the rational prime p in Z[r2] and Z[omega]"""
    if p == 2:
        return ('ram',)
    r = p % 8
    if r == 1:                                    # p = pi1 pi2 in Z[r2], each pi splits in Z[omega]
        g = 2
        while pow(g, (p - 1) // 2, p) != p - 1: g += 1
        z = pow(g, (p - 1) // 8, p)               # primitive 8th root of unity mod p
        basis = [[p, 0, 0, 0], [(-z) % p, 1, 0, 0], [(-z * z) % p, 0, 1, 0], [(-z * z * z) % p, 0, 0, 1]]
        red = [tuple(int(x) for x in row) for row in flint.fmpz_mat(basis).lll().tolist()]
        rho = None
        for c in red:
            if abs(znorm(wrelnorm(c))) == p: rho = c; break
        if rho is None:                           # a vector of norm 2p: divide by 1 + omega
            for c in red:
                y = wmul(wmul(c, (1, 0, 0, -1)), wemb((2, -1)))
                if all(x % 2 == 0 for x in y):
                    y = tuple(x // 2 for x in y)
                    if abs(znorm(wrelnorm(y))) == p: rho = y; break
        assert rho is not None, p
        pi1 = wrelnorm(rho)
        return ('split', rho, pi1, wtau5(rho), zconj(pi1))
    if r == 7:                                    # p = pi pi' in Z[r2], each inert in Z[omega]
        s = int(flint.fmpz(2).sqrtmod(p))
        pi = _gauss_reduce((p, 0), ((-s) % p, 1), 2)
        assert abs(znorm(pi)) == p, (p, pi)
        return ('inert', pi, zconj(pi))
    if r == 5:                                    # p inert in Z[r2], p = rho conj(rho), rho = x + y i
        s = int(flint.fmpz(p - 1).sqrtmod(p))
        x, y = _gauss_reduce((p, 0), (s, 1), 1)
        assert x * x + y * y == p
        return ('split_inert', (x, 0, y, 0))
    s = int(flint.fmpz(p - 2).sqrtmod(p))         # r == 3: p = x^2 + 2 y^2, rho = x + y r2 i = x + y (omega + omega^3)
    x, y = _gauss_reduce((p, 0), (s, 1), 2)
    assert x * x + 2 * y * y == p
    return ('split_inert', (x, y, 0, y))

def zval(M, pi, emax):
    v = 0
    while v < emax:
        q = zdiv(M, pi)
        if q is None: break
        M = q; v += 1
    return v

def relnorm_solutions(M):
    """all alpha in Z[omega] with alpha * conj(alpha) = M  (M in Z[r2] totally positive)"""
    N = znorm(M)
    assert N > 0 and M[0] > 0
    opts = []
    for (p, e) in flint.fmpz(N).factor():
        p = int(p); e = int(e); pd = prime_data(p)
        kind = pd[0]
        if kind == 'ram':
            opts.append([wpow(ONE_PLUS_OMEGA, e)])
        elif kind == 'split_inert':
            assert e % 2 == 0
            v = e // 2; rho = pd[1]; rc = wconj(rho)
            opts.append([wmul(wpow(rho, a), wpow(rc, v - a)) for a in range(v + 1)])
        elif kind == 'inert':
            pi, pi2 = pd[1], pd[2]
            v1 = zval(M, pi, e); v2 = e - v1
            if v1 % 2 or v2 % 2: return []
            opts.append([wemb(zmul(zpow(pi, v1 // 2), zpow(pi2, v2 // 2)))])
        else:
            rho1, pi1, rho2, pi2 = pd[1], pd[2], pd[3], pd[4]
            v1 = zval(M, pi1, e); v2 = e - v1
            r1c, r2c = wconj(rho1), wconj(rho2)
            o1 = [wmul(wpow(rho1, a), wpow(r1c, v1 - a)) for a in range(v1 + 1)]
            o2 = [wmul(wpow(rho2, b), wpow(r2c, v2 - b)) for b in range(v2 + 1)]
            opts.append([wmul(x, y) for x in o1 for y in o2])
    # base product and the unit correction (common to all choices: same relative norm)
    sols = [(1, 0, 0, 0)]
    for o in opts:
        sols = [wmul(s, x) for s in sols for x in o]
    u = zdiv(wrelnorm(sols[0]), M)
    assert u is not None and abs(znorm(u)) == 1, (M, u)
    m = round(math.log(s1f(u)) / (2 * LOG_U))
    corr = wemb(zpow((-1, 1), m)) if m > 0 else wemb(zpow((1, 1), -m))   # (1+r2)^(-m)
    out = []
    for s in sols:
        a = wmul(s, corr)
        for _ in range(8):
            out.append(a); a = wom(a)
    assert wrelnorm(out[0]) == M, (M, out[0])
    return out

# ============================================================== quaternions over Z[r2] (stored as 2w)
def in_order(Q):
    """Q = (A, B, C, D), each an integer pair: is Q/2 in O?"""
    (A0, A1), (B0, B1), (C0, C1), (D0, D1) = Q
    return (A0 - B0) % 2 == 0 and (A0 - C0) % 2 == 0 and (A0 - D0) % 2 == 0 and (A1 + B1 + C1 + D1) % 2 == 0

def div_r2(Q):
    """(2w)/r2 as integer coordinates if w/r2 in O, else None"""
    out = []
    for (x, y) in Q:
        if x % 2: return None
        out.append((y, x // 2))
    out = tuple(out)
    return out if in_order(out) else None

def is_primitive(Q):
    return div_r2(Q) is None

def qmul2(P, Q):
    """(2p)(2q)/2 = 2(pq): product in the 2w-scaling; returns None if not integral"""
    a1, b1, c1, d1 = P; a2, b2, c2, d2 = Q
    m = lambda x, y: zmul(x, y)
    A = zsub(zsub(zsub(m(a1, a2), m(b1, b2)), m(c1, c2)), m(d1, d2))
    B = zsub(zadd(zadd(m(a1, b2), m(b1, a2)), m(c1, d2)), m(d1, c2))
    C = zadd(zsub(zadd(m(a1, c2), m(c1, a2)), m(b1, d2)), m(d1, b2))
    D = zsub(zadd(zadd(m(a1, d2), m(d1, a2)), m(b1, c2)), m(c1, b2))
    R = (A, B, C, D)
    if any(x % 2 or y % 2 for (x, y) in R): return None
    return tuple((x // 2, y // 2) for (x, y) in R)

def qnrd4(Q):
    """4 nrd(w) = A^2 + B^2 + C^2 + D^2 in Z[r2]"""
    r = (0, 0)
    for x in Q: r = zadd(r, zsq(x))
    return r

def qinv2(Q):
    """conjugate (= nrd * inverse)"""
    A, B, C, D = Q
    return (A, (-B[0], -B[1]), (-C[0], -C[1]), (-D[0], -D[1]))

GATE_Q = {'T': ((2, 1), (0, -1), (0, 0), (0, 0)),        # 2[(1 + r2/2) - (r2/2) i], nrd 2 + r2
          'H': ((0, 0), (0, 1), (0, 0), (0, 1)),         # 2(i + k)/r2
          'S': ((0, 1), (0, -1), (0, 0), (0, 0))}        # 2(1 - i)/r2

def word_quat(seq):
    """exact numerator of a gate string (left to right = matrix product), its T-count v_P(nrd) before reduction"""
    Q = ((2, 0), (0, 0), (0, 0), (0, 0)); tv = 0
    for g in seq:
        Q = qmul2(Q, GATE_Q[g]); tv += (g == 'T')
    return Q, tv

def reduce_numerator(Q, tv):
    """strip powers of r2: returns (primitive numerator, minimal T-count)"""
    while True:
        R = div_r2(Q)
        if R is None: return Q, tv
        Q = R; tv -= 2

def canon_sign(Q):
    for (x, y) in Q:
        if x: return Q if x > 0 else tuple((-a, -b) for (a, b) in Q)
        if y: return Q if y > 0 else tuple((-a, -b) for (a, b) in Q)
    return Q

# ============================================================== torus enumeration, one level
def _c_candidates(X, Y):
    """all C in Z[r2] with |s1(C)| <= X, |s2(C)| <= Y (with a small safety margin)"""
    X += 1e-9 * (1 + X); Y += 1e-9 * (1 + Y)
    out = []
    m = int((X + Y) / (2 * R2)) + 1
    for c1 in range(-m, m + 1):
        lo = max(-X - c1 * R2, -Y + c1 * R2); hi = min(X - c1 * R2, Y + c1 * R2)
        for c0 in range(math.ceil(lo), math.floor(hi) + 1):
            out.append((c0, c1))
    return out

def enum_torus_level(t, eps, stats=None, as_tuples=True):
    """all primitive numerators 2w = (A,B,C,D) with nrd(w) = n_t and |W_00| >= 1 - eps^2/2, one of each sign pair.
    Returns a list of 8-tuples of ints (A0,A1,B0,B1,C0,C1,D0,D1)."""
    nt = n_t(t); nt4 = (4 * nt[0], 4 * nt[1])
    e = Fraction(eps); r = e * e - e ** 4 / 4                # exact
    R1sq = 4 * s1f(nt) * float(r); R2sq = 4 * s2f(nt)
    cands = _c_candidates(math.sqrt(R1sq), math.sqrt(R2sq))
    if not cands: return [] if as_tuples else np.zeros((0, 8), dtype=np.int64)
    C0 = np.array([c[0] for c in cands], dtype=np.int64); C1 = np.array([c[1] for c in cands], dtype=np.int64)
    S1 = np.array([s1f(c) for c in cands]); S2 = np.array([s2f(c) for c in cands])
    keep = (S1 * S1 <= R1sq * (1 + 1e-9) + 1e-12) & (S2 * S2 <= R2sq * (1 + 1e-9) + 1e-12)
    C0, C1, S1, S2 = C0[keep], C1[keep], S1[keep], S2[keep]
    # exact tube test for a pair: sigma_1(4 n_t r - C^2 - D^2) >= 0, sigma_2(4 n_t - C^2 - D^2) > 0
    rx, ry = nt4[0] * r, nt4[1] * r
    cache = {}
    out = array('q')
    npairs = 0
    for i in range(len(C0)):
        par = C0[i] & 1
        sel = ((C0 & 1) == par)
        a1 = S1[i] * S1[i] + S1 * S1; a2 = S2[i] * S2[i] + S2 * S2
        ok = sel & (a1 <= R1sq * (1 + 1e-9) + 1e-12) & (a2 <= R2sq * (1 + 1e-9) + 1e-12)
        Ci = (int(C0[i]), int(C1[i]))
        for j in np.nonzero(ok)[0]:
            Dj = (int(C0[j]), int(C1[j]))
            xi = zadd(zsq(Ci), zsq(Dj))
            if sign_qr2(rx - xi[0], ry - xi[1]) < 0: continue
            M = zsub(nt4, xi)
            if sign_qr2(M[0], -M[1]) <= 0: continue      # sigma_2(M) > 0 (sigma_1(M) > 0 follows)
            npairs += 1
            key = (M, par, (Ci[1] + Dj[1]) & 1)
            sols = cache.get(key)
            if sols is None:
                sols = []
                for (x0, x1, x2, x3) in relnorm_solutions(M):
                    if (x1 - x3) & 1 or (x0 - par) & 1 or (x2 - par) & 1 or (x1 - key[2]) & 1:
                        continue
                    sols.append(((x0, (x1 - x3) // 2), (x2, (x1 + x3) // 2)))
                cache[key] = sols
            for (A, B) in sols:
                Q = (A, B, Ci, Dj)
                if not is_primitive(Q): continue
                Qc = canon_sign(Q)
                if Qc is not Q: continue                  # keep one of +-w
                out.extend((A[0], A[1], B[0], B[1], Ci[0], Ci[1], Dj[0], Dj[1]))
    if stats is not None:
        stats['pairs'] = stats.get('pairs', 0) + npairs
        stats['cands'] = stats.get('cands', 0) + len(C0)
    return [tuple(out[k:k + 8]) for k in range(0, len(out), 8)] if as_tuples else np.frombuffer(out, dtype=np.int64).reshape(-1, 8).copy()

# ============================================================== features and distances
def features(words, t):
    """float features of torus words at level t: s = |W_01|^2 (transverse), phi = arg W_00"""
    nt = n_t(t); n4 = 4 * s1f(nt)
    s = np.empty(len(words)); phi = np.empty(len(words))
    for k, w in enumerate(words):
        w = [int(x) for x in w]
        c1 = s1f((w[4], w[5])); d1 = s1f((w[6], w[7]))
        s[k] = (c1 * c1 + d1 * d1) / n4
        phi[k] = math.atan2(s1f((w[2], w[3])), s1f((w[0], w[1])))
    return s, phi

def tube_dist(s):
    return np.sqrt(2 * s / (1 + np.sqrt(1 - s)))

def grid_dist(s, phi, Q):
    x = phi * Q / np.pi
    delta = (x - np.round(x)) * np.pi / Q
    sq = np.sqrt(1 - s)
    return np.sqrt(2 * s / (1 + sq) + 4 * sq * np.sin(delta / 2) ** 2)

def mp_features(w, t):
    r2 = mp.sqrt(2)
    s1 = lambda a, b: mp.mpf(int(a)) + mp.mpf(int(b)) * r2
    nt = n_t(t)
    n4 = 4 * s1(*nt)
    s = (s1(w[4], w[5]) ** 2 + s1(w[6], w[7]) ** 2) / n4
    phi = mp.atan2(s1(w[2], w[3]), s1(w[0], w[1]))
    return s, phi

def mp_tube_dist(s):
    return mp.sqrt(2 * s / (1 + mp.sqrt(1 - s)))

def mp_grid_dist(s, phi, Q):
    x = phi * Q / mp.pi
    delta = (x - mp.nint(x)) * mp.pi / Q
    sq = mp.sqrt(1 - s)
    return mp.sqrt(2 * s / (1 + sq) + 4 * sq * mp.sin(delta / 2) ** 2)

def within(dist, eps, exact_words, tvals, kind, Q=None, log=None):
    """boolean mask dist <= eps with near-ties resolved in high precision from the exact coordinates"""
    m = dist <= eps
    border = np.nonzero(np.abs(dist - eps) <= TIE_REL * eps)[0]
    for i in border:
        s, phi = mp_features(exact_words[i], int(tvals[i]))
        dm = mp_tube_dist(s) if kind == 'tube' else mp_grid_dist(s, phi, Q)
        m[i] = dm <= mp.mpf(eps) + MP_TIE
        if log is not None and abs(dm - mp.mpf(eps)) <= MP_TIE:
            log.append(('tie', kind, Q, int(tvals[i]), exact_words[i]))
    return m

def tuned_Q(eps):
    return int(math.floor(math.pi / (2 * math.asin(2 * eps))))

def eps_max_of_Q(Q):
    """largest eps with Q <= Q_eps: sin(pi/2Q) >= 2 eps"""
    return math.sin(math.pi / (2 * Q)) / 2


# ============================================================== exhaustive reference (Matsumoto-Amano normal form)
_H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
_S = np.array([[1, 0], [0, 1j]], dtype=complex)
_T = np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]], dtype=complex)

def _cliffords():
    def key(M):
        M = M / np.sqrt(np.linalg.det(M)); f = M.flatten()
        i = next(k for k in range(4) if abs(f[k]) > 1e-9); f = f / (f[i] / abs(f[i]))
        return tuple(np.round(f, 7))
    seen = {key(np.eye(2)): np.eye(2, dtype=complex)}; fr = [np.eye(2, dtype=complex)]
    while fr:
        new = []
        for M in fr:
            for g in (_H, _S):
                N = M @ g; k = key(N)
                if k not in seen: seen[k] = N; new.append(N)
        fr = new
    assert len(seen) == 24
    return np.array(list(seen.values()))

def exhaustive_tube_level(t, eps):
    """all words (T|e)(HT|SHT)^* C of T-count exactly t with |W_00| >= 1 - eps^2/2 (float), as SU(2) quaternions
    (a,b,c,d) with W = [[a+bi, c+di], [-c+di, a-bi]]; independent of the (c,d) machinery."""
    CL = _cliffords()
    HT, SHT = _H @ _T, _S @ _H @ _T
    if t == 0:
        P = np.array([[1, 0]], dtype=complex); dP = np.array([1], dtype=complex)
    else:
        P = np.array([_T[0], HT[0], SHT[0]]); dP = np.array([np.linalg.det(_T), np.linalg.det(HT), np.linalg.det(SHT)])
        for _ in range(t - 1):
            P = np.concatenate([P @ HT, P @ SHT]); dP = np.concatenate([dP * np.linalg.det(HT), dP * np.linalg.det(SHT)])
    thr = 1 - eps ** 2 / 2
    out = []
    for C in CL:
        x = P[:, 0] * C[0, 0] + P[:, 1] * C[1, 0]
        m = np.abs(x) >= thr - 1e-12
        if not m.any(): continue
        y = P[m, 0] * C[0, 1] + P[m, 1] * C[1, 1]
        sq = np.sqrt(dP[m] * np.linalg.det(C))
        x = x[m] / sq; y = y / sq
        out.append(np.stack([x.real, x.imag, y.real, y.imag], 1))
    return np.concatenate(out) if out else np.zeros((0, 4))

def canon_float(q):
    """sign-canonical float quaternions (first coordinate with |x| > 1e-6 positive)"""
    q = q.copy()
    for k in range(len(q)):
        for x in q[k]:
            if abs(x) > 1e-6:
                if x < 0: q[k] = -q[k]
                break
    return q

def words_to_float_quat(words, t):
    nt = n_t(t); n = math.sqrt(s1f(nt))
    q = np.array([[s1f((w[0], w[1])), s1f((w[2], w[3])), s1f((w[4], w[5])), s1f((w[6], w[7]))] for w in words]).reshape(-1, 4)
    return q / (2 * n)

# ============================================================== validation against the exhaustive enumeration
def _ref_grid_mask(q, eps, Q):
    """float grid test exactly as enum2.py / lowcost_counts.py (nearest grid point from arg W_00), no tie handling"""
    m0 = q[:, 0] + 1j * q[:, 1]
    d0 = np.round(-np.angle(m0) * Q / np.pi)
    re = np.abs(np.real(np.exp(1j * np.pi * d0 / Q) * m0))
    return np.sqrt(np.maximum(0.0, 2 - 2 * re)) <= eps

def cmd_validate(args):
    import pickle
    log = open(os.path.join(DATA, 'torus_tube_validation_log.txt'), 'w')
    def P(*a):
        s = ' '.join(str(x) for x in a); print(s, flush=True); log.write(s + '\n'); log.flush()
    P('torus_tube_enum.py validate', json.dumps(code_version()), 'tmax', args.tmax, 'eps_ref', args.eps_ref)
    rng = np.random.default_rng(20261011)
    eps_list = [2.0 ** -k for k in (3, 4, 5, 6, 7, 8)] + [0.024532, 0.024531, 0.0265, 0.0264991]
    eps_list += sorted(float(x) for x in np.exp(rng.uniform(np.log(2 ** -8), np.log(2 ** -3), 20)))
    def qs_for(e):
        Qe = tuned_Q(e)
        return sorted({Qe, Qe - 1, max(3, 8 * (Qe // 8)), max(3, Qe // 2), 3, 4})
    T0 = time.time()
    REF, MW, FE = {}, {}, {}
    set_ok = True
    for t in range(args.tmax + 1):
        t0 = time.time(); ref = canon_float(exhaustive_tube_level(t, args.eps_ref)); t1 = time.time()
        st = {}; ws = enum_torus_level(t, args.eps_ref, st); t2 = time.time()
        mine = canon_float(words_to_float_quat(ws, t))
        kr = np.round(ref * 1e6).astype(np.int64); km = np.round(mine * 1e6).astype(np.int64)
        ir = np.lexsort(kr.T[::-1]); im = np.lexsort(km.T[::-1])
        same = len(kr) == len(km) and np.array_equal(kr[ir], km[im])
        maxdiff = float(np.max(np.abs(ref[ir] - mine[im]))) if same and len(kr) else 0.0
        uniq = len(set(map(tuple, km))) == len(km)
        set_ok &= same and uniq and maxdiff < 1e-9
        P(f't={t:2d}: exhaustive tube words {len(ref):8d}  (c,d)-enumeration {len(mine):8d}  identical sets: {same}  '
          f'max |diff| {maxdiff:.1e}  keys unique: {uniq}  (c,d) pairs {st.get("pairs", 0)}  times {t1-t0:.1f}s / {t2-t1:.1f}s')
        REF[t] = ref[ir]; MW[t] = np.array([ws[i] for i in im], dtype=np.int64).reshape(-1, 8)
        FE[t] = features(MW[t], t)
    P(f'set comparison: {"PASS" if set_ok else "FAIL"}  ({time.time()-T0:.0f}s)')
    # counts: tube and grid, per shell, reference float method vs exact (c,d) method
    ties = []
    rows = []; cnt_ok = True
    for e in eps_list:
        for Q in [None] + qs_for(e):
            ref_sh, mine_sh = [], []
            for t in range(args.tmax + 1):
                q = REF[t]; ws = MW[t]
                if Q is None:
                    r = int(np.sum(np.abs(q[:, 0] + 1j * q[:, 1]) >= 1 - e ** 2 / 2))
                else:
                    r = int(np.sum(_ref_grid_mask(q, e, Q)))
                if len(ws):
                    s, phi = FE[t]; tv = np.full(len(ws), t)
                    d = tube_dist(s) if Q is None else grid_dist(s, phi, Q)
                    m = within(d, e, ws, tv, 'tube' if Q is None else 'grid', Q, ties)
                    mm = int(m.sum())
                else:
                    mm = 0
                ref_sh.append(r); mine_sh.append(mm)
            ok = ref_sh == mine_sh; cnt_ok &= ok
            rows.append({'eps': e, 'Q': Q, 'ref': ref_sh, 'mine': mine_sh, 'equal': ok})
            if not ok or Q is None or Q == tuned_Q(e):
                P(f'eps={e:.7g} Q={Q if Q else "tube":>4}: equal={ok}  shells(mine)={mine_sh}' + ('' if ok else f'  ref={ref_sh}'))
    P(f'count comparison over {len(rows)} (eps, Q) cells x {args.tmax+1} shells: {"PASS" if cnt_ok else "FAIL"}; exact ties met: {len(ties)}')
    # stored exhaustive data (enum2.py, data/enum_res2.pkl): identity coset, eps = 2^-4..2^-8, tuned Q
    st_ok = True
    if os.path.exists(os.path.join(DATA, 'enum_res2.pkl')):
        res = pickle.load(open(os.path.join(DATA, 'enum_res2.pkl'), 'rb'))
        nchk = 0
        for (t, name, e), (g, tb, tot) in sorted(res.items()):
            if name != 'I,I' or t > args.tmax: continue
            row = next(r for r in rows if r['eps'] == e and r['Q'] == tuned_Q(e))
            rowt = next(r for r in rows if r['eps'] == e and r['Q'] is None)
            ok = row['mine'][t] == g and rowt['mine'][t] == tb; st_ok &= ok; nchk += 1
            if not ok: P(f'  MISMATCH with enum_res2.pkl at t={t} eps={e}: stored (grid,tube)=({g},{tb}), (c,d): ({row["mine"][t]},{rowt["mine"][t]})')
        P(f'comparison with data/enum_res2.pkl (identity coset, {nchk} (t, eps) cells, grid and tube): {"PASS" if st_ok else "FAIL"}')
    # the known step: identity coset, eps = 0.024532, Q = 32, tau = 11
    row = next(r for r in rows if r['eps'] == 0.024532 and r['Q'] == 32)
    N11 = sum(row['mine'][:12]); creq = (N11 - 8 - 22) / (2 ** 11 * 0.024532 ** 2)
    P(f'identity coset, eps=0.024532, Q=Q_eps=32: grid shells {row["mine"][:13]}, cumulative at tau=11: {N11}, '
      f'required c (c0,c1)=(8,2): {creq:.1f}  (paper: 232 words, c >= 163.9)')
    json.dump({'code': code_version(), 'tmax': args.tmax, 'eps_ref': args.eps_ref, 'set_comparison_pass': bool(set_ok),
               'count_comparison_pass': bool(cnt_ok), 'enum_res2_pass': bool(st_ok), 'ties': [list(map(str, x)) for x in ties],
               'cells': rows, 'step_0.024532': {'N11': N11, 'c_req': creq}},
              open(os.path.join(DATA, 'torus_tube_validation.json'), 'w'), indent=0)
    P(f'total {time.time()-T0:.0f}s')

# ============================================================== vectorised features, band enumeration
import numba

def _s1v(x, y):
    """sigma_1(x + y r2) for int64 arrays, without cancellation"""
    x = np.asarray(x, dtype=np.int64); y = np.asarray(y, dtype=np.int64)
    out = x + y * R2
    bad = ((x > 0) & (y < 0)) | ((x < 0) & (y > 0))
    if bad.any():
        xb, yb = x[bad], y[bad]
        out[bad] = (xb * xb - 2 * yb * yb) / (xb - yb * R2)
    return out

def s1_nt(t):
    t = np.asarray(t, dtype=np.float64)
    return np.where(t % 2 == 0, 2.0 ** (t / 2), (2 + R2) * 2.0 ** ((t - 1) / 2))

def features_vec(W, tl):
    """W: (n,8) int64 numerators 2w, tl: levels (nrd(w) = n_tl). s = |W_01|^2, phi = arg W_00"""
    a = _s1v(W[:, 0], W[:, 1]); b = _s1v(W[:, 2], W[:, 3]); c = _s1v(W[:, 4], W[:, 5]); d = _s1v(W[:, 6], W[:, 7])
    return (c * c + d * d) / (4 * s1_nt(tl)), np.arctan2(b, a)

CACHE = os.environ.get('TORUS_TUBE_CACHE', '/tmp/conj1/cache')

def torus_levels(tau_enum, eps_hi, P=print):
    """torus words of every level t <= tau_enum within eps_hi (cached per level as .npy); returns W (n,8), tl (n,)"""
    d = os.path.join(CACHE, 'torus_' + float(eps_hi).hex()); os.makedirs(d, exist_ok=True)
    Ws, ts = [], []; stats = []
    for t in range(tau_enum + 1):
        f = os.path.join(d, f'level_{t:03d}.npy')
        if os.path.exists(f):
            arr = np.load(f); st = {'cached': True}
        else:
            t0 = time.time(); st = {}
            arr = enum_torus_level(t, eps_hi, st, as_tuples=False)
            np.save(f + '.tmp.npy', arr); os.replace(f + '.tmp.npy', f)
            st['sec'] = round(time.time() - t0, 2)
            P(f'  level t={t:3d}: {len(arr):10d} torus words, {st.get("pairs", 0):9d} (c,d) pairs, {st["sec"]:.1f}s')
        st['t'] = t; st['words'] = len(arr); stats.append(st)
        Ws.append(arr); ts.append(np.full(len(arr), t, dtype=np.int64))
    return np.concatenate(Ws), np.concatenate(ts), stats

# ============================================================== exact sup over eps of the required constant
def sup_requirement(tw, d, tau_range, eps_lo, eps_up, c0=8, c1=2):
    """For each tau: sup over eps in (eps_lo, eps_up] of (N(tau,eps) - c0 - c1 tau)/(2^tau eps^2), N(tau,eps) =
    #{t <= tau, d <= eps}.  N is a right-continuous step function, so the sup is attained at a jump point d_k or
    approached at eps_lo.  Returns per tau the best value overall and in the regimes x = 2^tau eps^2 < 1/8,
    [1/8, 1), >= 1, each as (c_req, eps, N)."""
    out = {}
    if eps_up <= eps_lo: return out
    m = d <= eps_up
    dd = d[m]; tt = tw[m]
    o = np.argsort(dd, kind='stable'); dd = dd[o]; tt = tt[o]
    for tau in tau_range:
        ok = tt <= tau
        dt = dd[ok]
        base = int(np.searchsorted(dt, eps_lo, side='right'))
        eps_c = np.concatenate([[eps_lo], dt[base:]])
        N_c = np.arange(base, len(dt) + 1)
        x = 2.0 ** tau * eps_c ** 2
        f = (N_c - c0 - c1 * tau) / x
        rec = {}
        for name, mask in (('all', None), ('x<1/8', x < 0.125), ('1/8<=x<1', (x >= 0.125) & (x < 1)), ('x>=1', x >= 1)):
            if mask is None:
                k = int(np.argmax(f))
            elif mask.any():
                k = int(np.argmax(np.where(mask, f, -np.inf)))
            else:
                rec[name] = None; continue
            rec[name] = (float(f[k]), float(eps_c[k]), int(N_c[k]))
        out[tau] = rec
    return out

@numba.njit(parallel=True, cache=True)
def _grid_sup_kernel(Qs, lefts, rights, tlo, thi, ts, ss, phis, dtube, tau_lo, tau_hi, c0, c1, res, rtau):
    """per modulus Q: sup over eps in (left, right] (category 0), over the tuned sub-interval (tlo, thi] (category 1,
    sup includes the limit eps -> tlo+), and over post-onset eps (2^tau eps^2 >= 1, category 2), of
    (N_Q(tau, eps) - c0 - c1 tau)/(2^tau eps^2), tau in [tau_lo, tau_hi].  Words sorted by dtube.
    res[qi, cat, :] = (c_req, tau, eps, N, x)."""
    n = len(ss)
    for qi in numba.prange(len(Qs)):
        Q = Qs[qi]; left = lefts[qi]; right = rights[qi]; tl_ = tlo[qi]; th_ = thi[qi]
        for cat in range(3):
            res[qi, cat, 0] = -1e300
        base = np.zeros(tau_hi + 1, np.int64)
        cnt = 0
        for k in range(n):
            if dtube[k] > right: break
            if ts[k] > tau_hi: continue
            x = phis[k] * Q / np.pi; delta = (x - np.round(x)) * np.pi / Q
            sq = np.sqrt(1.0 - ss[k]); dist = np.sqrt(2.0 * ss[k] / (1.0 + sq) + 4.0 * sq * np.sin(delta / 2) ** 2)
            if dist <= left: base[ts[k]] += 1
            elif dist <= right: cnt += 1
        bd = np.empty(cnt); bt = np.empty(cnt, np.int64); j = 0
        for k in range(n):
            if dtube[k] > right: break
            if ts[k] > tau_hi: continue
            x = phis[k] * Q / np.pi; delta = (x - np.round(x)) * np.pi / Q
            sq = np.sqrt(1.0 - ss[k]); dist = np.sqrt(2.0 * ss[k] / (1.0 + sq) + 4.0 * sq * np.sin(delta / 2) ** 2)
            if dist > left and dist <= right:
                bd[j] = dist; bt[j] = ts[k]; j += 1
        o = np.argsort(bd); bd = bd[o]; bt = bt[o]
        cum = 0
        for tau in range(0, tau_hi + 1):
            cum += base[tau]
            if tau < tau_lo: continue
            p2 = 2.0 ** tau; a = c0 + c1 * tau
            # left end (limit)
            xx = p2 * left * left; f = (cum - a) / xx
            rtau[qi, tau - tau_lo] = f
            if f > res[qi, 0, 0]:
                res[qi, 0, 0] = f; res[qi, 0, 1] = tau; res[qi, 0, 2] = left; res[qi, 0, 3] = cum; res[qi, 0, 4] = xx
            if xx >= 1.0 and f > res[qi, 2, 0]:
                res[qi, 2, 0] = f; res[qi, 2, 1] = tau; res[qi, 2, 2] = left; res[qi, 2, 3] = cum; res[qi, 2, 4] = xx
            run = cum; tdone = False
            for jj in range(cnt):
                if (not tdone) and bd[jj] > tl_:
                    if tl_ >= left and tl_ < th_:
                        xx = p2 * tl_ * tl_; f = (run - a) / xx
                        if f > res[qi, 1, 0]:
                            res[qi, 1, 0] = f; res[qi, 1, 1] = tau; res[qi, 1, 2] = tl_; res[qi, 1, 3] = run; res[qi, 1, 4] = xx
                    tdone = True
                if bt[jj] <= tau:
                    run += 1
                    xx = p2 * bd[jj] * bd[jj]; f = (run - a) / xx
                    if f > rtau[qi, tau - tau_lo]: rtau[qi, tau - tau_lo] = f
                    if f > res[qi, 0, 0]:
                        res[qi, 0, 0] = f; res[qi, 0, 1] = tau; res[qi, 0, 2] = bd[jj]; res[qi, 0, 3] = run; res[qi, 0, 4] = xx
                    if xx >= 1.0 and f > res[qi, 2, 0]:
                        res[qi, 2, 0] = f; res[qi, 2, 1] = tau; res[qi, 2, 2] = bd[jj]; res[qi, 2, 3] = run; res[qi, 2, 4] = xx
                    if bd[jj] > tl_ and bd[jj] <= th_ and f > res[qi, 1, 0]:
                        res[qi, 1, 0] = f; res[qi, 1, 1] = tau; res[qi, 1, 2] = bd[jj]; res[qi, 1, 3] = run; res[qi, 1, 4] = xx
            if (not tdone) and tl_ >= left and tl_ < th_ and tl_ < right:
                xx = p2 * tl_ * tl_; f = (run - a) / xx
                if f > res[qi, 1, 0]:
                    res[qi, 1, 0] = f; res[qi, 1, 1] = tau; res[qi, 1, 2] = tl_; res[qi, 1, 3] = run; res[qi, 1, 4] = xx

def grid_sup_allQ(tw, s, phi, Qs, eps_lo, eps_hi, tau_lo, tau_hi, c0=8, c1=2, chunk=1 << 20, ntop=25):
    """exact sup over eps in the band of the grid requirement for each modulus Q in Qs (Q <= Q_eps imposed),
    processed in chunks of moduli.  Returns (best[cat] = (c_req, tau, eps, N, x, Q) for cat = all/tuned/post-onset,
    top = ntop moduli by c_req (all) as rows (Q, c_req, tau, eps, N, tuned c_req), per-tau max (c_req, Q),
    tuned c_req of every modulus (float32))"""
    keep = tw <= tau_hi                                  # words above tau_hi play no role
    tw, s, phi = tw[keep], s[keep], phi[keep]
    o = np.argsort(s, kind='stable')
    ts, ss, ph = tw[o].astype(np.int64), s[o].astype(np.float64), phi[o].astype(np.float64)
    dt = tube_dist(ss)
    Qs = np.asarray(Qs, dtype=np.int64)
    best = np.full((3, 6), -np.inf); top = np.zeros((0, 6)); ptau = np.full((tau_hi - tau_lo + 1, 2), -np.inf)
    tuned_all = np.empty(len(Qs), dtype=np.float32)
    for a in range(0, len(Qs), chunk):
        Qc = Qs[a:a + chunk]
        emax = np.sin(np.pi / (2 * Qc)) / 2
        rights = np.minimum(eps_hi, emax); lefts = np.full(len(Qc), eps_lo)
        tlo = np.sin(np.pi / (2 * (Qc + 1))) / 2; thi = emax
        res = np.zeros((len(Qc), 3, 5)); rtau = np.full((len(Qc), tau_hi - tau_lo + 1), -1e300)
        _grid_sup_kernel(Qc, lefts, rights, tlo, thi, ts, ss, ph, dt, tau_lo, tau_hi, float(c0), float(c1), res, rtau)
        for cat in range(3):
            k = int(np.argmax(res[:, cat, 0]))
            if res[k, cat, 0] > best[cat, 0]: best[cat] = list(res[k, cat]) + [Qc[k]]
        kk = np.argsort(-res[:, 0, 0])[:ntop]
        rows = np.stack([Qc[kk], res[kk, 0, 0], res[kk, 0, 1], res[kk, 0, 2], res[kk, 0, 3], res[kk, 1, 0]], 1)
        top = np.concatenate([top, rows]); top = top[np.argsort(-top[:, 1])[:ntop]]
        j = np.argmax(rtau, axis=0); v = rtau[j, np.arange(rtau.shape[1])]
        upd = v > ptau[:, 0]; ptau[upd, 0] = v[upd]; ptau[upd, 1] = Qc[j[upd]]
        tuned_all[a:a + chunk] = res[:, 1, 0]
    return best, top, ptau, tuned_all

def shells_at(tw, d, eps, tau_max):
    return np.bincount(tw[(d <= eps) & (tw <= tau_max)], minlength=tau_max + 1)[:tau_max + 1].tolist()

def haar_grid_prob(eps, Q):
    psi = 2 * math.asin(eps / 2)
    return Q * (2 * psi - math.sin(2 * psi)) / math.pi

def haar_tube_prob(eps):
    return eps ** 2 - eps ** 4 / 4

def n_words_shell(t):
    return 24 if t == 0 else 36 * 2 ** t

# ============================================================== arithmetic frames: W = G1 V G2
def _zmul_v(a0, a1, b0, b1):
    return a0 * b0 + 2 * a1 * b1, a0 * b1 + a1 * b0

def qmul2_vec(P, Qm):
    """vectorised qmul2: P, Qm (n,8) int64 (2p, 2q) -> 2(pq); asserts integrality"""
    A1, B1, C1, D1 = [(P[:, 2 * i], P[:, 2 * i + 1]) for i in range(4)]
    A2, B2, C2, D2 = [(Qm[:, 2 * i], Qm[:, 2 * i + 1]) for i in range(4)]
    m = lambda x, y: _zmul_v(x[0], x[1], y[0], y[1])
    def comb(*terms):
        r0 = 0; r1 = 0
        for sgn, (x, y) in terms:
            u = m(x, y); r0 = r0 + sgn * u[0]; r1 = r1 + sgn * u[1]
        return r0, r1
    A = comb((1, (A1, A2)), (-1, (B1, B2)), (-1, (C1, C2)), (-1, (D1, D2)))
    B = comb((1, (A1, B2)), (1, (B1, A2)), (1, (C1, D2)), (-1, (D1, C2)))
    C = comb((1, (A1, C2)), (1, (C1, A2)), (-1, (B1, D2)), (1, (D1, B2)))
    D = comb((1, (A1, D2)), (1, (D1, A2)), (1, (B1, C2)), (-1, (C1, B2)))
    R = np.stack([A[0], A[1], B[0], B[1], C[0], C[1], D[0], D[1]], 1)
    assert not np.any(R % 2), 'product not integral'
    return R // 2

def reduce_vec(R, tv):
    """strip powers of r2 (vectorised div_r2): returns (primitive numerators, minimal T-counts)"""
    R = R.copy(); tv = tv.copy()
    while True:
        x0 = R[:, 0::2]; x1 = R[:, 1::2]
        cand = np.all(x0 % 2 == 0, axis=1)
        y0 = x1; y1 = x0 // 2                               # (x0 + x1 r2)/r2 = x1 + (x0/2) r2
        inO = (np.all((y0 - y0[:, :1]) % 2 == 0, axis=1)) & (y1.sum(axis=1) % 2 == 0)
        dv = cand & inO
        if not dv.any(): return R, tv
        Rn = np.empty_like(R[dv]); Rn[:, 0::2] = y0[dv]; Rn[:, 1::2] = y1[dv]
        R[dv] = Rn; tv[dv] -= 2

def frame_words(Wv, tv, g1, t1, g2, t2):
    """W = G1 V G2 for torus words V (levels tv): numerators and exact minimal T-counts of W"""
    n = len(Wv)
    G1 = np.tile(np.array([x for pair in g1 for x in pair], dtype=np.int64), (n, 1))
    G2 = np.tile(np.array([x for pair in g2 for x in pair], dtype=np.int64), (n, 1))
    R = qmul2_vec(qmul2_vec(G1, Wv), G2)
    return reduce_vec(R, tv + t1 + t2)

def clifford_strings():
    """H/S strings of the 24 single-qubit Cliffords modulo phase (BFS on exact numerators)"""
    seen = {canon_sign(word_quat('')[0]): ''}; fr = ['']
    while fr:
        nf = []
        for s in fr:
            for g in 'HS':
                Q, _ = word_quat(s + g); k = canon_sign(Q)
                if k not in seen: seen[k] = s + g; nf.append(s + g)
        fr = nf
    assert len(seen) == 24
    return sorted(seen.values(), key=lambda x: (len(x), x))

def random_ma_word(t, rng):
    """random Matsumoto-Amano word (T|e)(HT|SHT)^* C of T-count exactly t, as a gate string"""
    cl = clifford_strings()
    if t == 0: return cl[rng.integers(24)]
    lead = bool(rng.integers(2)); nsyl = t - 1 if lead else t
    s = 'T' if lead else ''
    for _ in range(nsyl): s += 'HT' if rng.integers(2) == 0 else 'SHT'
    return s + cl[rng.integers(24)]

def inverse_string(s):
    """gate string of the inverse: reverse, T^-1 = T^7 = T S^3 (mod phase), S^-1 = S^3, H^-1 = H"""
    out = ''
    for g in reversed(s):
        out += {'H': 'H', 'S': 'SSS', 'T': 'TSSS'}[g]
    return out

# ============================================================== one accuracy band: enumeration + analysis
def _dec_above(x, sig=7):
    """the smallest decimal with `sig` significant digits that is >= x"""
    e = math.floor(math.log10(x)) - sig + 1
    return math.ceil(x / 10 ** e - 1e-9) * 10 ** e

def cmd_band(args):
    L, K = args.L, args.K
    eps_hi = 2.0 ** -L; eps_lo = 2.0 ** -(L + 1); tau_max = 2 * L + K
    name = args.name
    logf = open(os.path.join(DATA, f'torus_tube_{name}_log.txt'), 'a')
    def P(*a):
        s = ' '.join(str(x) for x in a); print(s, flush=True); logf.write(s + '\n'); logf.flush()
    T0 = time.time()
    P(f'==== {time.strftime("%Y-%m-%d %H:%M:%S")} band L={L}: eps in ({eps_lo}, {eps_hi}], tau <= {tau_max}, frame {name}: '
      f'{args.G1!r},{args.G2!r}  cmd: {" ".join(sys.argv)}  code {json.dumps(code_version())}')
    # ---- frame
    g1, t1 = reduce_numerator(*word_quat(args.G1)); g2, t2 = reduce_numerator(*word_quat(args.G2))
    tau_enum = tau_max + t1 + t2
    Wv, tv, stats = torus_levels(tau_enum, eps_hi, P)
    if t1 == 0 and t2 == 0:
        tw = tv.copy()
    else:
        _, tw = frame_words(Wv, tv, g1, t1, g2, t2)
    keep = tw <= tau_max
    Wv, tv, tw = Wv[keep], tv[keep], tw[keep]
    s, phi = features_vec(Wv, tv)
    dtb = tube_dist(s)
    T_enum = time.time() - T0
    P(f'enumeration: torus levels <= {tau_enum}, {int(keep.sum())} words of T-count <= {tau_max} within {eps_hi} of the coset, {T_enum:.0f}s')
    out = {'code': code_version(), 'cmd': sys.argv, 'frame': {'name': name, 'G1': args.G1, 'G2': args.G2, 't1': t1, 't2': t2},
           'L': L, 'eps_band_left_open': [eps_lo, eps_hi], 'tau_max': tau_max, 'tau_enum_torus': tau_enum, 'c0': 8, 'c1': 2,
           'enumeration': {'levels': stats, 'n_words_kept': int(keep.sum()), 'seconds': round(T_enum, 1)},
           'tube_shells_at_eps_hi': np.bincount(tw, minlength=tau_max + 1).tolist()}
    # identity/Clifford frames: for tau <= 2L-8 only the eight T-powers are within eps <= 2^-L (Lemma identity and
    # k <= tau/2 + 3), so N <= c0 there; the grid scans start at 2L-10.  Arithmetic frames start at 0.
    tau_lo = max(0, 2 * L - 10) if (t1 == 0 and t2 == 0) else 0
    # ---- tube (upper bound for every Q)
    sup = sup_requirement(tw, dtb, range(0, tau_max + 1), eps_lo, eps_hi)
    bt = max(sup, key=lambda k: sup[k]['all'][0])
    out['tube'] = {'per_tau': {str(k): v for k, v in sup.items()}, 'max': {'tau': bt, 'c_req': sup[bt]['all'][0],
                   'eps': sup[bt]['all'][1], 'N': sup[bt]['all'][2]}}
    P(f'tube: max required c over the band = {sup[bt]["all"][0]:.1f} at tau={bt}, eps={sup[bt]["all"][1]:.8f}, N={sup[bt]["all"][2]}')
    # ---- grid, every Q <= Q_eps, exact sup over eps in the band, tau <= tau_allq
    Qmax = tuned_Q(eps_lo * (1 + 1e-12)); Qs = np.arange(3, Qmax + 1)
    Kall = args.Kall
    if Kall < 0:                     # budget: (#words with T-count <= 2L+Kall) x (#moduli) <~ 4e10
        Kall = 6
        while Kall < K and np.sum(tw <= 2 * L + Kall + 1) * len(Qs) < 4e10: Kall += 1
    tau_allq = min(tau_max, 2 * L + Kall)
    t0 = time.time()
    best, top, ptau, tq = grid_sup_allQ(tw, s, phi, Qs, eps_lo, eps_hi, tau_lo, tau_allq)
    T_allq = time.time() - t0
    g = {'Q_range': [3, int(Qmax)], 'tau_range': [tau_lo, tau_allq], 'seconds': round(T_allq, 1)}
    for cat, nm in enumerate(['all', 'tuned', 'post_onset']):
        r = best[cat]
        g['max_' + nm] = {'Q': int(r[5]), 'c_req': float(r[0]), 'tau': int(r[1]), 'eps': float(r[2]), 'N': int(r[3]), 'x': float(r[4])}
    g['top'] = [{'Q': int(r[0]), 'c_req': float(r[1]), 'tau': int(r[2]), 'eps': float(r[3]), 'N': int(r[4]),
                 'tuned_c_req': float(r[5])} for r in top]
    g['per_tau_max'] = {str(tau_lo + j): {'c_req': float(ptau[j, 0]), 'Q': int(ptau[j, 1])} for j in range(len(ptau))}
    # tuned Q only: for each Q its tuned interval; per-band maximum and median over Q
    okq = tq > -1e299
    g['tuned_summary'] = {'n_Q_with_tuned_interval_in_band': int(okq.sum()),
                          'median_c_req': float(np.median(tq[okq])) if okq.any() else None,
                          'q90_c_req': float(np.quantile(tq[okq], 0.9)) if okq.any() else None}
    out['grid_all_Q'] = g
    P(f'grid, all {len(Qs)} moduli 3..{Qmax}, tau {tau_lo}..{tau_allq}: max required c = {g["max_all"]["c_req"]:.1f} '
      f'(Q={g["max_all"]["Q"]}, tau={g["max_all"]["tau"]}, eps={g["max_all"]["eps"]:.8f}, N={g["max_all"]["N"]}, '
      f'2^tau eps^2={g["max_all"]["x"]:.3f}); tuned Q only: {g["max_tuned"]["c_req"]:.1f} (Q={g["max_tuned"]["Q"]}); '
      f'post-onset: {g["max_post_onset"]["c_req"]:.1f}; {T_allq:.0f}s')
    # ---- grid, selected moduli, all tau <= tau_max
    sel = set(int(r[0]) for r in top[:10])
    for e in (eps_hi, 2.0 ** -(L + 0.5)): sel.add(tuned_Q(e))
    j = 8
    while j <= Qmax: sel.add(j); j *= 2
    fixed = {}
    for Q in sorted(sel):
        eu = min(eps_hi, eps_max_of_Q(Q))
        d = grid_dist(s, phi, Q)
        sq = sup_requirement(tw, d, range(tau_lo, tau_max + 1), eps_lo, eu)
        if not sq: continue
        b = max(sq, key=lambda k: sq[k]['all'][0])
        fixed[str(Q)] = {'per_tau': {str(k): v['all'] for k, v in sq.items()},
                         'max': {'tau': b, 'c_req': sq[b]['all'][0], 'eps': sq[b]['all'][1], 'N': sq[b]['all'][2]}}
    out['grid_fixed_Q'] = fixed
    bq = max(fixed, key=lambda q: fixed[q]['max']['c_req'])
    P(f'grid, selected moduli {sorted(int(q) for q in fixed)}, tau <= {tau_max}: max {fixed[bq]["max"]["c_req"]:.1f} at Q={bq}, '
      f'tau={fixed[bq]["max"]["tau"]}')
    # ---- worst cell, re-counted with exact tie handling at a decimal eps just above the jump
    wc = g['max_all']; Q, tau, ew = wc['Q'], wc['tau'], wc['eps']
    d = grid_dist(s, phi, Q)
    dec = _dec_above(ew, 7)
    nxt = np.min(d[(d > ew * (1 + 1e-9)) & (tw <= tau)]) if np.any((d > ew * (1 + 1e-9)) & (tw <= tau)) else np.inf
    ties = []
    msk = within(d, dec, Wv, tv, 'grid', Q, ties) & (tw <= tau)
    sh = np.bincount(tw[msk], minlength=tau + 1).tolist()
    Ndec = int(msk.sum())
    pg = haar_grid_prob(dec, Q)
    out['worst_cell'] = {'Q': Q, 'tau': tau, 'eps_jump': ew, 'eps_decimal': dec, 'decimal_below_next_jump': bool(dec < nxt),
                         'Q_eps_decimal': tuned_Q(dec), 'N': Ndec, 'shells': sh,
                         'c_req_decimal': (Ndec - 8 - 2 * tau) / (2 ** tau * dec ** 2),
                         'haar_shells': [n_words_shell(t) * pg for t in range(tau + 1)],
                         'distinct_distances': sorted(set(np.round(d[msk & (tw >= 2)], 9).tolist()))[:20]}
    P(f'worst cell: Q={Q} (Q_eps={tuned_Q(dec)}), eps={dec:.7g} (jump at {ew:.10f}), tau={tau}: N={Ndec}, shells {sh}, '
      f'required c = {(Ndec - 8 - 2 * tau) / (2 ** tau * dec ** 2):.1f}')
    # ---- dyadic reference cell eps = 2^-L, tuned Q: shells and Haar ratios
    e = eps_hi; Q = tuned_Q(e)
    mg = within(grid_dist(s, phi, Q), e, Wv, tv, 'grid', Q, ties); mt = within(dtb, e, Wv, tv, 'tube', None, ties)
    gs = np.bincount(tw[mg], minlength=tau_max + 1).tolist(); ts_ = np.bincount(tw[mt], minlength=tau_max + 1).tolist()
    pg, pt = haar_grid_prob(e, Q), haar_tube_prob(e)
    hg = [n_words_shell(t) * pg for t in range(tau_max + 1)]; ht = [n_words_shell(t) * pt for t in range(tau_max + 1)]
    cg = np.cumsum(gs); chg = np.cumsum(hg)
    out['dyadic_cell'] = {'eps': e, 'Q': Q, 'grid_shells': gs, 'tube_shells': ts_, 'haar_grid_shells': hg, 'haar_tube_shells': ht,
                          'grid_shell_ratio': [gs[t] / hg[t] for t in range(tau_max + 1)],
                          'tube_shell_ratio': [ts_[t] / ht[t] for t in range(tau_max + 1)],
                          'grid_cum_ratio': (cg / chg).tolist(),
                          'c_req_per_tau': [(cg[t] - 8 - 2 * t) / (2 ** t * e ** 2) for t in range(tau_max + 1)]}
    out['ties'] = [[str(x) for x in tt] for tt in ties]
    out['seconds_total'] = round(time.time() - T0, 1)
    fn = os.path.join(DATA, f'torus_tube_{name}_L{L}.json')
    json.dump(out, open(fn, 'w'), indent=0)
    P(f'dyadic cell eps=2^-{L}, Q={Q}: grid shells {gs}; grid shell/Haar at t=tau_max: {gs[-1] / hg[-1]:.3f}; '
      f'tube shell/Haar at t=tau_max: {ts_[-1] / ht[-1]:.3f}')
    P(f'wrote {fn}  (total {time.time() - T0:.0f}s)')

# ============================================================== validation of the frame change of variables
def _mat_of_string(s):
    G = {'H': _H, 'S': _S, 'T': _T}; M = np.eye(2, dtype=complex)
    for g in s: M = M @ G[g]
    return M / np.sqrt(np.linalg.det(M))

def exhaustive_frame_level(t, G1, G2):
    """float m00 (SU(2)-normalised) of G1^dag W G2^dag for every word W of T-count exactly t (MA normal form)"""
    CL = _cliffords(); HT, SHT = _H @ _T, _S @ _H @ _T
    if t == 0:
        P = np.eye(2, dtype=complex)[None]
    else:
        P = np.array([_T, HT, SHT])
        for _ in range(t - 1): P = np.concatenate([P @ HT, P @ SHT])
    P = P / np.sqrt(np.linalg.det(P))[:, None, None]
    v = np.einsum('j,njk->nk', G1.conj().T[0], P)                  # row 0 of G1^dag P
    out = []
    for C in CL:
        C = C / np.sqrt(np.linalg.det(C))
        u = C @ G2.conj().T[:, 0]
        out.append(v @ u)
    return np.concatenate(out)

def cmd_validate_frames(args):
    log = open(os.path.join(DATA, 'torus_tube_validation_frames_log.txt'), 'w')
    def P(*a):
        s = ' '.join(str(x) for x in a); print(s, flush=True); log.write(s + '\n'); log.flush()
    P('torus_tube_enum.py validate-frames', json.dumps(code_version()), 'tmax', args.tmax)
    rng = np.random.default_rng(20261012)
    frames = []
    for tg in (1, 2, 3):
        frames.append((f'random pair, T-count {tg}', random_ma_word(tg, rng), random_ma_word(tg, rng)))
    g = random_ma_word(3, rng); frames.append(('axis frame (G, G^-1), T-count 3', g, inverse_string(g)))
    eps_list = [2.0 ** -3, 2.0 ** -4, 0.0437, 2.0 ** -5, 0.0265]
    allok = True; cells = []
    for nm, s1, s2 in frames:
        G1, G2 = _mat_of_string(s1), _mat_of_string(s2)
        g1, t1 = reduce_numerator(*word_quat(s1)); g2, t2 = reduce_numerator(*word_quat(s2))
        Wv, tv, _ = torus_levels(args.tmax + t1 + t2, max(eps_list), lambda *a: None)
        _, tw = frame_words(Wv, tv, g1, t1, g2, t2)
        k = tw <= args.tmax; Wv, tv, tw = Wv[k], tv[k], tw[k]
        s, phi = features_vec(Wv, tv)
        REF = [exhaustive_frame_level(t, G1, G2) for t in range(args.tmax + 1)]
        for e in eps_list:
            for Q in [None] + sorted({q for q in (tuned_Q(e), 8 * (tuned_Q(e) // 8)) if q >= 3}):
                if Q is None:
                    ref = [int(np.sum(np.abs(m) >= 1 - e ** 2 / 2)) for m in REF]
                    msk = within(tube_dist(s), e, Wv, tv, 'tube')
                else:
                    ref = []
                    for m in REF:
                        d0 = np.round(-np.angle(m) * Q / np.pi)
                        re = np.abs(np.real(np.exp(1j * np.pi * d0 / Q) * m))
                        ref.append(int(np.sum(np.sqrt(np.maximum(0, 2 - 2 * re)) <= e)))
                    msk = within(grid_dist(s, phi, Q), e, Wv, tv, 'grid', Q)
                mine = np.bincount(tw[msk], minlength=args.tmax + 1).tolist()
                ok = mine == ref; allok &= ok
                cells.append({'frame': nm, 'G1': s1, 'G2': s2, 'eps': e, 'Q': Q, 'ref': ref, 'mine': mine, 'equal': ok})
                P(f'{nm:34s} eps={e:<8.6g} Q={str(Q) if Q else "tube":>4}: equal={ok} shells={mine}' + ('' if ok else f' ref={ref}'))
    P(f'frame validation over {len(cells)} cells, T-count <= {args.tmax}: {"PASS" if allok else "FAIL"}')
    json.dump({'code': code_version(), 'tmax': args.tmax, 'pass': bool(allok), 'cells': cells},
              open(os.path.join(DATA, 'torus_tube_validation_frames.json'), 'w'), indent=0)

# ============================================================== brute-force recount of one cell
def cmd_verify_cell(args):
    """recount #{W: t_min(W) <= tau, exists d: d_proj(W, G1 R_z(2 pi d/Q) G2) <= eps} by exhaustive Matsumoto-Amano
    enumeration in floating point (independent of the (c,d) parametrisation); words whose float distance is within
    1e-9 (relative) of eps are reported, so that a tie would be visible."""
    e, Q, tau = args.eps, args.Q, args.tau
    assert Q <= tuned_Q(e), (Q, tuned_Q(e))
    G1, G2 = _mat_of_string(args.G1), _mat_of_string(args.G2)
    sh = []; near = 0; t0 = time.time()
    for t in range(tau + 1):
        m = exhaustive_frame_level(t, G1, G2)
        d0 = np.round(-np.angle(m) * Q / np.pi)
        re = np.abs(np.real(np.exp(1j * np.pi * d0 / Q) * m))
        d = np.sqrt(np.maximum(0, 2 - 2 * re))
        sh.append(int(np.sum(d <= e))); near += int(np.sum(np.abs(d - e) <= 1e-9 * e))
    N = sum(sh); c = (N - 8 - 2 * tau) / (2 ** tau * e ** 2)
    msg = (f'verify-cell (exhaustive MA enumeration, float): frame ({args.G1!r},{args.G2!r}) Q={Q} (Q_eps={tuned_Q(e)}) '
           f'eps={e!r} tau={tau}: shells {sh}, N={N}, words within 1e-9*eps of eps: {near}, '
           f'required c for (c0,c1)=(8,2): {c:.2f}  [{time.time() - t0:.0f}s]')
    print(msg, flush=True)
    with open(os.path.join(DATA, 'torus_tube_verify_cells_log.txt'), 'a') as f:
        f.write(time.strftime('%Y-%m-%d %H:%M:%S ') + json.dumps(code_version()) + '\n' + msg + '\n')

# ============================================================== certificate for a lower bound: explicit gate strings
def synthesize(Q2, t):
    """gate string of the unitary with primitive numerator Q2 (2w) and T-count t, by descent in the tree:
    w = C T w' with t_min(w') = t - 1 (C one of the 24 Cliffords); returns the string, whose T-count is t"""
    cl = clifford_strings()
    steps = [(c, word_quat(inverse_string(c + 'T'))[0]) for c in cl]
    s = ''; w = Q2; tt = t
    while tt > 0:
        for c, qi in steps:
            r = qmul2(qi, w)
            if r is None: continue
            r2, t2 = reduce_numerator(r, tt + 1)
            if t2 == tt - 1:
                s += c + 'T'; w = r2; tt = t2; break
        else:
            raise RuntimeError('descent failed')
    def unit_vec(q):                                # projective comparison (numerators differ by units of Z[r2])
        v = np.array([s1f(x) for x in q]); v = v / np.linalg.norm(v)
        k = next(j for j in range(4) if abs(v[j]) > 1e-9)
        return v if v[k] > 0 else -v
    vw = unit_vec(w)
    for c in cl:                                    # w is now a Clifford numerator
        if np.max(np.abs(unit_vec(word_quat(c)[0]) - vw)) < 1e-9: return s + c
    raise RuntimeError('no Clifford')

def cmd_verify_words(args):
    """certificate for N(tau, eps) >= N at the identity coset: every counted word gets an explicit gate string with
    T-count <= tau; its matrix is recomputed from the gates in mpmath, its grid distance re-evaluated, and the
    words are checked to be pairwise distinct modulo phase."""
    L, Q, e, tau = args.L, args.Q, args.eps, args.tau
    assert Q <= tuned_Q(e)
    Wv, tv, _ = torus_levels(tau, 2.0 ** -L, lambda *a: None)
    s, phi = features_vec(Wv, tv)
    m = within(grid_dist(s, phi, Q), e, Wv, tv, 'grid', Q)
    idx = np.nonzero(m)[0]
    G = {'H': mp.matrix([[1, 1], [1, -1]]) / mp.sqrt(2), 'S': mp.matrix([[1, 0], [0, 1j]]),
         'T': mp.matrix([[1, 0], [0, mp.expjpi(mp.mpf(1) / 4)]])}
    keys = set(); ok = 0; maxd = mp.mpf(0); t0 = time.time(); Tc = []
    for i in idx:
        w = tuple((int(Wv[i, 2 * k]), int(Wv[i, 2 * k + 1])) for k in range(4))
        seq = synthesize(w, int(tv[i]))
        M = mp.eye(2)
        for g in seq: M = M * G[g]
        M = M / mp.sqrt(M[0, 0] * M[1, 1] - M[0, 1] * M[1, 0])
        a = M[0, 0]
        d0 = int(mp.nint(-mp.arg(a) * Q / mp.pi)) % Q
        dist = mp.sqrt(max(mp.mpf(0), 2 - 2 * abs(mp.re(mp.expjpi(mp.mpf(d0) / Q) * a))))
        nT = seq.count('T')
        v = np.array([float(mp.re(M[0, 0])), float(mp.im(M[0, 0])), float(mp.re(M[0, 1])), float(mp.im(M[0, 1]))])
        k = next(j for j in range(4) if abs(v[j]) > 1e-9)
        key = tuple(np.round(v if v[k] > 0 else -v, 10))     # SU(2) modulo +-1
        keys.add(key); Tc.append(nT)
        if nT <= tau and dist <= mp.mpf(e): ok += 1
        maxd = max(maxd, dist)
    N = len(idx); c = (ok - 8 - 2 * tau) / (2 ** tau * e ** 2)
    msg = (f'verify-words (identity coset) Q={Q} (Q_eps={tuned_Q(e)}) eps={e!r} tau={tau}: {N} words counted; '
           f'{ok} have an explicit gate string with T-count <= tau and mpmath grid distance <= eps (max {mp.nstr(maxd, 12)}); '
           f'{len(keys)} distinct modulo phase; T-counts of the strings: {dict(sorted(collections.Counter(Tc).items()))}; '
           f'certified lower bound N >= {min(ok, len(keys))}, required c >= {(min(ok, len(keys)) - 8 - 2 * tau) / (2 ** tau * e ** 2):.2f}  [{time.time() - t0:.0f}s]')
    print(msg, flush=True)
    with open(os.path.join(DATA, 'torus_tube_verify_cells_log.txt'), 'a') as fh:
        fh.write(time.strftime('%Y-%m-%d %H:%M:%S ') + json.dumps(code_version()) + '\n' + msg + '\n')

# ============================================================== the chain t_k = 3 2^k - 1 (targeted step computation)
def class_words(t, xi):
    """all torus words of minimal T-count t whose transverse part has C^2 + D^2 = xi (one of each +-w)"""
    nt = n_t(t); M = zsub((4 * nt[0], 4 * nt[1]), xi)
    CD = []
    for (y0, y1, y2, y3) in relnorm_solutions(xi):          # C + D i with (C + D i)(C - D i) = xi
        if (y1 - y3) & 1: continue
        C = (y0, (y1 - y3) // 2); D = (y2, (y1 + y3) // 2)
        if (C[0] - D[0]) & 1: continue
        CD.append((C, D))
    sols_all = relnorm_solutions(M); cache = {}; out = []
    for C, D in CD:
        key = (C[0] & 1, (C[1] + D[1]) & 1)
        if key not in cache:
            cache[key] = [((x0, (x1 - x3) // 2), (x2, (x1 + x3) // 2)) for (x0, x1, x2, x3) in sols_all
                          if not ((x1 - x3) & 1 or (x0 - key[0]) & 1 or (x2 - key[0]) & 1 or (x1 - key[1]) & 1)]
        for A, B in cache[key]:
            Q = (A, B, C, D)
            if is_primitive(Q) and canon_sign(Q) is Q: out.append(Q)
    return out, M

def cmd_chain_step(args):
    """lower bounds for the required c from the two minimal-transverse classes at t = 3 2^k - 2 (C^2+D^2 = 2u^-2^k)
    and t = 3 2^k - 1 (C^2+D^2 = 2 r2 u^-(2^k-1)), u = 1 + r2; angles in 60-digit mpmath, grid offsets in exact
    200-bit fixed point; T-counts by theory (primitive, nrd = n_t), a random sample re-derived as gate strings."""
    k = args.k; m = args.m if args.m else 2 ** k; mp.mp.dps = 80
    logf = open(os.path.join(DATA, 'torus_tube_chain_log.txt'), 'a')
    def P(*a):
        s = ' '.join(str(x) for x in a); print(s, flush=True); logf.write(s + '\n'); logf.flush()
    P(f'==== {time.strftime("%Y-%m-%d %H:%M:%S")} chain-step m={m}  code {json.dumps(code_version())}')
    t1, t2 = 3 * m - 2, 3 * m - 1
    ubar = (-1, 1)                                                     # (1 + r2)^-1 = r2 - 1
    assert m % 2 == 0, 'for odd m the two elements are not totally positive'
    xi1 = zmul((2, 0), zpow(ubar, m)); xi2 = zmul((0, 2), zpow(ubar, m - 1))
    T0 = time.time()
    classes = []
    r2 = mp.sqrt(2); FIX = 200
    for t, xi in ((t1, xi1), (t2, xi2)):
        W, M = class_words(t, xi)
        s = (mp.mpf(xi[0]) + mp.mpf(xi[1]) * r2) / (4 * (mp.mpf(n_t(t)[0]) + mp.mpf(n_t(t)[1]) * r2))
        d = mp.sqrt(2 * s / (1 + mp.sqrt(1 - s)))
        fr = [int(mp.floor((mp.atan2(mp.mpf(B[0]) + B[1] * r2, mp.mpf(A[0]) + A[1] * r2) / mp.pi) * 2 ** FIX)) for (A, B, C, D) in W]
        classes.append({'t': t, 'xi': xi, 'M_norm_factors': [(int(p), int(e)) for p, e in flint.fmpz(abs(znorm(M))).factor()],
                        'words': W, 's': s, 'd': d, 'frac': fr, 'F': np.array(fr, dtype=object)})
        P(f'class t={t}: C^2+D^2 = {xi if m < 40 else "2 u^-..."}, N(4n_t - C^2 - D^2) = {classes[-1]["M_norm_factors"]}: '
          f'{len(W)} words at tube distance {mp.nstr(d, 15)} (2^t d^2 = {mp.nstr(2 ** t * d * d, 6)})  [{time.time() - T0:.0f}s]')
    # sample certificate by explicit gate strings
    rng = np.random.default_rng(1)
    for c in classes:
        idx = rng.choice(len(c['words']), size=min(args.sample, len(c['words'])), replace=False)
        okn = 0
        for i in idx:
            seq = synthesize(c['words'][i], c['t']); okn += (seq.count('T') == c['t'])
        P(f'  sample of {len(idx)} words of class t={c["t"]}: explicit gate strings with T-count {c["t"]}: {okn}/{len(idx)}')
    tau = t2
    def grid_count(c, e, Q):
        s = c['s']; sq = mp.sqrt(1 - s); base = 2 * s / (1 + sq)
        lim = (mp.mpf(e) ** 2 - base) / (4 * sq)                       # need sin^2(delta/2) <= lim
        if lim < 0: return 0
        dmax = 2 * mp.asin(mp.sqrt(lim))                               # |delta| <= dmax
        thr = int(mp.floor(dmax / mp.pi * Q * 2 ** FIX))                  # in units of 2^-FIX of a grid step
        one = 1 << FIX
        r = (c['F'] * Q) % one
        return int(np.count_nonzero(np.minimum(r, one - r) <= thr))
    best = {'grid': (-1e9,), 'tuned': (-1e9,), 'tube': (-1e9,)}
    rows = []
    for c_ref in classes:
        for eta in (0.0, 1e-9, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.75, 1.0):
            e = float(c_ref['d'] * (1 + eta)) * (1 + 1e-12)
            Qe = tuned_Q(e)
            Ntube = sum(len(c['words']) for c in classes if c['d'] <= e)
            x = 2.0 ** tau * e * e
            ct = (Ntube - 8 - 2 * tau) / x
            if ct > best['tube'][0]: best['tube'] = (ct, e, Ntube, x)
            for Q in range(max(3, Qe - args.window + 1), Qe + 1):
                Ng = sum(grid_count(c, e, Q) for c in classes)
                cg = (Ng - 8 - 2 * tau) / x
                rows.append((e, Q, Qe, Ng, Ntube, x, cg))
                if cg > best['grid'][0]: best['grid'] = (cg, e, Q, Ng, x)
                if Q == Qe and cg > best['tuned'][0]: best['tuned'] = (cg, e, Q, Ng, x)
    for e in sorted(set(r[0] for r in rows)):
        rr = [r for r in rows if r[0] == e]; b = max(rr, key=lambda r: r[6]); tq = next(r for r in rr if r[1] == r[2])
        P(f'  eps={e:.10e} (Q_eps={tq[2]}, 2^tau eps^2={tq[5]:.3f}, tube count {tq[4]}): best of {len(rr)} moduli Q={b[1]}: '
          f'grid count >= {b[3]}, c >= {b[6]:.1f}; tuned: count >= {tq[3]}, c >= {tq[6]:.1f}')
    P(f'chain m={m}, tau={tau}: required c >= {best["grid"][0]:.1f} (eps={best["grid"][1]:.10e}, Q={best["grid"][2]}, N>={best["grid"][3]}); '
      f'tuned Q: >= {best["tuned"][0]:.1f} (eps={best["tuned"][1]:.10e}, Q={best["tuned"][2]}, N>={best["tuned"][3]}); '
      f'tube: >= {best["tube"][0]:.1f}  [{time.time() - T0:.0f}s]')
    json.dump({'code': code_version(), 'm': m, 'tau': tau,
               'classes': [{'t': c['t'], 'xi': [str(v) for v in c['xi']], 'M_norm_factors': c['M_norm_factors'],
                            'n_words': len(c['words']), 'tube_distance': mp.nstr(c['d'], 20)} for c in classes],
               'best': {kk: [float(v) if isinstance(v, (float, int, np.floating)) else v for v in vv] for kk, vv in best.items()},
               'rows': rows}, open(os.path.join(DATA, f'torus_tube_chain_m{m}.json'), 'w'), indent=0)

def cmd_chain_survey(args):
    """Q-free (tube) lower bound from the two classes C^2 + D^2 = 2 u^-m (t = 3m-2) and 2 r2 u^-(m-1) (t = 3m-1),
    for which 4 n_t - C^2 - D^2 = unit * 2 r2^(0|1) * (X^m - 1), X = 4 + 2 r2; for every m <= mmax"""
    logf = open(os.path.join(DATA, 'torus_tube_chain_log.txt'), 'a')
    def P(*a):
        s = ' '.join(str(x) for x in a); print(s, flush=True); logf.write(s + '\n'); logf.flush()
    P(f'==== {time.strftime("%Y-%m-%d %H:%M:%S")} chain-survey mmax={args.mmax}  code {json.dumps(code_version())}')
    ubar = (-1, 1); out = []
    for m in range(2, args.mmax + 1, 2):          # for odd m neither element is totally positive
        t1, t2 = 3 * m - 2, 3 * m - 1
        xi1 = zmul((2, 0), zpow(ubar, m)); xi2 = zmul((0, 2), zpow(ubar, m - 1))
        assert s1f(xi1) > 0 and s2f(xi1) > 0 and s1f(xi2) > 0 and s2f(xi2) > 0
        t0 = time.time()
        if 8 * 2 ** sum(int(e) for p, e in flint.fmpz(abs(znorm(zsub(zpow((4, 2), m), (1, 0))))).factor()) > args.cap:
            P(f'm={m}: skipped (too many representations)'); continue
        W1, M1 = class_words(t1, xi1); W2, M2 = class_words(t2, xi2)
        s = s1f(xi1) / (4 * s1f(n_t(t1))); d = math.sqrt(2 * s / (1 + math.sqrt(1 - s)))
        x2 = 2.0 ** t2 * d * d
        c_tube = (len(W1) + len(W2) - 2 * t2) / x2
        fac = [(int(p), int(e)) for p, e in flint.fmpz(abs(znorm(zsub(zpow((4, 2), m), (1, 0))))).factor()]
        out.append({'m': m, 't': [t1, t2], 'words': [len(W1), len(W2)], 'tube_distance': d, 'log2_inv_d': -math.log2(d),
                    'x_at_t2': x2, 'tube_c_lower_bound': c_tube, 'N_Xm_minus_1_factors': fac})
        P(f'm={m:2d} t={t1},{t2} d=2^-{-math.log2(d):.2f} 2^t2 d^2={x2:9.3f} words {len(W1):7d}+{len(W2):7d}  '
          f'tube bound c >= {c_tube:8.1f}   N(X^m-1) = {fac}  [{time.time() - t0:.1f}s]')
    json.dump({'code': code_version(), 'rows': out}, open(os.path.join(DATA, 'torus_tube_chain_survey.json'), 'w'), indent=0)

if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    v = sub.add_parser('validate'); v.add_argument('--tmax', type=int, default=22); v.add_argument('--eps-ref', type=float, default=0.125)
    b = sub.add_parser('band', help='enumerate one accuracy band (2^-(L+1), 2^-L] and analyse it')
    b.add_argument('--L', type=int, required=True); b.add_argument('--K', type=int, default=12, help='tau_max = 2L + K')
    b.add_argument('--Kall', type=int, default=-1, help='the all-Q grid scan uses tau <= 2L + Kall (default: by budget)')
    b.add_argument('--G1', default='', help='gate string of G1 (default identity)'); b.add_argument('--G2', default='')
    b.add_argument('--name', default='identity')
    vf = sub.add_parser('validate-frames'); vf.add_argument('--tmax', type=int, default=16)
    vc = sub.add_parser('verify-cell', help='brute-force recount of one (frame, Q, eps, tau) cell')
    vc.add_argument('--Q', type=int, required=True); vc.add_argument('--eps', type=float, required=True)
    vc.add_argument('--tau', type=int, required=True); vc.add_argument('--G1', default=''); vc.add_argument('--G2', default='')
    vw = sub.add_parser('verify-words', help='explicit gate strings for every word of one identity-coset cell')
    vw.add_argument('--L', type=int, required=True, help='band whose torus cache (eps_hi = 2^-L) contains the cell')
    vw.add_argument('--Q', type=int, required=True); vw.add_argument('--eps', type=float, required=True)
    vw.add_argument('--tau', type=int, required=True)
    ch = sub.add_parser('chain-step', help='lower bounds from the minimal-transverse classes at t = 3 2^k - 2, 3 2^k - 1')
    ch.add_argument('--k', type=int, required=True); ch.add_argument('--sample', type=int, default=50)
    ch.add_argument('--window', type=int, default=256, help='moduli Q_eps - window + 1 .. Q_eps are tried for each eps')
    ch.add_argument('--m', type=int, default=0, help='use m instead of 2^k')
    cs = sub.add_parser('chain-survey'); cs.add_argument('--mmax', type=int, default=40); cs.add_argument('--cap', type=float, default=4e6)
    args = ap.parse_args()
    {'validate': cmd_validate, 'band': cmd_band, 'validate-frames': cmd_validate_frames,
     'verify-cell': cmd_verify_cell, 'verify-words': cmd_verify_words, 'chain-step': cmd_chain_step,
     'chain-survey': cmd_chain_survey}[args.cmd](args)
