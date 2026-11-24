#!/usr/bin/env python3
"""thm1_explicit.py -- explicit constants for Theorem 2 (thm:clifford) and Theorem 1 (thm:typical), and the
T-counts that Theorem 1 certifies at practical accuracy.  Pure Python 3 + numpy; every number printed is
reproducible (no randomness).  Run:  python scripts/thm1_explicit.py  [--selfcheck] [--exact]

Notation (proof of Theorem 2, explicit form).
  omega = e^{i pi/4};  O = maximal order (eq:order).  w = a+bi+cj+dk  <->  (z1, z2) = (a+bI, c+dI), and
      O = {(z1, z2) : z2 in Z[omega]/sqrt2,  z1 - z2 in Z[omega]}.
  With y_i = sqrt2 z_i:  y1, y2 in Z[omega],  y1 = y2 mod sqrt2,  |y1|^2 + |y2|^2 = 2 n_t,  where
      n_t = 2^{t/2} (t even),  (2+sqrt2) 2^{(t-1)/2} (t odd),   N(n_t) = 2^t.
  Tube of the torus {R_z(theta)} of projective radius eps:  |y2|_1^2 <= 2 eps'^2 sigma1(n_t),
      eps'^2 = eps^2 (1 - eps^2/4);  the norm equation gives |y2|_2^2 <= 2 sigma2(n_t).
  G_t(eps) = #{y in Z[omega] : |y|_1^2 <= 2 eps'^2 sigma1(n_t),  |y|_2^2 <= 2 sigma2(n_t)}
           <= (pi^2/4) (sqrt(2 eps') 2^{t/4} + kappa)^4,   kappa = sqrt(4+3 sqrt2)/2 = 1.4355,
           and G_t = 1 if 2 eps' 2^{t/2} < 1.
  r(m) = #{ideals A of Z[omega] : N_{Q(omega)/Q(sqrt2)} A = (m)} = prod over primes p | m split in Q(omega)/Q(sqrt2)
         of (e_p + 1)  (and 0 if some inert prime has odd exponent; the ramified prime (sqrt2) contributes 1).
  R(X) = max{ r(m) : N(m) <= X }, computed exactly (highly-composite search over the split prime ideals,
         whose norms are p, p for p = 1 mod 8 and p^2 for p = 3, 5 mod 8).
  Words (Theorem 2):  #{W : t_min(W) <= tau, W in T_eps(C1 R_z C2)} <= Wb(tau)
         := 8 + 4 * sum_{t in {tau-1, tau}, t >= 0} R(2^{t+2}) (G_t - 1).
  Angles (Theorem 1):  |{theta in [0,4pi) : T-count_eps(R_z(theta)) <= tau}| <= 16 arcsin(eps/2) Wb(tau);
         fraction of grid angles 2 pi d / Q (Q with sin(pi/2Q) > eps) <= Wb(tau) / Q.
  "All but a fraction delta need T-count >= N":  N = 1 + max{tau : fraction(tau) <= delta}.
"""
import math
import sys
from fractions import Fraction

import numpy as np

SQ2 = math.sqrt(2.0)
EPS0 = 1.0 + SQ2                       # fundamental unit of Z[sqrt2]; |eps0|_1 = eps0, |eps0|_2 = 1/eps0


# ----------------------------------------------------------------------------------------------------------
# 1. Geometry of Z[omega] in C^2 (Minkowski embedding y -> (y, sigma2 y), sigma2: omega -> omega^5)
# ----------------------------------------------------------------------------------------------------------
def cube_projection_radius():
    """max over t in [-1/2,1/2]^4 of |t0 + t1 w + t2 w^2 + t3 w^3| (attained at a vertex); the same for sigma2."""
    w = np.exp(1j * np.pi / 4)
    best = 0.0
    for s in range(16):
        v = sum(((s >> j) & 1 and 1 or -1) * w ** j for j in range(4)) / 2
        best = max(best, abs(v))
    return best


RHO = cube_projection_radius()                          # = sqrt(4+2 sqrt2)/2 = 1.30656
KAPPA = RHO * math.sqrt(2 + 2 * SQ2) / 2                # = sqrt(4+3 sqrt2)/2 = 1.43549
assert abs(RHO - math.sqrt(4 + 2 * SQ2) / 2) < 1e-12
assert abs(KAPPA - math.sqrt(4 + 3 * SQ2) / 2) < 1e-12


def n_t(t):
    """n_t = p + q sqrt2 as an integer pair."""
    if t % 2 == 0:
        return (2 ** (t // 2), 0)
    return (2 * 2 ** ((t - 1) // 2), 2 ** ((t - 1) // 2))


def zs2_mul(x, y):
    return (x[0] * y[0] + 2 * x[1] * y[1], x[0] * y[1] + x[1] * y[0])


def zs2_pow(x, e):
    out = (1, 0)
    for _ in range(e):
        out = zs2_mul(out, x)
    return out


def le0(A, B):
    """exact test A + B sqrt2 <= 0 for rationals A, B."""
    if A <= 0 and B <= 0:
        return True
    if A >= 0 and B >= 0:
        return A == 0 and B == 0
    if A > 0:            # B < 0
        return A * A <= 2 * B * B
    return 2 * B * B <= A * A    # A < 0 < B


def G_bound(t, eps):
    """closed-form bound (pi^2/4)(sqrt(2 eps') 2^{t/4} + kappa)^4, and 1 if 2 eps' 2^{t/2} < 1."""
    e1 = eps * math.sqrt(1 - eps * eps / 4)
    P = 2 * e1 * 2 ** (t / 2)
    if P < 1:
        return 1.0
    return (math.pi ** 2 / 4) * (math.sqrt(P) + KAPPA) ** 4


def G_exact(t, eps_frac, cap=4.0e7):
    """exact lattice count G_t(eps); eps_frac a Fraction. Returns None if the enumeration would exceed cap."""
    eps = float(eps_frac)
    e2 = eps_frac * eps_frac * (1 - eps_frac * eps_frac / 4)          # eps'^2, exact
    p, q = n_t(t)
    s1, s2 = p + q * SQ2, p - q * SQ2
    U2, V2 = 2 * float(e2) * s1, 2 * s2
    if U2 * V2 < 1:
        return 1
    # balance by a totally positive unit eps0^{2j}: U'^2 = eps0^{2j} U^2, V'^2 = eps0^{-2j} V^2
    j = round(-math.log(U2 / V2) / (4 * math.log(EPS0)))
    u = zs2_pow((3, 2), j) if j >= 0 else zs2_pow((3, -2), -j)
    pp, qq = zs2_mul(u, (p, q))
    # sigma_i(eps0^{2j} n) computed multiplicatively (no cancellation): sigma1 scales by eps0^{2j}, sigma2 by eps0^{-2j}
    U2p, V2p = U2 * EPS0 ** (2 * j), V2 * EPS0 ** (-2 * j)
    amax = (U2p + V2p) / 2                         # sum x_i^2 = (|y|_1^2 + |y|_2^2)/2
    B = int(math.isqrt(int(amax) + 1)) + 1
    if (2 * B + 1) ** 4 > cap * 50:
        return None
    rng = np.arange(-B, B + 1, dtype=np.int64)
    X1, X2, X3 = np.meshgrid(rng, rng, rng, indexing='ij')
    X1, X2, X3 = X1.ravel(), X2.ravel(), X3.ravel()
    base_a = X1 * X1 + X2 * X2 + X3 * X3
    base_b = X1 * X2 + X2 * X3
    count = 0
    c1A, c1B = 2 * e2 * pp, 2 * e2 * qq              # exact bound for sigma1:  a + b sqrt2 <= c1A + c1B sqrt2
    c2A, c2B = 2 * pp, 2 * qq                        # sigma2:  a - b sqrt2 <= c2A - c2B sqrt2
    tol = 1e-9 * max(1.0, U2p, V2p)
    for x0 in range(-B, B + 1):
        a = base_a + x0 * x0
        b = base_b + x0 * X1 - x0 * X3
        f1 = a + b * SQ2 - U2p
        f2 = a - b * SQ2 - V2p
        inside = (f1 <= -tol) & (f2 <= -tol)
        count += int(inside.sum())
        border = (~inside) & (f1 <= tol) & (f2 <= tol)
        for ai, bi in zip(a[border].tolist(), b[border].tolist()):
            if le0(ai - c1A, bi - c1B) and le0(ai - c2A, -(bi - c2B)):
                count += 1
    return count


# ----------------------------------------------------------------------------------------------------------
# 2. The arithmetic factor R(X)
# ----------------------------------------------------------------------------------------------------------
def primes_upto(n):
    s = bytearray([1]) * (n + 1)
    s[0:2] = b"\x00\x00"
    for i in range(2, int(n ** 0.5) + 1):
        if s[i]:
            s[i * i::i] = bytearray(len(s[i * i::i]))
    return [i for i in range(n + 1) if s[i]]


PRIMES = primes_upto(200000)


def split_norms(qmax):
    """norms of the prime ideals of Z[sqrt2] that split in Q(omega), all <= qmax, sorted, with multiplicity."""
    out = []
    for p in PRIMES:
        if p > qmax:
            break
        if p % 8 == 1:
            out += [p, p]
        elif p % 8 in (3, 5) and p * p <= qmax:
            out.append(p * p)
    return sorted(out)


def all_norms(qmax):
    """norms of all prime ideals of Z[sqrt2] (for d_K)."""
    out = []
    for p in PRIMES:
        if p > qmax:
            break
        if p == 2:
            out.append(2)
        elif p % 8 in (1, 7):
            out += [p, p]
        elif p * p <= qmax:
            out.append(p * p)
    return sorted(out)


def max_multiplicative(X, norms):
    """max of prod (e_i + 1) subject to prod q_i^{e_i} <= X, q_i = norms (sorted, with multiplicity).
    Exchange argument: an optimum has exponents non-increasing along the sorted list, so its support is an
    initial segment; the search below enumerates exactly those exponent vectors."""
    prod = 1
    for q in norms:
        prod *= q
    assert prod > X, "norm list too short"
    best = [1, []]
    n = len(norms)

    def dfs(i, rem, val, emax, exps):
        if val > best[0]:
            best[0], best[1] = val, list(exps)
        if i >= n or norms[i] > rem:
            return
        q, pw = norms[i], 1
        for e in range(1, emax + 1):
            pw *= q
            if pw > rem:
                break
            exps.append(e)
            dfs(i + 1, rem // pw, val * (e + 1), e, exps)
            exps.pop()

    dfs(0, X, 1, 10 ** 6, [])
    return best[0], best[1]


_RCACHE = {}
SPLIT = split_norms(10 ** 5)
ALLN = all_norms(10 ** 5)


def R(X):
    if X not in _RCACHE:
        _RCACHE[X] = max_multiplicative(X, SPLIT)[0]
    return _RCACHE[X]


def R_tau(t):
    return R(2 ** (t + 2)) if t >= 0 else 0


def log2_R_closed(X):
    """rigorous closed-form majorant: r <= d_K <= d(N)^2 and Nicolas-Robin log2 d(n) <= c ln n / ln ln n for n >= 3,
    optimal c = 1.53794 (attained at n = 6983776800), used here as 1.538 (ln n / ln ln n increases for n >= 16,
    and d(n) <= 6 for n < 16)."""
    lx = math.log(X)
    return 2 * max(math.log2(6), 1.538 * lx / math.log(lx))


# ----------------------------------------------------------------------------------------------------------
# 3. Bounds
# ----------------------------------------------------------------------------------------------------------
def words_bound(tau, eps, G):
    """Wb(tau) = 8 + 4 sum_{t in {tau-1,tau}} R(2^{t+2}) (G_t - 1)."""
    if tau < 0:
        return 0.0
    tot = 8.0
    for t in (tau - 1, tau):
        if t >= 0:
            tot += 4 * R_tau(t) * (G(t) - 1)
    return tot


def words_bound_closed(tau, eps):
    """the clean form stated in the theorem: 8 + (3 pi^2/2) R(2^{tau+2}) (sqrt(2 eps) 2^{tau/4} + 8/5)^4."""
    if tau < 0:
        return 0.0
    return 8 + 1.5 * math.pi ** 2 * R_tau(tau) * (math.sqrt(2 * eps) * 2 ** (tau / 4) + 1.6) ** 4


def theta_measure_per_word(eps):
    return 16 * math.asin(eps / 2)


def Q_tuned(eps):
    return int(math.floor(math.pi / (2 * math.asin(2 * eps))))


def certified(frac, delta, tau_hi=400):
    """N = 1 + max{tau : frac(tau) <= delta}; frac is non-decreasing in tau."""
    best = -1
    for tau in range(0, tau_hi):
        if frac(tau) <= delta:
            best = tau
        else:
            break
    return best + 1


TUBE_C1 = 72 * 4 * SQ2
TUBE_C2 = 24 / (1 - 2 ** -0.5) * 2 ** 1.5


def tube_bound(k, eps):
    """Theorem 3 (any frame), optimized over the Poisson width w (eq. tubew), w*eps < 1."""
    best = TUBE_C1 * 2 ** k * eps ** 2 + TUBE_C2 * (k + 1) * 2 ** (k / 2)
    for lw in np.linspace(-3, 30, 661):
        w = 2.0 ** lw
        if w * eps >= 1:
            break
        r = 1 - w * eps
        c1 = 72 * (w * w + 4 * r) ** 1.5 / (w * (1 + r))
        c2 = 24 / (1 - 2 ** -0.5) * (w * w + 4 * r) ** 1.5 / w ** 3
        best = min(best, c1 * 2 ** k * eps ** 2 + c2 * (k + 1) * 2 ** (k / 2))
    return best


def elementary_threshold(eps):
    """largest tau such that every word of T-count <= tau within eps of a z-rotation is a power of T
    (Lemma identity with |t|_1^2 <= eps'^2 2^k and the denominator exponent k <= ceil(tau/2) + 1)."""
    e2 = eps * eps * (1 - eps * eps / 4)
    tau = 0
    while e2 * 4.0 ** (-(-(tau + 1) // 2) + 1) < 1:
        tau += 1
    return tau


# ----------------------------------------------------------------------------------------------------------
# 4. Report
# ----------------------------------------------------------------------------------------------------------
def main():
    print("=" * 100)
    print("[1] Lattice constants")
    print(f"    Z[omega] in C^2: covolume 4 (basis 1,w,w^2,w^3 orthogonal of length sqrt2); cube-projection radius"
          f" rho = sqrt(4+2sqrt2)/2 = {RHO:.5f}")
    print(f"    kappa = rho (eps0^(1/2)+eps0^(-1/2))/2 = sqrt(4+3sqrt2)/2 = {KAPPA:.5f}")
    print(f"    => G(U,V) = #{{y: |y|_1<=U, |y|_2<=V}} <= (pi^2/4)(sqrt(UV)+kappa)^4, and = 1 if UV < 1")
    # check (a+k)^4 + (2^-1/4 a + k)^4 <= 3/2 (a + 8/5)^4 for all a >= 0 (used for the clean form)
    aa = np.concatenate([np.linspace(0, 50, 200001), np.logspace(1.7, 8, 20001)])
    lhs = (aa + KAPPA) ** 4 + (2 ** -0.25 * aa + KAPPA) ** 4
    rhs = 1.5 * (aa + 1.6) ** 4
    print(f"    clean form: max_a [(a+kappa)^4 + (2^(-1/4) a + kappa)^4] / [1.5 (a+8/5)^4] = {np.max(lhs / rhs):.5f} (<= 1)")
    assert np.max(lhs / rhs) <= 1
    # 2/(UV)-free check of the bound against exact counts
    print("    exact G vs bound (eps = 1e-10):")
    epsf = Fraction(1, 10 ** 10)
    for t in (64, 66, 68, 70, 72, 74, 76, 80, 84):
        ge = G_exact(t, epsf)
        print(f"      t={t:3d}  G_exact={ge:>10}  bound={G_bound(t, 1e-10):12.1f}  "
              f"main term pi^2 eps'^2 2^t = {math.pi**2 * 1e-20 * 2.0**t:12.1f}")

    print("=" * 100)
    print("[2] Arithmetic factor R(X) = max r(m), N(m) <= X  (r = number of ideals of Z[omega] of relative norm m)")
    print("    split prime-ideal norms (first 16):", SPLIT[:16])
    print(f"    {'tau':>4} {'log2 R(2^(tau+2))':>18} {'log2 max d_K':>14} {'closed form':>12}   optimal exponents of R")
    for tau in list(range(20, 161, 10)) + [99, 66]:
        X = 2 ** (tau + 2)
        r, ex = max_multiplicative(X, SPLIT)
        dk, _ = max_multiplicative(X, ALLN)
        print(f"    {tau:4d} {math.log2(r):18.3f} {math.log2(dk):14.3f} {log2_R_closed(X):12.2f}   {ex}")
    X = 16 * 2 ** 99
    dk, exk = max_multiplicative(X, ALLN)
    print(f"    paper (line 706): max d_K over N <= 16*2^99 is 2^{math.log2(dk):.2f} (exponents {exk}); "
          f"the relevant R there is 2^{math.log2(R(X)):.2f}")
    # Nicolas-Robin spot check
    n0 = 6983776800
    print(f"    Nicolas-Robin constant at the extremal n = {n0} (d(n) = 2304): "
          f"log2 d(n) ln ln n / ln n = {math.log2(2304) * math.log(math.log(n0)) / math.log(n0):.6f}  (closed form uses 1.538)")

    for eps_str, epsf in (("1e-6", Fraction(1, 10 ** 6)), ("1e-10", Fraction(1, 10 ** 10)),
                          ("1e-15", Fraction(1, 10 ** 15))):
        eps = float(epsf)
        L = math.log2(1 / eps)
        Q = Q_tuned(eps)
        mw = theta_measure_per_word(eps)
        gcache = {}

        def G(t):
            if t not in gcache:
                ge = G_exact(t, epsf)
                gcache[t] = ge if ge is not None else G_bound(t, eps)
            return gcache[t]

        print("=" * 100)
        print(f"[3] eps = {eps_str}: L = {L:.4f}, 3L = {3*L:.3f}, tuned Q = {Q}, theta-measure per word "
              f"16 arcsin(eps/2) = {mw:.6e} = {mw/eps:.5f} eps")
        fr_ex = lambda tau: mw / (4 * math.pi) * words_bound(tau, eps, G)
        fr_bd = lambda tau: mw / (4 * math.pi) * words_bound(tau, eps, lambda t: G_bound(t, eps))
        fr_cl = lambda tau: mw / (4 * math.pi) * words_bound_closed(tau, eps)
        gr_ex = lambda tau: words_bound(tau, eps, G) / Q
        gr_cl = lambda tau: words_bound_closed(tau, eps) / Q
        triv = lambda tau: mw / (4 * math.pi) * (72 * 2 ** tau - 48)
        tube = lambda tau: mw / (4 * math.pi) * tube_bound(tau, eps)
        tid = elementary_threshold(eps)
        vol = lambda tau: mw / (4 * math.pi) * 72 * eps * eps * 2.0 ** tau
        for delta in (1e-2, 1e-3):
            n_ex, n_bd, n_cl = certified(fr_ex, delta), certified(fr_bd, delta), certified(fr_cl, delta)
            g_ex, g_cl = certified(gr_ex, delta), certified(gr_cl, delta)
            n_tr, n_tb = certified(triv, delta), certified(tube, delta)
            print(f"    delta = {delta:g}:  Theorem 1 certifies T-count >= N for all but a fraction delta of angles:")
            print(f"        theta in [0,4pi):  N = {n_cl} (clean closed form)  {n_bd} (two-parity form, closed-form G)"
                  f"  {n_ex} (two-parity form, exact lattice count G)")
            print(f"        tuned grid Q:      N = {g_cl} (clean closed form)  {g_ex} (exact G)")
            print(f"        comparison:  trivial count of all 72*2^t words: N = {n_tr};  tube bound (Thm 3, best w): "
                  f"N = {n_tb};  Lemma identity: all but {128*math.asin(eps/2)/(4*math.pi):.1e} need >= {tid+1};"
                  f"  volume heuristic (72 eps^2 2^tau words, not a theorem): N = {certified(vol, delta)}")
        # detail rows around the threshold
        tau_c = certified(fr_ex, 1e-2) - 1
        print("    detail (exact G):  tau  log2R  G_tau  words-bound  fraction-bound  [3L - tau]")
        for tau in range(tau_c - 3, tau_c + 4):
            print(f"        {tau:4d} {math.log2(R_tau(tau)):6.2f} {G(tau):10.0f} {words_bound(tau, eps, G):14.4g} "
                  f"{fr_ex(tau):14.4g}   [{3*L - tau:6.2f}]")
        if eps_str == "1e-10":
            print("    measured: pygridsynth mean T-count 99.28 at d_proj <= 1e-10 over 20000 grid angles (Sec. IV);"
                  " not recomputed here")
    print("=" * 100)


# ----------------------------------------------------------------------------------------------------------
# 5. Optional: the exact number of words in the torus tube (replaces the worst case R by the actual r(m))
# ----------------------------------------------------------------------------------------------------------
_MR_BASES = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41)     # deterministic for n < 3.3e24


def is_prime(n):
    if n < 2:
        return False
    for p in _MR_BASES:
        if n % p == 0:
            return n == p
    assert n < 3317044064679887385961981, "Miller-Rabin bases not certified for n this large"
    d, s = n - 1, 0
    while d % 2 == 0:
        d //= 2
        s += 1
    for a in _MR_BASES:
        x = pow(a, d, n)
        if x in (1, n - 1):
            continue
        for _ in range(s - 1):
            x = x * x % n
            if x == n - 1:
                break
        else:
            return False
    return True


def _brent(n):
    if n % 2 == 0:
        return 2
    for c in range(1, 200):
        y, r, q, g, x, ys, m = 2, 1, 1, 1, 2, 2, 128
        while g == 1:
            x = y
            for _ in range(r):
                y = (y * y + c) % n
            k = 0
            while k < r and g == 1:
                ys = y
                for _ in range(min(m, r - k)):
                    y = (y * y + c) % n
                    q = q * abs(x - y) % n
                g = math.gcd(q, n)
                k += m
            r *= 2
        if g == n:
            g = 1
            while g == 1:
                ys = (ys * ys + c) % n
                g = math.gcd(abs(x - ys), n)
        if g != n:
            return g
    raise RuntimeError("rho failed")


def factor(n, out=None):
    out = {} if out is None else out
    for p in PRIMES[:2000]:
        if n % p == 0:
            while n % p == 0:
                out[p] = out.get(p, 0) + 1
                n //= p
        if p * p > n:
            break
    stack = [n] if n > 1 else []
    while stack:
        m = stack.pop()
        if m == 1:
            continue
        if is_prime(m):
            out[m] = out.get(m, 0) + 1
            continue
        d = _brent(m)
        stack += [d, m // d]
    return out


def _sqrt2_mod(p):
    """s with s^2 = 2 mod p, p = +-1 mod 8 (Tonelli-Shanks)."""
    q, s = p - 1, 0
    while q % 2 == 0:
        q //= 2
        s += 1
    z = 2
    while pow(z, (p - 1) // 2, p) != p - 1:
        z += 1
    m, c, t, r = s, pow(z, q, p), pow(2, q, p), pow(2, (q + 1) // 2, p)
    while t != 1:
        i, t2 = 0, t
        while t2 != 1:
            t2 = t2 * t2 % p
            i += 1
        b = pow(c, 1 << (m - i - 1), p)
        m, c, t, r = i, b * b % p, t * b * b % p, r * b % p
    return r


def r_of(u, v):
    """r(m) for m = u + v sqrt2 totally positive: number of ideals of Z[omega] of relative norm (m)."""
    N = u * u - 2 * v * v
    assert N > 0
    out = 1
    for p, e in factor(N).items():
        if p == 2:
            continue
        if p % 8 in (3, 5):                  # inert in Q(sqrt2), split in Q(omega)
            out *= e // 2 + 1
            continue
        s = _sqrt2_mod(p)                     # p = +-1 mod 8: p = P P' in Q(sqrt2)
        pk = p ** (e + 1)
        for _ in range(e + 1):                # Hensel-lift s to a root of x^2 = 2 mod p^(e+1)
            s = (s - (s * s - 2) * pow(2 * s, -1, pk)) % pk
        x, e1 = (u + v * s) % pk, 0
        while e1 <= e and x % p == 0:
            x //= p
            e1 += 1
        e1 = min(e1, e)
        e2 = e - e1
        if p % 8 == 1:
            out *= (e1 + 1) * (e2 + 1)
        elif e1 % 2 or e2 % 2:                # p = 7 mod 8: inert in Q(omega)/Q(sqrt2)
            return 0
    return out


def v_sqrt2(a, b):
    """sqrt2-adic valuation of a + b sqrt2 (not both zero)."""
    v = 0
    while a % 2 == 0:
        a, b = b, a // 2                      # (a + b sqrt2)/sqrt2 = b + (a/2) sqrt2
        v += 1
    return v


def box_points(t, eps_frac):
    """the y2 in the (unit-balanced) box, as (a, b) = |y2|^2 = a + b sqrt2, plus the balanced 2 n' = (P, Qc)."""
    e2 = eps_frac * eps_frac * (1 - eps_frac * eps_frac / 4)
    p, q = n_t(t)
    s1, s2 = p + q * SQ2, p - q * SQ2
    U2, V2 = 2 * float(e2) * s1, 2 * s2
    if U2 * V2 < 1:
        return [(0, 0)], (2 * p, 2 * q)
    j = round(-math.log(U2 / V2) / (4 * math.log(EPS0)))
    u = zs2_pow((3, 2), j) if j >= 0 else zs2_pow((3, -2), -j)
    pp, qq = zs2_mul(u, (p, q))
    U2p, V2p = U2 * EPS0 ** (2 * j), V2 * EPS0 ** (-2 * j)
    B = int(math.isqrt(int((U2p + V2p) / 2) + 1)) + 1
    rng = np.arange(-B, B + 1, dtype=np.int64)
    X1, X2, X3 = [z.ravel() for z in np.meshgrid(rng, rng, rng, indexing='ij')]
    base_a, base_b = X1 * X1 + X2 * X2 + X3 * X3, X1 * X2 + X2 * X3
    c1A, c1B, c2A, c2B = 2 * e2 * pp, 2 * e2 * qq, 2 * pp, 2 * qq
    tol = 1e-9 * max(1.0, U2p, V2p)
    pts = []
    for x0 in range(-B, B + 1):
        a = base_a + x0 * x0
        b = base_b + x0 * X1 - x0 * X3
        f1, f2 = a + b * SQ2 - U2p, a - b * SQ2 - V2p
        cand = (f1 <= tol) & (f2 <= tol)
        for ai, bi, g1, g2 in zip(a[cand].tolist(), b[cand].tolist(), f1[cand].tolist(), f2[cand].tolist()):
            if (g1 <= -tol and g2 <= -tol) or (le0(ai - c1A, bi - c1B) and le0(ai - c2A, -(bi - c2B))):
                pts.append((ai, bi))
    return pts, (2 * pp, 2 * qq)


def A_exact(t, eps_frac):
    """#{w in O : nrd(w) = n_t, w in the eps-tube of the torus}, exactly, and max r(m) over the box."""
    pts, (P, Qc) = box_points(t, eps_frac)
    tot, rmax = 0, 1
    for a, b in pts:
        if a == 0 and b == 0:
            tot += 8
            continue
        mu, mv = P - a, Qc - b
        rr = r_of(mu, mv)
        rmax = max(rmax, rr)
        v2, vm = v_sqrt2(a, b), v_sqrt2(mu, mv)
        if v2 == 0:
            tot += 4 * rr
        elif v2 == 1:
            tot += 8 * rr if vm == 1 else 0
        else:
            tot += 8 * rr if vm >= 2 else 0
    return tot, rmax, len(pts)


def words_exact(tau, eps_frac):
    if tau < 0:
        return 0, 1
    a1, r1, _ = A_exact(tau, eps_frac)
    a0, r0, _ = (A_exact(tau - 1, eps_frac) if tau >= 1 else (0, 1, 0))
    assert (a1 + a0) % 2 == 0
    return (a1 + a0) // 2, max(r0, r1)


def exact_section(cases):
    import time
    print("=" * 100)
    print("[4] Exact number of words in the torus tube (actual r(m) instead of the worst case R; computer-assisted)")
    for eps_str, epsf, taus in cases:
        eps = float(epsf)
        mw = theta_measure_per_word(eps) / (4 * math.pi)
        print(f"    eps = {eps_str}:  tau, exact #words, bound Wb(tau), ratio, max r over box vs R(2^(tau+2)),"
              f" exact fraction, volume heuristic 72 eps^2 2^tau")
        for tau in taus:
            t0 = time.time()
            cnt, rmax = words_exact(tau, epsf)
            wb = words_bound(tau, eps, lambda t: G_exact(t, epsf))
            print(f"      {tau:4d} {cnt:14d} {wb:14.4g} {wb / cnt:8.1f}   {rmax:6d} vs {R_tau(tau):8d}   "
                  f"{mw * cnt:10.3e}   {72 * eps * eps * 2.0 ** tau:12.4g}   ({time.time() - t0:.1f}s)")


def selfcheck():
    """Brute-force check of Wb(tau) >= true count at the identity coset for small tau (uses research/zomega.py)."""
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'research'))
    from zomega import all_words, canon, val
    tmax = 10
    best = {}
    for t, W in all_words(tmax):
        c = canon(W)
        if c not in best or t < best[c][0]:
            best[c] = (t, W)
    for eps in (0.125, 0.08, 0.05, 0.03):
        e2 = eps * eps * (1 - eps * eps / 4)
        rows = []
        for tau in range(tmax + 1):
            cnt = sum(1 for (t, (k, M)) in best.values()
                      if t <= tau and abs(val(M[1][0])) ** 2 / 2 ** k <= e2 + 1e-12)
            wb = words_bound(tau, eps, lambda t: G_exact(t, Fraction(eps).limit_denominator(10 ** 9)))
            assert cnt <= wb, (eps, tau, cnt, wb)
            ex, _ = words_exact(tau, Fraction(eps).limit_denominator(10 ** 9))
            assert ex == cnt, (eps, tau, ex, cnt)
            rows.append((tau, cnt, int(wb)))
        print(f"selfcheck eps={eps}: (tau, true count = exact count of [4], bound)", rows)


if __name__ == "__main__":
    if "--selfcheck" in sys.argv:
        selfcheck()
    main()
    if "--exact" in sys.argv:
        exact_section([("1e-6", Fraction(1, 10 ** 6), range(38, 51)),
                       ("1e-10", Fraction(1, 10 ** 10), range(64, 79))])
