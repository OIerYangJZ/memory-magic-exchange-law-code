"""thm_numbers.py -- the bounds of Theorems 1 (rate one) and 3 (rate two, and its finite form, Corollary 4) per coordinate and
per pre-final round, as functions of the bits shed per coordinate b = log2 K - S/m, at eps = 1e-10, delta = 0,
m = 1e4.  Reproduces Table I (b = log2 K) and the curves of Fig. 3(b).
Corollary 4 is evaluated with the width-w tube constants (eq. (D10)); Table I uses (w, lambda, n) =
(2^14, 5/16, 40), valid for eps < 2^-14, which is within 1e-3 of the best value of a broad scan (also printed);
Fig. 3 optimizes (w, lambda, n) pointwise.  Each choice gives a valid bound (b - B(w, lambda, n)) / lambda,
so the pointwise maximum is one too.
Theorem 8 (rate three under Conjecture 1) has no explicit constants and is not evaluated."""
import numpy as np

EPS = 1e-10
L = np.log2(1 / EPS)
Q = int(np.floor(np.pi / (2 * np.arcsin(2 * EPS))))
K = Q - 1
k = np.log2(K)
m = 1e4

# constants C1, C2 of Theorem 2 and A_2 of Theorem 3 (eqs. (5), (6))
C1 = 72 * 4 * np.sqrt(2)
C2 = 24 / (1 - 2 ** -0.5) * 2 ** 1.5
tau_star = int(np.ceil(2 * k))
Z = sum((C1 * 2 ** t * EPS ** 2 + C2 * (t + 1) * 2 ** (t / 2)) * 2 ** (-t / 2) for t in range(tau_star + 1))
A2 = 1 + np.log2(Z)


def fixed_point(f, x0=1.0):
    """largest x >= 0 with x = f(x), f decreasing; bisection on g(x) = x - f(x)"""
    if f(0.0) <= 0:
        return 0.0
    lo, hi = 0.0, 1.0
    while hi - f(hi) < 0:
        hi *= 2
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if mid - f(mid) < 0: lo = mid
        else: hi = mid
    return 0.5 * (lo + hi)

def thm1(b):   # eq. (3): S + T + m log2 36 + rho_m(T) >= m log2 K, per coordinate, m -> large
    return fixed_point(lambda x: b - np.log2(36) - np.log2(np.e * (1 + x)) - (2 * np.log2(1 + m * x) + 1) / m)

def thm3(b):
    return max(0.0, 2 * (b - A2))

# Corollary 4: Lemma 9 (summed profile with an exact tail) with U(t) = min(72 2^t - 48, C1(w) 2^t eps^2 + C2(w)(t+1) 2^(t/2))
W_TUBE, LAM, NCUT = 2 ** 14, 5 / 16, 40
def tube_consts(w, eps=EPS):
    rho = 1 - w * eps
    assert 0 < rho < 1, 'width w needs w*eps < 1'
    return (72 * (w * w + 4 * rho) ** 1.5 / (w * (1 + rho)),
            24 / (1 - 2 ** -0.5) * (w * w + 4 * rho) ** 1.5 / w ** 3)

def cor4(b, lam=LAM, n=NCUT, w=W_TUBE, eps=EPS):
    Qe = int(np.floor(np.pi / (2 * np.arcsin(2 * eps)))); Ke = Qe - 1
    C1w, C2w = tube_consts(w, eps); q = 2.0 ** -lam
    t = np.arange(n + 1)
    U = np.minimum(72 * 2.0 ** t - 48, C1w * 2.0 ** t * eps ** 2 + C2w * (t + 1) * 2.0 ** (t / 2))
    Zup = q ** n * U[n] + (1 - q) * np.sum(q ** t[:n] * U[:n])
    return max(0.0, (b - np.log2(Zup + Ke * 2.0 ** (-lam * (n + 1)))) / lam)

_GRID = {}
def cor4_grid(eps=EPS, tmax=160):
    """all (B, lambda) of Corollary 4 over w = 2^j (w eps < 1), lambda in [0.05, 1], 0 <= n <= tmax:
    the bound is (b - B) / lambda with B = log2(Z_up + K 2^(-lambda (n+1))) independent of b"""
    if eps in _GRID:
        return _GRID[eps]
    Ke = int(np.floor(np.pi / (2 * np.arcsin(2 * eps)))) - 1
    T = np.arange(tmax + 1); Bs, lams = [], []
    for j in range(1, 60):
        w = 2.0 ** j
        if w * eps >= 1:
            break
        C1w, C2w = tube_consts(w, eps)
        U = np.minimum(72 * 2.0 ** T - 48, C1w * 2.0 ** T * eps ** 2 + C2w * (T + 1) * 2.0 ** (T / 2))
        for lam in np.arange(0.05, 1.0001, 0.005):
            q = 2.0 ** -lam
            pref = np.concatenate([[0.0], np.cumsum(q ** T * U)])[:-1]
            Bs.append(np.log2(q ** T * U + (1 - q) * pref + Ke * 2.0 ** (-lam * (T + 1))))
            lams.append(np.full(T.shape, lam))
    _GRID[eps] = (np.concatenate(Bs), np.concatenate(lams))
    return _GRID[eps]

def cor4_best(b, eps=EPS):
    """(w, lambda, n) chosen pointwise; every choice gives a valid bound, so the maximum is one"""
    B, lam = cor4_grid(eps)
    return max(0.0, float(np.max((b - B) / lam)))

# achievable point: shares synthesized to d_proj <= eps/2 (the certified r = 2 point), data/gridsynth_tau.json
import json as _json, os as _os
_GS = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), _os.pardir, 'data')
ACH = _json.load(open(_os.path.join(_GS, 'gridsynth_tau.json')))['tau_mean']                 # 102.32
CAL = _json.load(open(_os.path.join(_GS, 'gridsynth_tau_dproj_eps.json')))['tau_mean']       # d_proj <= eps


if __name__ == '__main__':
    print(f'eps={EPS:g}  L={L:.3f}  Q_eps={Q}  log2K={k:.3f}')
    print(f'tau*={tau_star}  A2={A2:.3f}')
    x4 = cor4(k)
    print(f'Cor 4 (w={W_TUBE}, lambda={LAM}, n={NCUT}): T_t/m = {x4:.5f}   per bit = {x4 / k:.4f}   '
          f'-dT/dS = 1/lambda = {1 / LAM:.3f}   (Thm 3 as stated: {thm3(k):.3f}; ratio achievable/Cor 4 = {ACH / x4:.2f})')
    print(f'Cor 4, best over the scan (w = 2^j, lambda in [0.05, 1], n <= 160): {cor4_best(k):.5f}')
    for name, f, slope in (('Thm 1', thm1, 1), ('Thm 3', thm3, 2)):
        x = f(k)
        print(f'{name}: T_t/m = {x:.3f}   per bit = {x / k:.3f}   slope = {slope:.3f}')
    for name, f in (('Thm 1', thm1), ('Thm 3', thm3)):
        h = 1e-4
        print(f'{name}: finite-precision slope -dT/dS at S=0 = {(f(k) - f(k - h)) / h:.4f}')
    print(f'calibration (gridsynth, d_proj <= eps): {CAL:.3f}, per bit = {CAL / k:.3f}')
    print(f'achievable (gridsynth, shares at d_proj <= eps/2): {ACH:.3f}, per bit = {ACH / k:.3f}, slope = {ACH / np.log2(Q):.4f}')
    print(f'ratio achievable / Thm 1 = {ACH / thm1(k):.2f};  / Cor 4 = {ACH / cor4(k):.2f}')
    # cross-over of Thm 1 and Thm 3 in S/m
    bs = np.linspace(0, k, 2001)
    d = np.array([thm3(b) - thm1(b) for b in bs])
    i = np.argmax(d > 0)
    print(f'Thm 3 exceeds Thm 1 for S/m < {k - bs[i]:.2f} bits')
    d4 = np.array([cor4_best(b) - thm1(b) for b in bs])
    i4 = np.argmax(d4 > 0)
    print(f'Cor 4 (optimized pointwise) exceeds Thm 1 for S/m < {k - bs[i4]:.2f} bits ({100 * (k - bs[i4]) / k:.0f}% of the range)')
    for Lx in np.arange(3.0, 40.0, 0.1):     # S = 0 crossover in accuracy (Thm 1 evaluated at the same eps; eps <= 1/8)
        ex = 2.0 ** -Lx; kx = np.log2(int(np.floor(np.pi / (2 * np.arcsin(2 * ex)))) - 1)
        t1 = fixed_point(lambda x: kx - np.log2(36) - np.log2(np.e * (1 + x)) - (2 * np.log2(1 + m * x) + 1) / m)
        if cor4_best(kx, eps=ex) > t1:
            print(f'at S=0, Cor 4 exceeds Thm 1 from L = {Lx:.1f} on'); break
