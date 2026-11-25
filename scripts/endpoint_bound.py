#!/usr/bin/env python3
"""endpoint_bound.py -- the zero-memory endpoint of the rate-two bound (IEEE TIT version, Corollary 4).

At S = 0 and delta = 0 a word committed on a coordinate determines that coordinate (nearest-grid decoding), and the
coordinate is uniform on K values given the other rounds, so
    E t_min >= sum_{tau >= 0} (1 - min(K, U_w(tau)) / K)_+,
    U_w(tau) = min(72*2^tau - 48, C1(w) 2^tau eps^2 + C2(w) (tau+1) 2^(tau/2)),
with C1(w), C2(w) the width-w constants of the tube bound (eq. tubew); w may be chosen separately for every tau.

Prints (1) an exact rational lower bound with w = 2^14, every irrational constant replaced by a rational upper bound,
and (2) a 60-digit evaluation with w optimised for each tau.  Run:  python scripts/endpoint_bound.py
"""
from fractions import Fraction as F
from math import isqrt
import mpmath as mp

EPS_STR = '1e-10'


def tuned_K(eps_mp):
    x = mp.pi / (2 * mp.asin(2 * eps_mp))
    Q = int(mp.floor(x))
    assert 0.01 < float(x - Q) < 0.99, 'floor too close to an integer'
    return Q - 1


def sqrt_bounds(q, bits=200):
    s = isqrt((q.numerator << (2 * bits)) // q.denominator)
    return F(s, 1 << bits), F(s + 1, 1 << bits)


def rational_lower_bound(eps=F(1, 10**10), w=F(2**14)):
    mp.mp.dps = 80
    K = tuned_K(mp.mpf(eps.numerator) / eps.denominator)
    sq2_lo, sq2_up = sqrt_bounds(F(2))
    r = 1 - w * eps
    assert 0 < r < 1
    X = w * w + 4 * r
    A_up = X * sqrt_bounds(X)[1]                 # >= (w^2 + 4r)^{3/2}
    C1_up = 72 * A_up / (w * (1 + r))
    C2_up = 24 * A_up / (w**3 * (1 - 1 / sq2_lo))   # 1/(1 - 2^{-1/2}) <= 1/(1 - 1/sq2_lo)
    total, tau = F(0), 0
    while True:
        p2h = F(2**(tau // 2)) * (sq2_up if tau % 2 else 1)
        U = min(F(72 * 2**tau - 48), C1_up * 2**tau * eps * eps + C2_up * (tau + 1) * p2h)
        term = 1 - min(F(K), U) / K
        if term <= 0:
            return total, K, tau
        total += term
        tau += 1


def optimised(eps_str=EPS_STR):
    mp.mp.dps = 60
    eps = mp.mpf(eps_str)
    K = tuned_K(eps)

    def U_w(tau, w):
        r = 1 - w * eps
        A = (w**2 + 4 * r)**mp.mpf(1.5)
        return (72 * A / (w * (1 + r))) * mp.mpf(2)**tau * eps**2 \
            + (24 / (1 - 2**mp.mpf(-0.5))) * A / w**3 * (tau + 1) * mp.mpf(2)**(mp.mpf(tau) / 2)

    total, tau = mp.mpf(0), 0
    while True:
        ws = [mp.mpf(2)**(j / mp.mpf(16)) for j in range(16 * 33) if mp.mpf(2)**(j / mp.mpf(16)) * eps < 1]
        U = min(mp.mpf(72) * 2**tau - 48, min(U_w(tau, w) for w in ws))
        term = 1 - min(mp.mpf(K), U) / K
        if term <= 0:
            return total, K
        total += term
        tau += 1


if __name__ == '__main__':
    lb, K, n = rational_lower_bound()
    print(f'eps = 1e-10, tuned K = {K}, log2 K = {float(mp.log(K, 2)):.3f}')
    print(f'exact rational lower bound (w = 2^14, {n} terms): {float(lb):.10f}   > 39.968: {lb > F(39968, 1000)}')
    val, _ = optimised()
    print(f'60-digit value, w optimised per tau:           {mp.nstr(val, 12)}')
    print(f'per bit: {float(lb) / float(mp.log(K, 2)):.4f}')
