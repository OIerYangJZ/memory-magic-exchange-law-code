"""c_orbit.py -- the orbit of words of least transverse norm for Z = sqrt2^p eps0^q, level t = g p - 2, built
directly (y2 = omega^a eps0^k, y1 over all solutions of y1 conj(y1) = m = 2 n_t - eps0^(2k), y1 = y2 mod sqrt2),
in exact arithmetic with mpmath angles; tube and grid requirements of Conjecture 1 at the identity coset for
(c0, c1) = (8, 2).  Grid: frame (I, R_z(shift)), moduli Q <= Q_eps (near Q_eps, multiples of 8, random),
the shift maximized exactly (sliding window on the residues phi Q / pi mod 1, computed in fixed point)."""
import sys, time, math, json, random, bisect
sys.path.insert(0, 'scripts')
import c_survey as CS
T = CS.T
import flint
import mpmath as mp

FACT = {}


class _Fmpz:
    def __init__(self, n):
        self.n = int(n); self.f = flint.fmpz(n)

    def factor(self):
        if abs(self.n) in FACT:
            return [(flint.fmpz(p), e) for p, e in FACT[abs(self.n)]]
        return self.f.factor()

    def sqrtmod(self, p):
        return self.f.sqrtmod(p)


class _FP:
    fmpz = _Fmpz
    fmpz_mat = flint.fmpz_mat


T.flint = _FP


def phi_d_at(d, Z):
    co = [int(c) for c in flint.fmpz_poly.cyclotomic(d).coeffs()]
    acc = (0, 0)
    for c in reversed(co):
        acc = T.zadd(T.zmul(acc, Z), (c, 0))
    return acc


def Zof(p, q):
    s = T.zpow((2, 0), p // 2) if p % 2 == 0 else T.zmul(T.zpow((2, 0), p // 2), (0, 1))
    return T.zmul(s, CS.unit_pow(q))


R2W = (0, 1, 0, -1)      # sqrt2 = omega - omega^3


def orbit(p, q, g, nrand=600, seed=1):
    t0 = time.time()
    t = g * p - 2
    k = ((t % 2) - g * q) // 2
    assert (t % 2) - g * q == 2 * k
    mp.mp.dps = int(t * 0.31) + 60
    Z = Zof(p, q)
    nt = T.n_t(t)
    u2k = CS.unit_pow(2 * k)
    m = T.zsub((2 * nt[0], 2 * nt[1]), u2k)
    # factorization of N(m) from the cyclotomic pieces of Z^g - 1
    fac = {}
    prod = 1
    for d in [d for d in range(1, g + 1) if g % d == 0]:
        nF = abs(T.znorm(phi_d_at(d, Z)))
        prod *= nF
        for pr, e in flint.fmpz(nF).factor():
            fac[int(pr)] = fac.get(int(pr), 0) + int(e)
    assert prod == abs(T.znorm(m))
    FACT[abs(T.znorm(m))] = sorted(fac.items())
    Y1 = T.relnorm_solutions(m)                  # all y1 with y1 conj(y1) = m
    uk = T.wemb(CS.unit_pow(k))
    Y2 = []
    a = uk
    for _ in range(8):
        Y2.append(a); a = T.wom(a)
    pairs = []
    for y2 in Y2:
        for y1 in Y1:
            dlt = T.wmul(tuple(u - v for u, v in zip(y1, y2)), R2W)
            if all(c % 2 == 0 for c in dlt):
                pairs.append((y1, y2))
    nwords = len(pairs) // 2
    # angles: phi = arg sigma_1(y1), omega -> e^{i pi/4}
    w8 = [mp.expjpi(mp.mpf(j) / 4) for j in range(4)]
    phis = []
    seen = set()
    for y1, y2 in pairs:
        key = (tuple(-c for c in y1), tuple(-c for c in y2))
        if key in seen:
            continue
        seen.add((y1, y2))
        z = sum(mp.mpf(c) * w8[j] for j, c in enumerate(y1))
        phis.append(mp.arg(z))
    assert len(phis) == nwords, (len(phis), nwords)
    # T-powers: phi = -j pi/8
    phis_T = [-mp.pi * j / 8 for j in range(8)]
    # transverse distance of the orbit
    e0 = 1 + mp.sqrt(2)
    s = mp.power(e0, 2 * k) / (2 * (nt[0] + nt[1] * mp.sqrt(2)))
    sq = mp.sqrt(1 - s)
    d = mp.sqrt(2 * s / (1 + sq))
    x0 = mp.mpf(2) ** t * d * d
    Ntube = 8 + nwords
    tube_req = float((Ntube - 8 - 2 * t) / x0)
    # grid
    B = t + 80
    allphi = phis + phis_T
    fix = [int(mp.floor(ph / mp.pi * mp.mpf(2) ** B)) % (1 << B) for ph in allphi]
    isT = [False] * len(phis) + [True] * 8
    rng = random.Random(seed + t)
    best = (-1e18, None)
    for f in (1.02, 1.05, 1.1, 1.15, 1.2, 1.3, 1.4, 1.5, 1.75, 2.0, 2.5, 3.0):
        eps = d * f
        # half-width delta: 2 s/(1+sq) + 4 sq sin^2(delta/2) <= eps^2
        val = (eps * eps - d * d) / (4 * sq)
        delta = 2 * mp.asin(mp.sqrt(val))
        deltaT = 2 * mp.asin(eps / 2)            # T-powers: s = 0
        Qe = int(mp.floor(mp.pi / (2 * mp.asin(2 * eps))))
        Qs = set(Qe - j for j in range(64) if Qe - j >= 3)
        Qs |= set(Qe - (Qe % 8) - 8 * j for j in range(64) if Qe - (Qe % 8) - 8 * j >= 8)
        lo = max(3, Qe // 2)
        Qs |= set(rng.randint(lo, Qe) for _ in range(nrand))
        for Q in Qs:
            w = float(2 * delta * Q / mp.pi)          # window in units of the period pi/Q
            wT = float(2 * deltaT * Q / mp.pi)
            if w >= 1:
                n = len(phis)
            else:
                res = sorted(((F * Q) % (1 << B)) / float(1 << B) for F, tflag in zip(fix, isT) if not tflag)
                res2 = res + [r + 1.0 for r in res]
                n = max(bisect.bisect_right(res2, res2[i] + w) - i for i in range(len(res)))
            nT = 8 if wT >= 1 else 0                  # T-powers: count only if every one is caught (conservative)
            N = n + nT
            xx = float(mp.mpf(2) ** t * eps * eps)
            r = (N - 8 - 2 * t) / xx
            if r > best[0]:
                best = (r, dict(f=f, Q=Q, Qe=Qe, N=N, x=xx))
    return dict(p=p, q=q, g=g, t=t, k=k, x0=float(x0), words=nwords, r_sols=len(Y1) // 8, tube_req=tube_req,
                grid_req=best[0], grid=best[1], secs=round(time.time() - t0, 1))


if __name__ == '__main__':
    p, q = int(sys.argv[1]), int(sys.argv[2])
    for g in map(int, sys.argv[3:]):
        print(json.dumps(orbit(p, q, g)), flush=True)
