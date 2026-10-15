"""search for Z = sqrt2^p eps0^q with |Z'| slightly above 1 such that Z^g - 1 is a relative norm from Z[omega]
(no inert prime to an odd power) with many split prime factors; prints r(Z^g - 1) and x = |Z'|^g / 4."""
import sys, time, math, json, signal
sys.path.insert(0, 'scripts')
import c_survey as CS
T = CS.T
import flint

LN2, LNE = math.log(2), math.log(1 + math.sqrt(2))
ALPHA = LN2 / (2 * LNE)


class TO(Exception):
    pass


def _alarm(*a):
    raise TO()


signal.signal(signal.SIGALRM, _alarm)


def phi_d_at(d, Z):
    co = [int(c) for c in flint.fmpz_poly.cyclotomic(d).coeffs()]
    acc = (0, 0)
    for c in reversed(co):
        acc = T.zadd(T.zmul(acc, Z), (c, 0))
    return acc


def Zof(p, q):
    s = T.zpow((2, 0), p // 2) if p % 2 == 0 else T.zmul(T.zpow((2, 0), p // 2), (0, 1))
    return T.zmul(s, CS.unit_pow(q))


def analyse(F, timeout):
    """(clean, nsplit_ideals, r_contribution) for the element F of Z[sqrt2], or None on timeout"""
    nF = abs(T.znorm(F))
    signal.alarm(timeout)
    try:
        fl = [(int(p), int(e)) for p, e in flint.fmpz(nF).factor()]
    except TO:
        return None
    finally:
        signal.alarm(0)
    r = 1; nsp = 0
    for p, e in fl:
        if p == 2:
            continue
        pd = T.prime_data(p)
        if pd[0] == 'split_inert':
            r *= e // 2 + 1; nsp += 1
        elif pd[0] == 'inert':
            v1 = T.zval(F, pd[1], e); v2 = e - v1
            if v1 % 2 or v2 % 2:
                return (False, 0, 0)
        else:
            v1 = T.zval(F, pd[2], e); v2 = e - v1
            r *= (v1 + 1) * (v2 + 1); nsp += (v1 > 0) + (v2 > 0)
    return (True, nsp, r)


def main(pmax, lmax, gmax, timeout):
    for p in range(1, pmax + 1):
        for q in (math.floor(ALPHA * p), math.floor(ALPHA * p) - 1):
            l1 = p / 2 * LN2 - q * LNE          # log |Z'|
            if not (0 < l1 < lmax):
                continue
            Z = Zof(p, q)
            res = {}
            ok = True
            for d in (1, 2):
                a = analyse(phi_d_at(d, Z), timeout)
                res[d] = a
                if a is None or not a[0]:
                    ok = False; break
            if not ok:
                continue
            # Z - 1, Z + 1 clean: try g = 2^a * small, the Phi_d(Z) with 8 | d are clean in principle
            for g in [g for g in range(2, gmax + 1) if g % 2 == 0]:
                if (g * q - (g * p - 2)) % 2:
                    continue
                x = math.exp(g * l1) / 4
                if x > 64:
                    break
                r = 1; clean = True; to = False
                for d in [d for d in range(1, g + 1) if g % d == 0]:
                    a = res.get(d) or analyse(phi_d_at(d, Z), timeout)
                    res[d] = a
                    if a is None:
                        to = True; break
                    if not a[0]:
                        clean = False; break
                    r *= a[2]
                if to or not clean:
                    continue
                print(json.dumps(dict(p=p, q=q, g=g, t=g * p - 2, logZc=l1, x=x, r=r, words_est=8 * r * 8 // 4,
                                      tube_req_est=(8 * r * 8 // 4) / x)), flush=True)


if __name__ == '__main__':
    main(int(sys.argv[1]), float(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]))
