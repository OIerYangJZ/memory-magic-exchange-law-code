"""c_sample.py -- orbits of least transverse norm near the identity coset, level by level.

For each level t and y2 = eps0^k (1 - omega)^j (j = 0, 1; the two smallest admissible k), the words with this
transverse coordinate are (y1, omega^a y2) with y1 conj(y1) = m := 2 n_t - y2 conj(y2) and y1 = omega^a y2 mod
sqrt2; their number is set by the representation number r(m).  We factor N(m) (flint, with a time limit), build
the orbit exactly, and evaluate the tube requirement and the grid requirement of Conjecture 1 (frame (I, R_z(shift)),
shift optimized exactly, moduli Q <= Q_eps near Q_eps, multiples of 8 and random) for (c0, c1) = (8, 2).
Orbit-only counts are lower bounds on the counts, so both requirements are lower bounds for the identity coset
(the grid one for Conjecture 1 as stated, which quantifies over all frames)."""
import sys, time, math, json, random, bisect, signal
sys.path.insert(0, 'scripts')
import c_survey as CS
T = CS.T
import flint
import mpmath as mp
from multiprocessing import Pool

LNE = math.log(1 + math.sqrt(2))
R2W = (0, 1, 0, -1)
ONE_MINUS_OMEGA = (1, -1, 0, 0)


class TO(Exception):
    pass


def _alarm(*a):
    raise TO()


def grid_eval(phis, d, s, t, nrand, seed):
    sq = mp.sqrt(1 - s)
    B = t + 80
    fix = [int(mp.floor(ph / mp.pi * mp.mpf(2) ** B)) % (1 << B) for ph in phis]
    rng = random.Random(seed)
    best = (-1e18, None)
    for f in (1.02, 1.05, 1.1, 1.15, 1.2, 1.3, 1.4, 1.5, 1.75, 2.0, 2.5, 3.0):
        eps = d * f
        delta = 2 * mp.asin(mp.sqrt((eps * eps - d * d) / (4 * sq)))
        deltaT = 2 * mp.asin(eps / 2)
        Qe = int(mp.floor(mp.pi / (2 * mp.asin(2 * eps))))
        Qs = set(Qe - j for j in range(64) if Qe - j >= 3)
        Qs |= set(Qe - (Qe % 8) - 8 * j for j in range(64) if Qe - (Qe % 8) - 8 * j >= 8)
        lo = max(3, Qe // 2)
        Qs |= set(rng.randint(lo, Qe) for _ in range(nrand))
        xx = float(mp.mpf(2) ** t * eps * eps)
        for Q in Qs:
            w = float(2 * delta * Q / mp.pi)
            if w >= 1:
                n = len(phis)
            else:
                res = sorted(((F * Q) % (1 << B)) / float(1 << B) for F in fix)
                res2 = res + [r + 1.0 for r in res]
                n = max(bisect.bisect_right(res2, res2[i] + w) - i for i in range(len(res)))
            nT = 8 if float(2 * deltaT * Q / mp.pi) >= 1 else 0
            r = (n + nT - 8 - 2 * t) / xx
            if r > best[0]:
                best = (r, dict(f=f, Q=Q, Qe=Qe, N=n + nT, x=xx))
    return best


def one(args):
    t, j, extra, timeout, nrand, grid_thr = args
    signal.signal(signal.SIGALRM, _alarm)
    mp.mp.dps = int(t * 0.31) + 60
    nt = T.n_t(t)
    s1n, s2n = mp.mpf(nt[0]) + nt[1] * mp.sqrt(2), mp.mpf(nt[0]) - nt[1] * mp.sqrt(2)
    out = []
    # y2 conj(y2) = eps0^(2k) (2 - sqrt2)^j; sigma_2 of it = eps0^(-2k) (2 + sqrt2)^j; need < 2 sigma_2(n_t)
    kmin = math.floor(math.log((2 + math.sqrt(2)) ** j / (2 * float(s2n))) / (2 * LNE)) + 1
    for k in range(kmin - 1, kmin + 1 + extra):
        y2 = T.wemb(CS.unit_pow(k))
        for _ in range(j):
            y2 = T.wmul(y2, ONE_MINUS_OMEGA)
        n2 = T.wrelnorm(y2)
        m = T.zsub((2 * nt[0], 2 * nt[1]), n2)
        if not (m[0] > 0 and T.sign_qr2(m[0], -m[1]) > 0 and T.sign_qr2(m[0], m[1]) > 0):
            continue
        s = (mp.mpf(n2[0]) + n2[1] * mp.sqrt(2)) / (2 * s1n)
        d = mp.sqrt(2 * s / (1 + mp.sqrt(1 - s)))
        x = float(mp.mpf(2) ** t * d * d)
        rec = dict(t=t, j=j, k=k, x=x)
        t0 = time.time()
        signal.alarm(timeout)
        try:
            Y1 = T.relnorm_solutions(m)
        except TO:
            rec.update(status='timeout'); out.append(rec); continue
        finally:
            signal.alarm(0)
        Y2 = []
        a = y2
        for _ in range(8):
            Y2.append(a); a = T.wom(a)
        pairs = []
        for b in Y2:
            for y1 in Y1:
                dl = T.wmul(tuple(u - v for u, v in zip(y1, b)), R2W)
                if all(c % 2 == 0 for c in dl):
                    pairs.append((y1, b))
        nwords = len(pairs) // 2
        rec.update(status='ok', r=len(Y1) // 8, words=nwords, tube_req=(nwords - 2 * t) / x,
                   secs=round(time.time() - t0, 1))
        if nwords and rec['tube_req'] >= grid_thr:
            w8 = [mp.expjpi(mp.mpf(i) / 4) for i in range(4)]
            seen = set(); phis = []
            for y1, b in pairs:
                if (tuple(-c for c in y1), tuple(-c for c in b)) in seen:
                    continue
                seen.add((y1, b))
                phis.append(mp.arg(sum(mp.mpf(c) * w8[i] for i, c in enumerate(y1))))
            g = grid_eval(phis, d, s, t, nrand, seed=t * 10 + j)
            rec.update(grid_req=g[0], grid=g[1])
        out.append(rec)
    return out


if __name__ == '__main__':
    t0, t1, procs = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
    timeout = int(sys.argv[4]) if len(sys.argv) > 4 else 20
    outf = sys.argv[5] if len(sys.argv) > 5 else 'data/c_sample.jsonl'
    jobs = [(t, j, 1, timeout, 400, 400.0) for t in range(t0, t1 + 1) for j in (0, 1)]
    with Pool(procs) as pool, open(outf, 'a') as f:
        for recs in pool.imap_unordered(one, jobs):
            for r in recs:
                f.write(json.dumps(r) + '\n'); f.flush()
                if r.get('status') == 'ok' and r['words'] and r['tube_req'] > 300:
                    print(json.dumps(r), flush=True)
