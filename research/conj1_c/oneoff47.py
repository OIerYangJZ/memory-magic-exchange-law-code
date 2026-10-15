import sys, math
sys.path.insert(0, 'scripts')
import c_survey as CS
import numpy as np
T = CS.T
tau = 47
eps0 = 2.447978e-07
eps_hi, s, phi, tt, W = CS.collect(tau, 2.0 ** tau * (eps0 * 3.5) ** 2)
dt = T.tube_dist(s); sq = np.sqrt(1 - s)
best = []
for f in (1.0, 1.01, 1.03, 1.05, 1.1, 1.15, 1.2, 1.3):
    e = eps0 * f * (1 - 1e-12)
    ok = dt <= e
    half = 2 * np.arcsin(np.minimum(1.0, np.sqrt(np.maximum((e * e - dt[ok] ** 2) / (4 * sq[ok]), 0))))
    Qe = T.tuned_Q(e)
    for Q in [2919736, Qe] + list(range(Qe - 2000, Qe + 1)):
        if Q > Qe: continue
        n_noshift = int((T.grid_dist(s[ok], phi[ok], Q) <= e).sum())
        n, x = CS.max_shift_count(phi[ok], half, math.pi / Q)
        xx = 2.0 ** tau * e * e
        best.append(((n - 8 - 2 * tau) / xx, (n_noshift - 8 - 2 * tau) / xx, f, Q, n, n_noshift))
best.sort(reverse=True)
for b in best[:8]:
    print('shift %.1f  noshift %.1f  f=%.2f Q=%d N=%d N0=%d' % b)
