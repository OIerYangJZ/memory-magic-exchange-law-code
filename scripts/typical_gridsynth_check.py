"""typical_gridsynth_check.py -- Ross-Selinger (pygridsynth) against the exact optimum on small tuned grids.

For every grid angle 2 pi d / Q_eps, eps = 2^-L, L = 5..10, synthesizes a word with pygridsynth
(up_to_phase=True, parameter e = 2 eps, the convention of gridsynth_run.py: the word then has
d_proj <= eps up to O(eps^3), see below), evaluates its d_proj to the target exactly (mpmath, 40 digits),
and compares its T-count with the exact minimal T-count tau_min(d) of data/typical_enum.json
(scripts/typical_enum.py).  Writes data/typical_gridsynth_check.json.

pygridsynth accepts a candidate when Re(conj(z) u) >= sqrt(1 - e^2/4); d_proj <= eps needs
|Re| >= 1 - eps^2/2, which is slightly stronger, so at e = 2 eps a returned word can exceed eps by a
relative eps^2/8 (at L = 5: 1.5e-5, for two angles, where it beats tau_min by 3).  Such words are
flagged (d_proj > eps) and excluded from the comparison.  At eps = 1e-10 the excess is 1e-21 relative
and gridsynth_run.py measures worst d_proj <= eps directly.

Run from the repository root:  .venv/bin/python scripts/typical_gridsynth_check.py   (seconds on 10 cores)
"""
import json, os
from multiprocessing import Pool

import mpmath
import numpy as np
from pygridsynth import GridsynthConfig, gridsynth_gates

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, os.pardir, 'data')
DPS = 40


def gate_matrices():
    H = mpmath.matrix([[1, 1], [1, -1]]) / mpmath.sqrt(2)
    return {'H': H, 'S': mpmath.matrix([[1, 0], [0, 1j]]),
            'T': mpmath.matrix([[1, 0], [0, mpmath.expjpi(mpmath.mpf(1) / 4)]]),
            'X': mpmath.matrix([[0, 1], [1, 0]]), 'I': mpmath.eye(2)}


def job(arg):
    L, d, Q = arg
    if d == 0:
        return 0, 0.0
    with mpmath.workdps(DPS):
        eps = mpmath.mpf(2) ** -L
        th = 2 * mpmath.pi * d / Q
        g = gridsynth_gates(th, 2 * eps, cfg=GridsynthConfig(up_to_phase=True))
        G = gate_matrices(); M = mpmath.eye(2)
        for c in g:
            if c in G:
                M = M * G[c]
            elif c != 'W':                        # W is the global phase omega
                raise ValueError(c)
        U = M * mpmath.matrix([[mpmath.expj(th / 2), 0], [0, mpmath.expj(-th / 2)]])   # M Rz(th)^dagger
        tr = abs(U[0, 0] + U[1, 1]) / abs(mpmath.sqrt(mpmath.det(M)))
        dproj = mpmath.sqrt(max(mpmath.mpf(0), 2 - tr))
        return g.count('T'), float(dproj / eps)


if __name__ == '__main__':
    E = json.load(open(os.path.join(DATA, 'typical_enum.json')))
    out = {'convention': 'pygridsynth up_to_phase=True, e = 2 eps; d_proj_over_eps exact (mpmath)', 'L': {}}
    with Pool() as P:
        for L in range(5, 11):
            Q = E['Q'][str(L)]; tm = np.array(E['taumin'][str(L)])
            res = P.map(job, [(L, d, Q) for d in range(Q)])
            gs = np.array([r[0] for r in res]); dr = np.array([r[1] for r in res])
            valid = dr <= 1.0
            known = (tm >= 0) & valid
            gap = gs[known] - tm[known]
            rec = {'Q': Q, 'gridsynth_tcount': gs.tolist(), 'dproj_over_eps': [round(x, 9) for x in dr],
                   'n_exceeding_eps': int((~valid).sum()), 'n_compared': int(known.sum()),
                   'gap_min': int(gap.min()), 'gap_max': int(gap.max()), 'gap_mean': round(float(gap.mean()), 4),
                   'gap_hist': {int(k): int(v) for k, v in zip(*np.unique(gap, return_counts=True))},
                   'unknown_tau_min_gridsynth_min': int(gs[(tm < 0) & valid].min()) if ((tm < 0) & valid).any() else None,
                   'TMAX': E['TMAX']}
            out['L'][L] = rec
            print(L, Q, {k: v for k, v in rec.items() if k not in ('gridsynth_tcount', 'dproj_over_eps')}, flush=True)
    json.dump(out, open(os.path.join(DATA, 'typical_gridsynth_check.json'), 'w'), indent=0)
    print('wrote data/typical_gridsynth_check.json')
