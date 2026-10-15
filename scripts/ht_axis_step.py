"""ht_axis_step.py -- the largest required constant of Conjecture 1 found on the HT-axis coset.

Supplemental Sec. S1 (axis cosets) and Appendix M.  The dyadic scans (conj1_scan.py, scan2.py) give c >= 53.5
on the HT-axis frame at eps = 2^-5.  At non-dyadic accuracies the requirement is larger: a six-word orbit sits
at d_proj = 0.026499139 from the grid of modulus Q = 29, and for eps just above it (eps = 0.0265, still
Q_eps = 29) there are 69 words of minimal T-count <= 9 within eps of a grid rotation on the coset, so that
c0 + c1 tau + c 2^tau eps^2 >= 69 at tau = 9 forces c >= 119.6 with (c0, c1) = (8, 2).  For comparison the
same count at eps = 0.0264991 (just below the orbit) is 63.  The overall largest requirement we know is still
the identity-coset step c >= 163.9 (astra1.py).

Re-uses the enumeration and counting code of conj1_scan.py.  Run from the repository root:
    .venv/bin/python scripts/ht_axis_step.py        (seconds; output in data/ht_axis_step_output.txt)"""
import numpy as np, sys
import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))
exec(open(_os.path.join(_HERE, 'conj1_scan.py')).read().split('# ---- 1. reproduce')[0]
     .replace('TMAX = int(sys.argv[1]) if len(sys.argv) > 1 else 14', 'TMAX = 9'))
G1, G2 = frames['axis HT (exact)']
c0, c1, tau = 8, 2, 9
for eps in (0.0264991, 0.0265):
    Q = Qeps(eps)
    g, tb = counts(G1, G2, eps, Q)
    cum = int(g[:tau + 1].sum())
    print(f"HT-axis frame, eps={eps}, Q_eps={Q} (sin(pi/2Q)={np.sin(np.pi / (2 * Q)):.6f} >= 2 eps): "
          f"grid shells t=0..{tau} {g[:tau + 1].tolist()}, cumulative {cum}; "
          f"required c = (N - c0 - c1 tau)/(2^tau eps^2) = {(cum - c0 - c1 * tau) / (2 ** tau * eps ** 2):.1f}")
# the distances of the words near eps (grid of Q = 29)
Q = 29; th = 2 * np.pi * np.arange(Q) / Q; e1, e2 = np.exp(1j * th / 2), np.exp(-1j * th / 2)
near = []
for t in range(tau + 1):
    M = np.einsum('ij,njk,kl->nil', G1.conj().T, SH[t], G2.conj().T)
    v = np.abs(M[:, 0, 0][:, None] * e1[None, :] + M[:, 1, 1][:, None] * e2[None, :]).max(axis=1)
    d = np.sqrt(np.maximum(0, 2 - v))
    sel = (d > 0.0264) & (d < 0.0266)
    near += [(t, round(float(x), 9)) for x in d[sel]]
print(f"words of T-count <= {tau} at distance in (0.0264, 0.0266) from the Q = 29 grid: {len(near)}, "
      f"at distances {sorted(set(x for _, x in near))}, T-counts {sorted(set(t for t, _ in near))}")
