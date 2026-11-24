"""make_fig_typical.py -- Fig. 1 (fig_typical.pdf): Theorems 1 and 2 against exact counts and Ross--Selinger data.

(a) Theorem 2 at the identity coset: the number of Clifford+T unitaries of minimal T-count <= tau within
    projective distance eps = 2^-L of the torus {R_z(theta)}, against tau - 2L = log2(eps^2 2^tau), with the
    Haar volume 72 eps^2 2^tau and the 8 words lying on the torus (exhaustive enumeration to T-count 28).
(b) Theorem 1: the fraction of theta in [0, 4 pi) for which R_z(theta) has an eps-approximation of T-count
    <= 3L - s (exact, from the same enumeration), and the fraction of the 20 000 sampled grid angles at
    eps = 1e-10 whose Ross--Selinger word has T-count <= 3L - s (a lower bound on the fraction for the optimum).
(c) The distribution of T-count - 3L: exact optimum on all grid angles of the tuned grid at L = 8, and
    Ross--Selinger at eps = 1e-10 (an upper bound on the optimum for each angle).

Inputs:  data/typical_enum.json            (scripts/typical_enum.py, ~3.5 min, ~4 GB)
         data/gridsynth_tau_dproj_eps.json  (scripts/gridsynth_run.py --dproj 1e-10 --out gridsynth_tau_dproj_eps.json)
Run from the repository root:  .venv/bin/python scripts/make_fig_typical.py   (seconds; writes fig_typical.pdf,
byte-identical on reruns: no creator/producer/date metadata)."""
import json, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(os.path.join(HERE, os.pardir))

plt.rcParams.update({
    'font.family': 'serif', 'font.serif': ['STIXGeneral'], 'mathtext.fontset': 'stix',
    'font.size': 8, 'axes.labelsize': 8, 'xtick.labelsize': 7, 'ytick.labelsize': 7, 'legend.fontsize': 7,
    'axes.linewidth': 0.6, 'xtick.major.width': 0.6, 'ytick.major.width': 0.6,
    'xtick.minor.width': 0.4, 'ytick.minor.width': 0.4, 'xtick.major.size': 2.5, 'ytick.major.size': 2.5,
    'xtick.minor.size': 1.5, 'ytick.minor.size': 1.5, 'xtick.direction': 'in', 'ytick.direction': 'in',
    'pdf.fonttype': 42, 'lines.linewidth': 1.0, 'legend.frameon': False, 'legend.handlelength': 1.6,
})

D = json.load(open('data/typical_enum.json'))
GS = json.load(open('data/gridsynth_tau_dproj_eps.json'))

LS = (4, 6, 8, 10, 12)                                  # accuracies shown, eps = 2^-L
# ordinal blue ramp (light -> dark), validated: monotone lightness, adjacent gaps >= 0.06
BLUE = dict(zip(LS, ('#86b6ef', '#5598e7', '#2a78d6', '#1c5cab', '#0d366b')))
ORANGE = '#eb6834'                                      # Ross--Selinger
INK, GREY = '#0b0b0b', '#76756f'
MK = dict(zip(LS, ('o', 's', '^', 'D', 'v')))

fig, axs = plt.subplots(1, 3, figsize=(6.5, 2.3), gridspec_kw=dict(width_ratios=(1, 1.08, 1)))
fig.subplots_adjust(left=0.075, right=0.995, bottom=0.175, top=0.925, wspace=0.36)

# ------------------------------------------------------------------ (a) Theorem 2: count vs volume
ax = axs[0]
xlo, xhi = -5, 13
xs = np.linspace(xlo, xhi, 200)
ax.plot(xs, 72 * 2.0 ** xs, '--', color=GREY, lw=0.9, zorder=1)
ax.axhline(8, ls=':', color=GREY, lw=0.9, zorder=1)
for L in LS:
    n = np.cumsum(D['tube'][str(L)])
    tau = np.arange(len(n)); x = tau - 2 * L
    m = (x >= xlo) & (x <= xhi)
    ax.plot(x[m], n[m], '-', marker=MK[L], ms=2.6, mew=0, color=BLUE[L], label=r'$L=%d$' % L, zorder=3)
ax.set_yscale('log')
ax.set_xlim(xlo - 0.3, xhi + 0.3); ax.set_ylim(2, 2e6)
ax.set_xticks(range(-4, 13, 4))
ax.set_xlabel(r'$\tau-2L=\log_2(\varepsilon^2\,2^\tau)$')
ax.set_ylabel(r'words with $t_{\min}\leq\tau$ in the $\varepsilon$-tube', labelpad=1)
ax.text(5.6, 72 * 2.0 ** 5.6 * 0.12, r'volume $72\,\varepsilon^2 2^\tau$', color=GREY, rotation=0,
        ha='left', va='top', fontsize=7)
ax.text(xhi, 6.3, r'8 words on the torus', color=GREY, ha='right', va='top', fontsize=7)
ax.legend(loc='upper left', handlelength=1.4, borderaxespad=0.3, labelspacing=0.25)

# ------------------------------------------------------------------ (b) Theorem 1: lower tail
ax = axs[1]
slo, shi = -2.5, 15.5
ss = np.linspace(0, shi, 50)
ax.plot(ss, 2.0 ** -ss, '--', color=GREY, lw=0.9, zorder=1, label=r'$2^{-s}$')
for L in LS:
    meas = np.array(D['meas'][str(L)]); tau = np.arange(len(meas)); s = 3 * L - tau
    m = (s >= slo) & (s <= shi)
    ax.plot(s[m], meas[m], '-', marker=MK[L], ms=2.6, mew=0, color=BLUE[L], zorder=3)
    # direct label at the right end of the floor 16 eps / pi (the 8 words on the torus)
    sl = min(shi, 3 * L - 1) - 0.2
    ax.text(sl, meas[3 * L - int(np.ceil(sl))] * 1.18, r'$L=%d$' % L, color=INK, ha='right', va='bottom', fontsize=7)
# Ross--Selinger at eps = 1e-10, d_proj <= eps: fraction of sampled grid angles with tau_RS <= 3L - s
taus = np.array(GS['taus']); N = len(taus); L_gs = GS['L']
tv = np.arange(taus.min(), int(np.floor(3 * L_gs - slo)) + 1)
k = np.array([(taus <= t).sum() for t in tv])
sel = k >= 5
s_gs = 3 * L_gs - tv[sel]; p = k[sel] / N
ax.errorbar(s_gs, p, yerr=np.sqrt(k[sel]) / N, fmt='-', marker='o', ms=2.8, mfc='white', mec=ORANGE, mew=0.9,
            color=ORANGE, lw=0.9, elinewidth=0.6, capsize=0, zorder=4,
            label='Ross–Selinger, ' + r'$\varepsilon=10^{-10}$')
ax.set_yscale('log')
ax.set_xlim(slo, shi); ax.set_ylim(1e-4, 1.6)
ax.set_xticks(range(0, 16, 3))
ax.set_xlabel(r'$s$  ($T$-count $\leq 3L-s$)')
ax.set_ylabel(r'fraction of angles $\theta$', labelpad=1)
from matplotlib.lines import Line2D
hd, lb = ax.get_legend_handles_labels()
hd.insert(0, Line2D([], [], color=BLUE[8], marker=MK[8], ms=2.6, mew=0)); lb.insert(0, r'optimum, $\varepsilon=2^{-L}$')
ax.legend(hd, lb, loc='lower left', borderaxespad=0.3, labelspacing=0.25)

# ------------------------------------------------------------------ (c) distribution of T-count - 3L
ax = axs[2]
L8 = 8
tm = np.array(D['taumin'][str(L8)]); assert (tm >= 0).all()
vals, cnt = np.unique(tm - 3 * L8, return_counts=True)
tv_gs, cnt_gs = np.unique(taus, return_counts=True)
ax.axvline(0, ymax=0.63, color=GREY, lw=0.6, zorder=0)
ax.bar(tv_gs - 3 * L_gs, cnt_gs / N, width=0.72, color=ORANGE, alpha=0.9, lw=0, zorder=2,
       label='Ross–Selinger,\n' + r'$\varepsilon=10^{-10}$, $3L=%.2f$' % (3 * L_gs))
v = vals[vals >= -10]; edges = np.arange(v.min() - 0.5, v.max() + 1.0, 1.0)   # off-scale tau <= 1 annotated
h = np.array([cnt[vals == e + 0.5].sum() if (vals == e + 0.5).any() else 0 for e in edges[:-1]]) / len(tm)
ax.stairs(h, edges, color=BLUE[L8], lw=1.3, zorder=3, label=r'optimum, $\varepsilon=2^{-8}$, $3L=24$')
off = cnt[vals < -10].sum()
ax.annotate('%.1f%%' % (100 * off / len(tm)) + r' at $\tau\leq1$' + '\n(off scale)', xy=(-10.45, 0.006),
            xytext=(-10.25, 0.105), fontsize=7, color=INK, ha='left', va='bottom',
            arrowprops=dict(arrowstyle='->', lw=0.6, color=INK, shrinkA=1, shrinkB=0))
ax.set_xlim(-10.5, 8.5); ax.set_ylim(0, 0.56)
ax.set_xticks(range(-9, 9, 3))
ax.set_xlabel(r'$T$-count $-\;3L$')
ax.set_ylabel('fraction of grid angles', labelpad=1)
ax.legend(loc='upper left', borderaxespad=0.3, labelspacing=0.4)

for ax, lab in zip(axs, ('(a)', '(b)', '(c)')):
    ax.text(-0.2, 1.025, lab, transform=ax.transAxes, ha='left', va='bottom', fontsize=9)

fig.savefig('fig_typical.pdf', metadata={'Creator': None, 'Producer': None, 'CreationDate': None})
print('wrote fig_typical.pdf')
