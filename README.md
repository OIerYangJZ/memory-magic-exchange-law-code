# A Memory–Magic Exchange Law in Streaming Clifford+T Compilation

Source, data and scripts for the manuscript submitted to PRX Quantum (October 2026). This repository
is a snapshot of the authors' working repository taken on 2026-10-15. Every number, table and figure
in the paper is produced by a script in `scripts/` from an input in `data/`, and this file says which.

The manuscript is `main.tex` (REVTeX 4.2, PRX Quantum style): main text, appendices with all
proofs of the main-text theorems, and references. The Supplemental Material is `supplement.tex`
(extended numerics, side information, model dependence and batching, the law under mixing).
The two files cross-reference each other through `xr-hyper` (labels of the other file carry
the prefix `S:` or `M:`), so build them in the same directory in the order
`pdflatex main && pdflatex supplement && pdflatex main && pdflatex supplement && pdflatex main`.

**arXiv.** [arXiv:2609.37368](https://arxiv.org/abs/2609.37368). Version 1 proves the unconditional rate
17/7; version 2 and the submitted manuscript here prove every rate below 5/2.

**Status.** Not yet peer reviewed. All results are proven except Theorem 8 and Corollary S3, which are
conditional on Conjecture 1, and the achievable rates of Corollary 5 and Proposition S2, which
are conditional on the synthesis hypotheses stated there (Eq. (13) and Eq. (S11)). Conjecture 1 is the volume law for
every frame up to a subexponential factor `2^{o(tau)}`; at Clifford cosets it is Theorem 6. Earlier versions stated it with fixed
constants `c0 + c1 tau + c 2^tau eps^2`, which the exact enumeration and a heuristic count make implausible (the required `c`
reaches 626.3 at `L = 21`, `torus_tube_enum.py`, Table S3, and `research/conj1_c/`), so no number in the paper depends on Conjecture 1.

## Environment

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
```

`numpy` and `mpmath` carry everything except the figures (`matplotlib`), the
Ross–Selinger synthesis and the SMT check. `gridsynth_run.py` and `run_mixed_achievable.py` need
`pygridsynth` (the latter can also call the Haskell `gridsynth` binary, `--backend cli`, whose accuracy
convention has not been checked here); `research/uniform52/verify_z3.py`
needs `z3-solver`, and `research/uniform52/check_cvc5.py` (the same SMT formula under a second solver) needs
`cvc5`. All of them are in `requirements.txt`. The C++ example of Remark S2,
`research/conditional3/ballpts.cpp`, needs a C++17 compiler; the build lines (with optional
OpenMP, including macOS with Homebrew `libomp`) are at the top of the file.

`research/` holds the working notes, exact certificates and independent checks behind the
appendices; `research/README.md` says which of them the paper uses and which are historical.

Run scripts from the repository root:

```bash
.venv/bin/python scripts/<name>.py
```

The one exception is `scripts/enum2.py`, which reads and writes its pickle relative to the
current directory and must be run from inside `data/`.

## What produces what

| Paper | Script | Input / output in `data/` |
|---|---|---|
| Fig. 1 (schematic) | — (drawn by hand, `fig0_schematic.pdf`) | — |
| Table S1, Fig. 2 (coset counts to `t = 22`) | `enum_arith.py`, `arith_analysis.py`, `fig1_fig.py` | `enum_arith.pkl`, `enum_res2.pkl` |
| Table S2 (low-cost words per frame) | `lowcost_counts.py` | `lowcost_counts.json` |
| Tables I and II (the bounds at `eps = 1e-10`, including Corollary 4, the finite form of Theorem 3: 37.40) | `thm_numbers.py` | `gridsynth_tau.json`, `gridsynth_tau_dproj_eps.json` (read for the achievable row) |
| Fig. 3(a) (exact optima, both error budgets) | `frontier_eps_half.py`, `make_fig2.py` | `taumin_eps_half.json`, `frontier_L{5,6}_eps2.json` |
| Fig. 3(b), Tables I and II (Ross–Selinger at chemistry scale) | `gridsynth_run.py` (20 000 angles at `d_proj <= eps/2`, the certified share accuracy: 102.32), then `gridsynth_run.py --dproj 1e-10 --out gridsynth_tau_dproj_eps.json` (the same angles at `d_proj <= eps`, the calibration: 99.28), then `make_fig2.py` | `gridsynth_tau.json`, `gridsynth_tau_dproj_eps.json`, `gridsynth_cache.json` |
| Sec. S1, exact `tau_min` for `L <= 6` | `taumin.py` | `taumin.pkl`, `taumin_exact.json` |
| Sec. S1, axis cosets: first scan (25 frames x 4 accuracies, `tau <= 15`) | `conj1_scan.py 15` | `conj1_scan_output.txt` |
| Sec. S1, axis cosets: second scan (272 frames x 5 accuracies, 1360 cells) | `scan2.py`, then `scan2_report.py` | `scan2.jsonl`, `scan2_summary.txt` |
| Sec. S1, required-`c` profiles to `t = 17` | `ht_profile.py`, `profile2.py` | `ht_profile_output.txt`, `profile2_{a,b}.txt` |
| Sec. S1, the identity-coset step at `L = 5` (232 words; a fixed-constant bound would need `c >= 163.9`) | `astra1.py` | `astra1_output.txt` |
| Sec. IV, Sec. S1, Table S3 (exact tube enumeration at the identity coset for every `eps` in `(2^-25, 1/8]` and every modulus; a fixed-constant bound would need `c >= 626.3`; arithmetic frames; the family `t = 3m - 1`) | `torus_tube_enum.py` (needs `python-flint`, `numba`): `validate --tmax 22`, `validate-frames --tmax 17`, `band --L <L> --K <16..18> --name identity`, `band ... --G1 <word> --G2 <word>`, `verify-cell`, `verify-words`, `chain-survey`, `chain-step`; about 80 min for the identity octaves `L = 3..24` on 10 cores, with a per-level cache of up to 16 GB (`TORUS_TUBE_CACHE`) | `torus_tube_identity_L*.json`, `torus_tube_{arith,axis,genuine}_t*_L*.json`, `torus_tube_validation*.json`, `torus_tube_chain_*.json`, logs `*_log.txt` |
| Sec. S1, "Larger T-counts" (near-onset scan with shifted grids to `tau = 85`, least-norm orbits to `t = 300`, algebraic families `Z^g - 1`) | `research/conj1_c/` (`c_survey.py`, `c_sample.py`, `c_search.py`, `c_orbit.py`; copy `scripts/torus_tube_enum.py` next to them; run on a server, see its README) | `research/conj1_c/*.jsonl`, `c_search.log` |
| Sec. IV, Sec. S1, the non-dyadic HT-axis step (69 words at `eps = 0.0265`, `Q = 29`, `c >= 119.6`) | `ht_axis_step.py` | `ht_axis_step_output.txt` |
| Sec. S4, all constants (Thm. S3, Cor. S2, Prop. S2) | `mixing_bounds.py` | `mixing_bounds_output.txt` |
| Sec. S4, Table S4, single-word `gridsynth` means at the mixing accuracies `d_proj <= sqrt(eps/2), sqrt(eps/4), sqrt(eps/10)` | `run_mixed_achievable.py --u-file data/u_20000.txt` (log in `mixed_achievable_log.txt`) | `u_20000.txt`, `mixed_cache.json`, `mixed_achievable_results.json` |
| Base enumeration (Matsumoto–Amano normal forms, resumable) | `cd data && ../.venv/bin/python ../scripts/enum2.py 0 22` | `enum_res2.pkl` |
| Thm. 9/10 (elementary tube bound, rate 11/5; proof in App. G): exact rational certificate | `research/cert/certificate.py 9/5` | `research/cert/certificate_output.txt` |
| Rate 2.23 with optimised box shape (remark after Thm. 10) | `research/cert/certificate_223.py 179/100 179/100 737/500 400/223` | `research/cert/certificate_223_output.txt` |
| Thm. 11/12 (height dichotomy, rate 17/7; proof in App. H, research notes D0–D7 in `research/dioph/DIOPH_NOTES.md`): exact rational certificate | `research/dioph/cert_dioph.py 28/17 28/17 157/100 11183/6800 400 100 12` (needs numpy; use `.venv/bin/python`) | `research/dioph/cert_17_7_output.txt` |
| Rate 12/5, same method, larger margin (research only, superseded by Thm. 4; needs the relaxed level-1 bound D1 of the notes) | `research/dioph/cert_dioph.py 5/3 5/3 31/20 41/25 400 100` | `research/dioph/cert_12_5_output.txt` |
| App. I (rational projections): exact certificate of the finite instance, rate 249/100 (proofs and review in `research/annulus/`) | `research/annulus/cert_proj.py 400/249 400/249 1192747/747000 319751/199200 400 100 12` | `research/annulus/cert_249_100_output.txt` |
| Thm. 4 (every rate below 5/2; proof in App. J, notes and review in `research/uniform52/`): continuum model and exact SMT check for all `0 < mu <= 1/10` | `research/uniform52/verify_z3.py 1/10 10/33` (needs `z3-solver`); model `research/uniform52/cont_model.py`; `verify_z3.py --facts` and `check_cvc5.py` (second solver) as in `research/uniform52/CERTIFICATE.md` | `research/uniform52/z3_outputs.txt`, `z3_facts_output.txt`, `cvc5_outputs.txt`, `CERTIFICATE.md` (hashes, versions, outputs) |
| Remark S2 (ball example: 126 words of T-count 25 within radius 0.0025 of g/\|g\|, g = ((3+2√2)+i+j−k)/2, all on one sphere section; notes and review in `research/conditional3/`) | `research/conditional3/ballpts.cpp` (`ballpts 25 5.828427 1 1 -1 0.0025`, C++17, build line in the file; prints the stored 126 words, possibly in another order) and `research/conditional3/review/check_sections.py` (needs numpy) | `research/conditional3/review/bp_25_0.0025.txt`, `research/conditional3/review/known_output.txt` |

`scan2_report.py` recomputes, from the stored `scan2.jsonl`, every count quoted in the
axis-coset paragraph of Sec. S1 — the 272/1360/20400 totals, the 17 cells above `c = 48`,
the per-accuracy maxima, and the linear-term check. `scan2.jsonl` was produced by an earlier
version of the frame keys in `conj1_scan.py`, whose sign convention for a rotation axis
depended on floating-point noise: its 272 frames contain the 213 distinct axes of words of
cost `<= 3`, six of them twice with opposite signs (identical counts), and a sample of forty
cost-4/5 axes. The keys are now canonical and the samples are taken from sorted lists, so a
fresh `scan2.py` run is deterministic across machines; it samples a different set of forty
cost-4/5 axes, and the numbers in the paper are those of `scan2_report.py` on the stored file.

## Quick reproduction

Cheap (seconds to a couple of minutes, all from cached data):

```bash
.venv/bin/python scripts/thm_numbers.py      # Tables I and II
.venv/bin/python scripts/mixing_bounds.py    # every constant of Sec. S4
.venv/bin/python scripts/scan2_report.py     # every count in the axis-coset paragraph
.venv/bin/python scripts/astra1.py           # the 232-word step (c >= 163.9 for a fixed-constant bound) at L = 5
.venv/bin/python scripts/torus_tube_enum.py verify-words --L 21 --Q 2919736 --eps 2.447978e-07 --tau 47   # the 5384-word cell (c >= 626.3 for a fixed-constant bound, ~20 s)
.venv/bin/python scripts/make_fig2.py        # Fig. 3
```

Expensive, and only needed to rebuild the cached data:

| | cost |
|---|---|
| `enum2.py 0 22` (from inside `data/`) | ~20 min and ~2.6 GB peak, single core; the `t = 22` layer alone is ~10 min. Each extra `t` doubles both. The times it prints are cumulative. |
| `scan2.py` | hours; resumable, appends to `data/scan2.jsonl` (see above) |
| `conj1_scan.py 15` | ~1 min |
| `lowcost_counts.py` | ~20 s |
| `ht_profile.py`, `profile2.py 0`, `profile2.py 1` | ~20 s each |
| `frontier_eps_half.py 6 24` | ~17 s, ~2 GB |
| `gridsynth_run.py` (both runs) | seconds with the stored `gridsynth_cache.json`; without it about 2 min per run on 10 cores |
| `run_mixed_achievable.py --u-file data/u_20000.txt` | seconds with the stored `mixed_cache.json`; without it about 3 min |

Times measured on a 10-core Apple-silicon laptop (Python 3.14, numpy 2.5).

## Determinism

Every randomised choice is seeded, so the outputs above are reproducible bit for bit
(timing lines excepted; the figures are written without matplotlib's version string, so
they are byte-identical too):

- Haar and perturbed frames: `numpy.random.default_rng(914)` in `conj1_scan.py`; `scan2.py`
  inherits the same generator by `exec`.
- The 20 000 grid angles of Fig. 3(b) and of the mixing measurement:
  `random.Random(20260903).randrange(7853981633)`, stored as `data/u_20000.txt`.
- `gridsynth_run.py`: `--seed 20260903` by default.

Two places where floating point used to decide a discrete outcome are now made exact:
`lowcost_counts.py` re-evaluates every word whose distance is within `1e-9` of `eps` with
50-digit `mpmath` arithmetic from its gate sequence and counts exact ties as within `eps`
(this changed one stored count, at `t = 16`, that no table uses, from 11772 to 11773); and
`conj1_scan.py`/`scan2.py` key rotation axes canonically and sample from sorted lists.

Synthesis is cached by content — `(angle, accuracy) -> T-count` in `gridsynth_cache.json`
and `mixed_cache.json` — so reruns cost nothing and do not depend on the synthesiser's own
tie-breaking.

## Conventions worth knowing before reading the scripts

- Accuracy is the up-to-phase operator norm, `d_proj(A,B) = inf_phi ||A - e^{i phi} B||`, which is
  what the paper calls `eps`. `pygridsynth` 2.0.0 (with `up_to_phase=True`) does **not** take `d_proj`:
  it accepts a candidate when `Re(conj(z) u) >= sqrt(1 - e^2/4)`, so the word it returns for parameter
  `e` has `d_proj <= e/2` (measured worst case 0.499 e to 0.4999 e). Both synthesis scripts therefore
  take the target `d_proj` (`--dproj`) and call pygridsynth with `e = 2 d_proj`, and each run
  re-synthesizes a subsample and records the worst `d_proj` achieved (`worst_dproj`, which must not
  exceed the target). Earlier versions passed `d_proj` itself, so their "eps/2" numbers (105.34, and
  the mixing values 53.43 and 54.96) were really at `d_proj <= eps/4` and `sqrt(eps/8)`, `sqrt(eps/16)`.
- `L := log2(1/eps)`, and the tuned modulus is `Q_eps = floor(pi / (2 arcsin 2 eps))`.
- The frontier slope is `E_u tau / log2 K` with `u` uniform on all of `Z_Q`, the zero share
  costing nothing — not the mean over nonzero shares, which is larger by `Q/(Q-1)`.
- Fig. 3(a) reports two error budgets. The certified `r = 2` point of Theorem 5 synthesizes
  each committed share at `eps/2`; evaluating at `eps` instead is a single-rotation
  calibration and not a point of the family. Both sequences approach 3 (calibration 2.478,
  2.721, 3.020; certified family 3.333, 3.334, 3.113, the last at `eps = 1e-10` with gridsynth).

`data/enum_res2.pkl.bak-t19` is the `t <= 19` enumeration, kept for comparison.

## License

Code is under the MIT License; data, manuscript sources, figures and notes are under CC BY 4.0.
See `LICENSE` for which files fall under which.
