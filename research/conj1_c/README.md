# How large must the constant c of Conjecture 1 be? (2026-10-12)

All runs on `school-server` (`~/mmx_claude/csurvey`), not on the laptop. Requirements are for `(c0, c1) = (8, 2)`.
The scripts import `scripts/torus_tube_enum.py` (copy it next to them to run).

## Files

- `c_survey.py`: for each τ, all words of minimal T-count ≤ τ within `eps_hi = sqrt(64/2^τ)` of the torus. It gives the exact tube requirement (sup over ε, independent of the modulus). It also gives a lower bound on the grid requirement for the frames `(I, R_z(shift))`: the shift is optimised exactly, and Q is taken near `Q_eps` plus 400 random moduli in `[Q_eps/2, Q_eps]`.
  - Validated against `data/typical_enum.json` (every cell with ≤ 20000 words, L = 4..12, t ≤ 28).
  - It reproduces the tube column of `torus_tube_enum.py` (929.7 at τ = 23, 980.8 at τ = 47) and the 626.3 cell.
  - The features are computed in floating point from coordinates of size 2^(t/4), so results are reliable only up to τ ≈ 85. Beyond that, use `c_sample.py`/`c_orbit.py`, which work in mpmath.
  - Output: `c_survey_X64.jsonl` (τ = 20..~190; trust τ ≤ 85).
- `c_sample.py`: orbits of least transverse norm at each level t. Here y2 = (1+√2)^k (1−ω)^j with j = 0, 1 and the two smallest admissible k (x ≲ 12).
  - Each orbit is built exactly (y1 over all solutions of y1·ȳ1 = m = 2n_t − y2·ȳ2, with the congruence y1 ≡ ω^a y2 mod √2). Angles are in mpmath.
  - Output: `c_sample.jsonl`, t = 20..~308. It stalled on hard factorizations: flint's C factoring ignores SIGALRM.
- `c_search.py`, `c_orbit.py`: algebraic families Y = Z^g, Z = √2^p (1+√2)^q, with Y − 1 = ∏ Φ_d(Z).
  - The x-position is |Z'|^g / 4, which stays bounded when q/p sits just below log2 / (2 log(1+√2)) = 0.39322.
  - The prime divisors of Φ_d(Z) with 8 | d are ≡ 1 mod 8, hence split in Q(ω).
  - Output: `c_search.log` (p ≤ ~200).
- `oneoff47.py`: the 626.3 cell with the shift optimised. It gives 626.3, so the shift brings no gain there.

## Results

- **Largest requirement found:** still 626.3. It sits at the identity coset: τ = 47, ε = 2.447978e-7, Q = 2919736, 5384 words, certified with gate strings by `torus_tube_enum.py verify-words`.
  - Shifted grids (frames (I, R_z(φ))) give at most 533.7 (τ = 23).
  - Least-norm orbits up to t ≈ 300 give at most 520 (t = 115).
- **Tube (modulus-free) requirements**, an upper bound for every modulus and shift at the same (τ, ε): at most 980.8 for τ ≤ 85 near the onset (x ≤ 64).
  - The (3,1) family at t = 94 gives 826, at x = 40.
  - The (191,75) family at t = 380 gives 878, at x = 0.30.
- **The linear term matters at large τ.** An orbit must exceed c0 + 2τ before it costs anything in c. Along the sampled least-norm orbits, max(16r − 2t) drops from 1956 (t < 100) to 796 (t in [100,200)) to 110 (t in [200,300)). The representation numbers r(m) there stay at 2–64.
- **The share of m with r(m) > 0** falls as (log N)^(−1/2): 0.24, 0.145, 0.074 per hundred levels. This matches the model "m is composed only of split primes".

## Heuristic

Treat r(m) as random, using the Sathe–Selberg count of integers built from split primes: P(r > 0, ω = k) ≈ (log N)^(−1) λ^(k−1)/(k−1)!, with λ = ½ log log N.

- **Random orbits.**
  - The expected number of violations per level of c0 + c1τ + c·x decays only like τ^(−0.09)/c.
  - So the sum over levels diverges: no fixed (c0, c1, c) should hold for all τ.
  - Violations of c = 4096 should be very sparse, around one per several thousand levels at τ in the hundreds.
- **Polynomial additive terms.** An additive term c1·τ^2 instead of c1·τ makes the random-orbit sum converge.
- **Cyclotomic families defeat any polynomial term.**
  - Take Z = √2^p (1+√2)^q with |Z'|^g bounded (good rational approximations q/p, g a power of 2 up to p) and Z ± 1 norms.
  - Then the representation number grows like exp(c (log τ)^2).
  - Only a subexponential additive term such as 2^(o(τ)), or the factor R(2^(τ+2)) of Theorem 2, survives.
  - The only families with clean Z ± 1 found for p ≤ 200 are (3,1) and (191,75). The first one with bounded x is (191,75), at t = 380, 762, 1526, …; beyond g = 2 its norms are too large to factor.

## Consequence for the paper (done in main.tex/supplement.tex)

Conjecture 1 is now the volume law for every frame up to a subexponential factor,
`#{W : t_min(W) <= tau, W in T_eps(C)} <= 2^{psi(tau)} (1 + 2^tau eps^2)`, `psi(tau) = o(tau)` (tube form). At Clifford cosets it
is Theorem 2 (`2^psi = 785 R(2^{tau+2})`). Theorem 8 and Corollary S3 now give every rate below 3 (per coarse bit under mixing) with
non-explicit constants, plus quantization at cost `(2 - nu) L` with an `o(L)` allowance. Lemma 8 (cost-weighted entropy), the grid-cover
lemma of the Supplement, eq. (Hcons), the conditional row of Table I and the rate-three curve of Fig. 3(b) are gone.
