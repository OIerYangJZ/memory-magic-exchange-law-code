# Certificate record for Theorem 4 (every rate below 5/2), Appendix J

The continuum claim C(a0, T) of Appendix J is checked as the unsatisfiability of its negation, a
quantifier-free formula of linear real arithmetic, in exact rational arithmetic. Appendix J states the
formula in full; `cont_model.py` implements it over a generic number type and `verify_z3.py` builds it as
z3 terms.

## Files (SHA-256)

| file | SHA-256 |
|---|---|
| `cont_model.py` | `1d284c62b9f21780dec38d0f77ede6daf9e4e17abb78837ff7d3622525e77201` |
| `verify_z3.py` | `912a802e8cf5781f71eaf6b9a420e7d498b182f81e24c1b2296dbf74def3caeb` |
| `check_cvc5.py` | `f7985d076bd28d9573ded9e88e6b648cc9a5c774dbe8b8a61e3bea8c98580ce7` |
| `negC_mu1-10_c10-33.smt2` | `3cba94bba6633c36b67a1728e01dc043dc421dd76ada67da91ad18fee6037003` |
| `negC_mu1-10_c3-10.smt2` | `26b1e66e5ae6c87ced7a755fe7dc244741dbecc562556638fe1ab4592612dfcb` |

## Environment

Python 3.14.7, z3-solver 5.1.0, cvc5 1.4.2 (Python package). Both solvers decide QF_LRA exactly
over the rationals; nothing here is floating-point or interval arithmetic.

## Domain

mu in (0, mu0], g in (0, 201/100], y > 0, X >= 0, D in [l_eps, 1 - 2g] for crossing pieces. There is no
upper bound on y or X. (Versions before 2026-10-11 also imposed y <= 3 and X <= 6; removing them changes
no result, and `verify_z3.py --facts` checks why they are not needed.)

## Commands and complete outputs

From this directory:

```
python3 verify_z3.py 1/10 10/33 --smt2 negC_mu1-10_c10-33.smt2   # the theorem: unsat
python3 verify_z3.py 1/10 3/10  --smt2 negC_mu1-10_c3-10.smt2    # sharpness control: sat
python3 check_cvc5.py negC_mu1-10_c10-33.smt2 negC_mu1-10_c3-10.smt2
python3 verify_z3.py --facts
```

`z3_outputs.txt` (all z3 runs, including the controls near the sharp constant 10/33):

```
mu0=1/10 c=1/2: unsat  (0.1s)
VERIFIED: for all mu in (0, mu0], HIGH/LOW split closes at every class g with T = E1 + c*mu < a0

mu0=1/300 c=10/33: unsat  (0.1s)
VERIFIED: for all mu in (0, mu0], HIGH/LOW split closes at every class g with T = E1 + c*mu < a0

mu0=1/300 c=303/1000: sat  (0.1s)
{'mu': 1/300, 'g': 5579837/3510000, 'y': 6961501/5850000, 'X': 9175639/23400000, 'D': -76496743/35100000}

mu0=1/300 c=38/125: unsat  (0.1s)
VERIFIED: for all mu in (0, mu0], HIGH/LOW split closes at every class g with T = E1 + c*mu < a0

mu0=1/300 c=0: sat  (0.1s)
{'mu': 1/300, 'g': 64771/40800, 'y': 9101/7650, 'X': 24271/10200, 'D': -133133/61200}

mu0=1/10 c=10/33: unsat  (0.1s)
VERIFIED: for all mu in (0, mu0], HIGH/LOW split closes at every class g with T = E1 + c*mu < a0

mu0=1/10 c=3/10: sat  (0.0s)
{'mu': 1/10, 'g': 3007/2200, 'y': 537/550, 'X': 2773/4400, 'D': -1909/1100}
```

`cvc5_outputs.txt`:

```
cvc5 1.4.2
negC_mu1-10_c10-33.smt2: unsat  (0.35s)
negC_mu1-10_c3-10.smt2: sat  (0.22s)
```

`z3_facts_output.txt`:

```
z3 5.1.0
(i)  HIGH(g,y) > T with y >= 6/5: unsat  (unsat = fact holds)
(ii) tangent piece, y < 6/5, X >= 6, N2 or N3 != pm: unsat  (unsat = fact holds)
(ii) crossing piece, y < 6/5, X >= 6, N2 or N3 != pm: unsat  (unsat = fact holds)
```

A satisfying assignment for a sat run is a witness that the case analysis fails for that constant; the
unsat result for c = 10/33 on mu in (0, 1/10] is the statement used in Appendix J.
