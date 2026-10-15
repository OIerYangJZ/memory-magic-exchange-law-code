"""Exact (z3, linear real arithmetic) verification of the continuum inequalities near the 5/2 limit.

Claim: for all mu in (0, mu0], with a0 = a = 8/5 + mu, b = E1 = (8 - 2 a0)/3 and T = E1 + c*mu,
there is no (g, y) with 0 < g <= 201/100, y > 0 such that
      HIGH(g, y) > T   and   LOW(g, y) > T.
Since HIGH is nonincreasing in eta, this is equivalent to: for every g the least eta* with HIGH <= T
has LOW(g, y) <= T for all y <= eta*  (the HIGH/LOW split of App. F closes at every class g).
Adversary choices (rank, X, crossing depth D, piece) are existential; prover choices are explicit
(the minimum over the patch fibration and the candidate cells of cont_model.cost_upper), so the negated
claim is a quantifier-free LRA formula.  UNSAT = claim proved (exactly, over Q).

The variables are mu in (0, mu0], g in (0, 201/100], y > 0, X >= 0 and D in [l_eps, 1 - 2g], with no upper
bound on y or X (earlier versions also imposed y <= 3, X <= 6; the result does not change, and `--facts`
checks the two reasons why no bound is needed: HIGH > T forces y < 6/5, and for y < 6/5, X >= 6 the rank-2
and rank-3 routes reduce to pm).

  python3 verify_z3.py MU0 C                 # check the claim for T = E1 + C*mu, mu in (0, MU0]
  python3 verify_z3.py MU0 C --smt2 FILE     # also write the negated claim in SMT-LIB 2 (see check_cvc5.py)
  python3 verify_z3.py --facts               # the two facts above, for mu0 = 1/10, C = 10/33"""
import sys, time, z3
import cont_model as CM

def zmax(*xs):
    xs = [x if isinstance(x, z3.ExprRef) else z3.RealVal(x) for x in xs]
    r = xs[0]
    for x in xs[1:]: r = z3.If(x > r, x, r)
    return r
def zmin(*xs):
    xs = [x if isinstance(x, z3.ExprRef) else z3.RealVal(x) for x in xs]
    r = xs[0]
    for x in xs[1:]: r = z3.If(x < r, x, r)
    return r
CM.Num.mx = staticmethod(zmax); CM.Num.mn = staticmethod(zmin)
CM.Num.ite = staticmethod(lambda c, x, y: z3.If(c, x, y))
CM.Num.AND = staticmethod(lambda *cs: z3.And(*cs))
CM.Num.BIG = z3.RealVal(10**6)
CM.FR = lambda p, q=1: z3.Q(p, q)

def build(mu0, c):
    mu, g, y, X, D = z3.Reals('mu g y X D')
    a0 = z3.Q(8, 5) + mu
    E1 = (8 - 2 * a0) / 3
    T = E1 + c * mu
    P = dict(mu=mu, a0=a0, a=a0, b=E1, E1=E1, T=T)
    gam = zmin(g, a0)
    lde = zmax(1 - g - a0, 1 - 2 * a0)
    dom = [mu > 0, mu <= mu0, g > 0, g <= z3.Q(201, 100), y > 0, X >= 0]
    high_fail = CM.high(P, g, y) > T
    cases = []
    for tangent in (True, False):
        tot = CM.piece_total(P, g, y, D, tangent)
        (N2, P2), (N3, P3), _, _ = CM.routes(P, g, y, X, tot)
        v4 = zmax(0, 4 * y - 2 * gam) + tot
        fail = z3.Or(tot > T, v4 > T, z3.And(N3 > T, P3 > T), z3.And(N2 > T, P2 > T))
        if not tangent:
            fail = z3.And(D >= lde, D <= 1 - 2 * g, fail)
        cases.append(fail)
    return z3.And(*dom, high_fail, z3.Or(*cases)), (mu, g, y, X, D)

def facts(mu0=z3.Q(1, 10), c=z3.Q(10, 33)):
    """(i) HIGH(g, y) > T forces y < 6/5;  (ii) for y < 6/5 and X >= 6, N2 = N3 = pm on every piece."""
    mu, g, y, X, D = z3.Reals('mu g y X D')
    a0 = z3.Q(8, 5) + mu; E1 = (8 - 2 * a0) / 3; T = E1 + c * mu
    P = dict(mu=mu, a0=a0, a=a0, b=E1, E1=E1, T=T)
    base = [mu > 0, mu <= mu0, g > 0, g <= z3.Q(201, 100)]
    s = z3.Solver(); s.add(*base, y >= z3.Q(6, 5), CM.high(P, g, y) > T)
    print(f"(i)  HIGH(g,y) > T with y >= 6/5: {s.check()}  (unsat = fact holds)")
    for tangent in (True, False):
        tot = CM.piece_total(P, g, y, D, tangent)
        (N2, P2), (N3, P3), _, _ = CM.routes(P, g, y, X, tot)
        s = z3.Solver(); s.add(*base, y > 0, y < z3.Q(6, 5), X >= 6, z3.Or(N2 != tot, N3 != tot))
        print(f"(ii) {'tangent' if tangent else 'crossing'} piece, y < 6/5, X >= 6, N2 or N3 != pm: {s.check()}  (unsat = fact holds)")

if __name__ == "__main__":
    from fractions import Fraction as F
    if '--facts' in sys.argv:
        print(f"z3 {z3.get_version_string()}")
        facts(); sys.exit(0)
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    smt2 = sys.argv[sys.argv.index('--smt2') + 1] if '--smt2' in sys.argv else None
    if smt2: args.remove(smt2)
    mu0 = F(args[0]) if len(args) > 0 else F(1, 300)
    c = F(args[1]) if len(args) > 1 else F(1, 2)
    t = time.time()
    phi, vs = build(z3.Q(mu0.numerator, mu0.denominator), z3.Q(c.numerator, c.denominator))
    s = z3.Solver(); s.add(phi)
    if smt2:
        open(smt2, 'w').write('(set-logic QF_LRA)\n' + s.to_smt2())
    r = s.check()
    print(f"mu0={mu0} c={c}: {r}  ({time.time()-t:.1f}s)", flush=True)
    if r == z3.sat:
        m = s.model(); print({str(v): m.eval(v) for v in vs})
    elif r == z3.unsat:
        print("VERIFIED: for all mu in (0, mu0], HIGH/LOW split closes at every class g with T = E1 + c*mu < a0")
