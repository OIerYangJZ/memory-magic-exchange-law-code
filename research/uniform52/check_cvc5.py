"""Second, independent solver for the SMT-LIB 2 export of verify_z3.py (Appendix J).

  python3 verify_z3.py 1/10 10/33 --smt2 negC_mu1-10_c10-33.smt2    # z3: unsat
  python3 check_cvc5.py negC_mu1-10_c10-33.smt2                      # cvc5: unsat

Needs the `cvc5` Python package (pip install cvc5)."""
import sys, time
import cvc5

def check(path):
    tm = cvc5.TermManager(); s = cvc5.Solver(tm)
    parser = cvc5.InputParser(s)
    parser.setFileInput(cvc5.InputLanguage.SMT_LIB_2_6, path)
    sm = parser.getSymbolManager()
    t = time.time(); res = []
    while True:
        cmd = parser.nextCommand()
        if cmd.isNull():
            break
        out = cmd.invoke(s, sm)
        if out.strip():
            res.append(out.strip())
    return res, time.time() - t

if __name__ == "__main__":
    print(f"cvc5 {cvc5.__version__ if hasattr(cvc5, '__version__') else ''}".strip())
    for path in sys.argv[1:]:
        res, dt = check(path)
        print(f"{path}: {' '.join(res)}  ({dt:.2f}s)")
