"""
Validation of the DMRG solver (tbdmrg, block2) against exact diagonalisation and known results.

    python tests/validate_dmrg.py            (needs  python -m pip install block2; a few minutes)

1. Exact diagonalisation (tbmany) on clusters that ED can do: random hoppings with the full
   projected interaction (even and odd electron numbers), a bipartite cluster with a triplet
   ground state (Lieb), and the reduced model of the triangulene chain (3 units, 9 sites):
   energies, total spins, charges, <S_i.S_j>, spin-flip weights per site and per atom, the
   inelastic spectrum at a tip, and the charge excitations (tip spectrum A(w)).
2. Known results beyond ED: free electrons on 24 sites with random hoppings (U = 0, exact
   from the single-particle levels), the ground-state energy of the half-filled Hubbard chain
   (Lieb-Wu, U = 4t) from open chains of 12, 16 and 20 sites, and Lieb's theorem for a bipartite
   chain with two more A than B sites (18 sites, U = 2t): ground state S = 1.
3. Stability: different random starts give the same energies; the energies decrease and
   converge with the bond dimension; the child process (used by the GUI) gives the same
   result as a run in this process.
"""
import os
import sys
import time
import numpy as np
from scipy.integrate import quad
from scipy.special import j0, j1

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import tbcore as tb          # noqa: E402
import tbmany as tm          # noqa: E402
import tbdmrg as td          # noqa: E402

ok = True


def check(name, cond, info=""):
    global ok
    ok &= bool(cond)
    print(("PASS " if cond else "FAIL ") + name + ("   " + info if info else ""), flush=True)


def cluster(T, terms):
    Mu, Md = tm.ph_shift(terms, len(T)) if terms else (0 * T, 0 * T)
    return dict(T=T, Tup=T + Mu, Tdn=T + Md, terms=terms)


def chain(L, U):
    T = np.zeros((L, L))
    for i in range(L - 1):
        T[i, i + 1] = T[i + 1, i] = -1.0
    return dict(T=T, Tup=T.copy(), Tdn=T.copy(), terms=[(a, a, a, a, U) for a in range(L)])


def compare_with_ed(name, cl, ne, nst=10, nch=6, opts=None):
    rng = np.random.default_rng(11)
    Ns = len(cl["T"])
    t0 = time.time()
    ed = tm.solve_cluster(cl, n_elec=ne, nstates=nst, dense_limit=3000)
    t1 = time.time()
    dm = td.solve_cluster(cl, n_elec=ne, nstates=nst, opts=opts or dict(bond_dim=300))
    t2 = time.time()
    le, we = tm.level_weights(ed)
    ld, wd = tm.level_weights(dm)
    W = rng.normal(size=(9, Ns)) / 3
    errs = dict(E=np.abs(ed["E"] - dm["E"]).max(), S=np.abs(ed["S"] - dm["S"]).max(),
                n=np.abs(ed["n"] - dm["n"]).max(), C=np.abs(ed["C"] - dm["C"]).max())
    same = le == ld
    if same:
        errs["W_site"] = np.abs(we - wd).max() / we[1:].max()
        wa = max(np.abs(tm.analyse_transition(ed, L, W)["w_atom"] - tm.analyse_transition(dm, L, W)["w_atom"]).max()
                 for L in le[1:])
        errs["W_atom"] = wa / max(tm.analyse_transition(ed, L, W)["w_atom"].max() for L in le[1:])
        v = rng.normal(size=Ns)
        _, te, ge = tm.inelastic_amplitudes(ed)
        _, tdd, gd = tm.inelastic_amplitudes(dm)
        a, b = tm.inelastic_weights_at(te, ge, v), tm.inelastic_weights_at(tdd, gd, v)
        errs["IETS"] = np.abs(a - b).max() / a[1:].max()
    ce = tm.charge_excitations(ed, cl, nstates=nch, dense_limit=3000)
    cd = tm.charge_excitations(dm, cl, nstates=nch)          # dispatches to tbdmrg
    v = rng.normal(size=Ns)
    Ee, We, _ = tm.tip_peaks(ce, v)
    Ed, Wd, _ = tm.tip_peaks(cd, v)
    grid = np.linspace(min(Ee.min(), Ed.min()) - 0.3, max(Ee.max(), Ed.max()) + 0.3, 4000)
    Ae, Ad = tm.broadened(Ee, We, grid, 0.01), tm.broadened(Ed, Wd, grid, 0.01)
    errs["peaks"] = np.abs(np.sort(Ee) - np.sort(Ed)).max()
    errs["A(w)"] = np.abs(Ae - Ad).max() / Ae.max()
    good = (same and errs["E"] < 1e-8 and errs["S"] < 1e-5 and errs["n"] < 1e-4 and errs["C"] < 1e-4
            and errs["W_site"] < 1e-3 and errs["W_atom"] < 1e-3 and errs["IETS"] < 1e-3
            and errs["peaks"] < 1e-8 and errs["A(w)"] < 1e-3)
    check(f"DMRG = ED, {name} ({Ns} sites, {ne} e, S0 = {ed['S'][0]:.1f}; ED {t1 - t0:.0f} s, DMRG {t2 - t1:.0f} s)", good,
          ", ".join(f"{k} {v:.1e}" for k, v in errs.items()) + ("" if same else "  levels differ"))
    return ed, dm


def main():
    ok_b2, msg = td.available()
    if not ok_b2:
        print("skip: " + msg)
        return
    print(f"block2 {msg}")

    # ---------------------------------------------------------------- 1. against ED
    rng = np.random.default_rng(7)
    Ns = 7
    T = rng.normal(size=(Ns, Ns)) * 0.4
    T = (T + T.T) / 2
    Wq, _ = np.linalg.qr(rng.normal(size=(16, Ns)))
    full = cluster(T, tm.interaction_terms(Wq, 2.5, mode="full", thr=0.0))
    compare_with_ed("random hoppings, full projected interaction, even N", full, 6)
    compare_with_ed("random hoppings, full projected interaction, odd N", full, 7)
    T = np.zeros((6, 6))
    for a, b in [(0, 1), (0, 2), (0, 3), (4, 3), (4, 5)]:
        T[a, b] = T[b, a] = -1.0
    T += 0.05 * np.diag(rng.normal(size=6))
    compare_with_ed("bipartite cluster N_A = 2, N_B = 4: triplet ground state, triplet -> triplet weights from S_z = 1",
                    cluster(T, [(a, a, a, a, 2.0) for a in range(6)]), 6, nst=12)

    st = tb.load_structure(os.path.join(os.path.dirname(HERE), "examples", "triangulene_chain.cdxml"))
    H = tb.hoppings(st)
    near = lambda p: int(np.argmin(np.linalg.norm(st.xy - p, axis=1)))
    pos = {"S1": [(11.8, 13.1), (14.2, 13.1)], "S2": [(13.0, 3.9)], "S3": [(13.0, -0.3)],
           "S4": [(4.3, 0.4)], "S5": [(4.3, -3.9)], "S6": [(3.1, -13.1), (5.5, -13.1)]}
    text, _ = tb.build_model(H, st, [(k, [near(p) for p in v], None) for k, v in pos.items()])
    m = tb.parse_eff(text, st)
    r = tb.fit(H, st, m)
    w, _, _ = tm.wannier_orbitals(H, st, m)
    ipr, _ = tm.orbital_spread(w)
    cl9 = tm.build_cluster(H, st, m, r["params"], ncells=2, periodic=False, U=3.0, EF=r["EF"],
                           U_override={n: 3.0 * ipr[k] for k, n in enumerate(m.sites)}, keep_last=["S4", "S5", "S6"])
    n9, _ = tm.neutral_filling(m, r, 2, len(cl9["T"]))
    ed9, dm9 = compare_with_ed("reduced model of the triangulene chain, 3 units", cl9, n9, nst=10, nch=6)

    # ---------------------------------------------------------------- 2. known results beyond ED
    L = 24
    rng = np.random.default_rng(1)
    dist = np.abs(np.subtract.outer(np.arange(L), np.arange(L)))
    T = -rng.uniform(0.5, 1.5, size=(L, L)) * (dist == 1) + rng.normal(scale=0.2, size=(L, L)) * (dist == 2)
    T = (T + T.T) / 2 + np.diag(0.8 * (-1) ** np.arange(L))      # random hoppings, staggered potential
    eps = np.linalg.eigvalsh(T)
    t0 = time.time()
    rf = td.solve_cluster(cluster(T, []), n_elec=L, nstates=3, opts=dict(bond_dim=200))
    ex = 2 * eps[:L // 2].sum()
    ex1 = ex + eps[L // 2] - eps[L // 2 - 1]
    check(f"free electrons, 24 sites, random 1st and 2nd neighbour hoppings (dimension 7e12): ground state and the "
          f"degenerate singlet + triplet of the first excitation = sums of single-particle levels ({time.time() - t0:.0f} s)",
          abs(rf["E"][0] - ex) < 1e-8 and np.abs(rf["E"][1:] - ex1).max() < 1e-8
          and sorted(np.round(rf["S"][1:], 6)) == [0.0, 1.0],
          f"errors {rf['E'][0] - ex:.1e}, {np.abs(rf['E'][1:] - ex1).max():.1e} eV; spins {np.round(rf['S'], 6)}")

    U = 4.0
    with np.errstate(over="ignore"):
        eLW = -4 * quad(lambda x: j0(x) * j1(x) / (x * (1 + np.exp(min(x * U / 2, 700.0)))), 0, np.inf, limit=500)[0]
    Ls, El = [12, 16, 20], []
    t0 = time.time()
    for L in Ls:
        El.append(td.solve_cluster(chain(L, U), n_elec=L, nstates=1, opts=dict(bond_dim=200))["E"][0])
    x = np.array(Ls, float)
    e_inf = np.linalg.solve(np.c_[x, np.ones(3), 1 / x], El)[0]
    check(f"half-filled Hubbard chain U = 4t: E(L) = e L + b + c/L from open chains of 12, 16, 20 sites gives the "
          f"Lieb-Wu energy per site e = {eLW:.6f} t ({time.time() - t0:.0f} s)", abs(e_inf - eLW) < 1e-4,
          f"e = {e_inf:.6f}, difference {e_inf - eLW:.1e}")

    # bipartite stub chain: backbone of 16 sites (8 A, 8 B) with two dangling A sites on B sites
    Lb = 16
    T = np.zeros((Lb + 2, Lb + 2))
    for i in range(Lb - 1):
        T[i, i + 1] = T[i + 1, i] = -1.0
    for d, b_ in ((Lb, 5), (Lb + 1, 11)):
        T[d, b_] = T[b_, d] = -1.0
    t0 = time.time()
    rt = td.solve_cluster(cluster(T, [(a, a, a, a, 2.0) for a in range(Lb + 2)]), n_elec=Lb + 2, nstates=2,
                          opts=dict(bond_dim=200))
    dwt = max(i["discarded"] for i in rt["dmrg_info"])
    check(f"Lieb's theorem beyond ED: bipartite stub chain, 18 sites, N_A - N_B = 2, U = 2t (dimension 2.4e9): "
          f"ground state S = 1 ({time.time() - t0:.0f} s)", abs(rt["S"][0] - 1) < 1e-6 and dwt < 1e-6,
          f"S = {np.round(rt['S'], 6)}, discarded weight {dwt:.0e}")

    # ---------------------------------------------------------------- 3. stability
    ra = td.solve_cluster(cl9, n_elec=n9, nstates=6, opts=dict(bond_dim=300, seed=1))
    rb = td.solve_cluster(cl9, n_elec=n9, nstates=6, opts=dict(bond_dim=300, seed=987))
    check("two random starts (seeds 1 and 987) give the same states", np.abs(ra["E"] - rb["E"]).max() < 1e-9
          and np.abs(ra["S"] - rb["S"]).max() < 1e-6, f"max dE {np.abs(ra['E'] - rb['E']).max():.1e} eV")
    Es, dws = [], []
    for M in (16, 32, 64, 128):
        rr = td.solve_cluster(chain(16, 4.0), n_elec=16, nstates=3, opts=dict(bond_dim=M))
        Es.append(rr["E"])
        dws.append(max(i["discarded"] for i in rr["dmrg_info"]))
    Es = np.array(Es)
    dif = np.abs(Es[:-1] - Es[-1]).max(axis=1)
    check("bond dimension 16, 32, 64, 128 (Hubbard chain, 16 sites, 3 states): energies decrease and converge, "
          "discarded weight drops", np.all(np.diff(Es, axis=0) < 1e-9) and np.all(np.diff(dif) < 0) and dws[-1] < dws[0],
          "error vs M = 128: " + ", ".join(f"{d:.1e}" for d in dif) + " eV; discarded " +
          ", ".join(f"{d:.0e}" for d in dws))
    rp = td.solve_cluster(cl9, n_elec=n9, nstates=4, opts=dict(bond_dim=300), inprocess=True)
    check("run in this process = run in a separate process (as in the GUI and above)",
          np.abs(rp["E"] - dm9["E"][:4]).max() < 1e-9 and np.abs(rp["C"] - dm9["C"][:4]).max() < 1e-4)

    print("\nALL PASSED" if ok else "\nSOME CHECKS FAILED")


if __name__ == "__main__":
    main()
