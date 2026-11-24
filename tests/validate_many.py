"""
Validation of tbmany against independent codes and exact results.

    python tests/validate_many.py

Needs (optional, the corresponding checks are skipped otherwise):
  openfermion  (pip install openfermion)                        exact fermionic Hamiltonians
  hubbard      (pip install git+https://github.com/dipc-cc/hubbard.git)  mean-field Hubbard
"""
import os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import tbcore as tb, tbmany as tm

ok = True
def check(name, cond, info=""):
    global ok
    ok &= bool(cond)
    print(("PASS " if cond else "FAIL ") + name + ("   " + info if info else ""))

rng = np.random.default_rng(0)

# ---------------------------------------------------------------- exact results
for t, U in [(1.0, 4.0), (0.3, 2.0)]:
    r = tm.run_ed(np.array([[0, -t], [-t, 0]]), [(0, 0, 0, 0, U), (1, 1, 1, 1, U)], 2, nstates=6)
    check(f"Hubbard dimer t={t} U={U}", abs(r["E"][0] - (U - np.sqrt(U**2 + 16*t**2)) / 2) < 1e-12 and r["S"][0] < 1e-6)
h = tm.HeisenbergED(4)
check("Heisenberg 4-site ring E0 = -2J", abs(h.spectrum({(0, 1): 1, (1, 2): 1, (2, 3): 1, (0, 3): 1})[0][0] + 2) < 1e-12)
check("Heisenberg 4-site chain", abs(h.spectrum({(0, 1): 1, (1, 2): 1, (2, 3): 1})[0][0] + (3 + 2*np.sqrt(3))/4) < 1e-12)
n = 10
E0 = tm.HeisenbergED(n).spectrum({tuple(sorted((i, (i+1) % n))): 1.0 for i in range(n)}, with_spin=False)[0][0]
check("Heisenberg 10-site ring E0 = -4.515446", abs(E0 + 4.51544635) < 1e-6, f"{E0:.8f}")
T = rng.normal(size=(8, 8)); T = (T + T.T) / 2; terms = [(a, a, a, a, 2.0) for a in range(8)]
d, l = tm.run_ed(T, terms, 8, 10, dense_limit=10**6), tm.run_ed(T, terms, 8, 10, dense_limit=10)
check("Lanczos = dense diagonalisation", np.abs(d["E"] - l["E"]).max() < 1e-10)

for U in (0.5, 2.0, 8.0):
    T = np.zeros((4, 4)); T[0, 1:] = T[1:, 0] = -1.0            # star: 1 A site, 3 B sites
    r = tm.run_ed(T, [(a, a, a, a, U) for a in range(4)], 4, nstates=4, sz=2)
    r0 = tm.run_ed(T, [(a, a, a, a, U) for a in range(4)], 4, nstates=4)
    check(f"Lieb theorem (ED): star N_A=1, N_B=3, U={U}: ground state S = 1",
          abs(r0["S"][0] - 1) < 1e-6 and abs(r0["E"][0] - r["E"][0]) < 1e-9)

# ---------------------------------------------------------------- OpenFermion
try:
    from openfermion import FermionOperator, get_sparse_operator
    def of_spectrum(T, terms, Ns, nup, ndn):
        Hf = FermionOperator()
        for a in range(Ns):
            for b in range(Ns):
                if T[a, b]:
                    for s in (0, 1):
                        Hf += FermionOperator(((2*a+s, 1), (2*b+s, 0)), T[a, b])
        for a, b, c, dd, v in terms:
            Hf += FermionOperator(((2*a, 1), (2*b, 0), (2*c+1, 1), (2*dd+1, 0)), v)
        M = get_sparse_operator(Hf, n_qubits=2*Ns).toarray()
        idx = [k for k in range(4**Ns) if sum((k >> (2*Ns-1-m)) & 1 for m in range(0, 2*Ns, 2)) == nup
               and sum((k >> (2*Ns-1-m)) & 1 for m in range(1, 2*Ns, 2)) == ndn]
        return np.linalg.eigvalsh(M[np.ix_(idx, idx)])
    for Ns, nup, ndn in [(4, 2, 2), (5, 3, 2), (6, 3, 3)]:
        T = rng.normal(size=(Ns, Ns)); T = (T + T.T) / 2
        W, _ = np.linalg.qr(rng.normal(size=(30, Ns)))
        terms = tm.interaction_terms(W, 2.5, mode="full", thr=0.0)
        ed = tm.HubbardED(T, terms, nup, ndn)
        E = ed.solve(ed.dim)[0]
        dE = np.abs(E - of_spectrum(T, terms, Ns, nup, ndn)).max()
        check(f"ED = OpenFermion, full projected interaction, {Ns} sites ({nup},{ndn})", dE < 1e-10, f"max dE {dE:.1e}")
except ImportError:
    print("skip OpenFermion checks (not installed)")

# ---------------------------------------------------------------- hubbard package (DIPC)
try:
    import warnings; warnings.filterwarnings("ignore")
    import sisl
    from hubbard import HubbardHamiltonian, sp2, density
    def compare(st, periodic, nup, ndn, up, dn, nk, kT, U=3.0):
        g = sisl.Geometry(np.c_[st.xy, np.zeros(st.N)], atoms=sisl.Atom(6, R=1.6),
                          lattice=sisl.Lattice([st.T[0] if periodic else 60.0, 60.0, 20.0]))
        g.set_nsc([3 if periodic else 1, 1, 1])
        HH = HubbardHamiltonian(sp2(g, t1=2.7, t2=0.2, t3=0.18, spin="polarized"), U=U, q=(nup, ndn), nkpt=[nk, 1, 1], kT=kT)
        HH.set_polarization(list(up), dn=list(dn))
        HH.converge(density.calc_n, tol=1e-9, steps=1000)
        n0 = np.array([np.full(st.N, nup / st.N), np.full(st.N, ndn / st.N)])
        for a in up: n0[:, a] = [1, 0]
        for a in dn: n0[:, a] = [0, 1]
        H = tb.hoppings(st) if periodic else {0: tb.hoppings(st)[0]}
        r = tm.mean_field_hubbard(H, nup, ndn, U=U, nk=nk, kT=kT, n0=n0, periodic=periodic, tol=1e-9, maxiter=3000)
        return np.abs(np.array(HH.n) - r["n"]).max(), r["E"] - (HH.Etot + U * st.N / 2), r
    tri = tb.from_hexmap("..O..\n.O.O.\nO.O.O\n", "tri", finite=True); tri.T = np.zeros(2); tri.periodic = False
    col = tb.sublattice_colors(tri); maj = np.where(col == np.argmax(np.bincount(col)))[0]
    dn_, dE, r1 = compare(tri, False, 12, 10, maj[:3], [], 1, 1e-4)
    check("MFH = hubbard: [3]triangulene S=1", dn_ < 1e-6 and abs(dE) < 1e-6, f"dn {dn_:.1e}, dE {dE:.1e}")
    dn_, dE, r0 = compare(tri, False, 11, 11, maj[:2], maj[2:4], 1, 1e-4)
    check("MFH = hubbard: [3]triangulene S=0", dn_ < 1e-6 and abs(dE) < 1e-6, f"dn {dn_:.1e}, dE {dE:.1e}")
    check("Lieb: triangulene S=1 below S=0", r1["E"] < r0["E"], f"{1e3*(r0['E']-r1['E']):.1f} meV")
    st = tb.load_structure(os.path.join(os.path.dirname(HERE), "examples", "real_goblet_chain.hex"))
    near = lambda x, y: int(np.argmin(np.linalg.norm(st.xy - [x, y], axis=1)))
    Tu, Td = near(3.689, 1.42), near(12.298, -24.85)
    dn_, dE, _ = compare(st, True, 52, 52, [Tu], [Td], 24, 0.005)
    check("MFH = hubbard: periodic goblet chain, AFM", dn_ < 1e-6 and abs(dE) < 1e-6, f"dn {dn_:.1e}, dE {dE:.1e}")
except ImportError:
    print("skip hubbard-package checks (not installed)")

# ---------------------------------------------------------------- effective orbitals
st = tb.load_structure(os.path.join(os.path.dirname(HERE), "examples", "real_goblet_chain.hex"))
m = tb.parse_eff(open(os.path.join(os.path.dirname(HERE), "examples", "real_goblet_chain.eff")).read(), st)
H = tb.hoppings(st)
w, hR, _ = tm.wannier_orbitals(H, st, m)
Rs = sorted(w)
O = np.array([[sum(w[r][:, a] @ w[r][:, b] for r in Rs) for b in range(4)] for a in range(4)])
check("effective orbitals orthonormal", np.allclose(O, np.eye(4), atol=1e-6))

# ---------------------------------------------------------------- spin-flip transitions
for Ns, ne, sz in [(6, 6, None), (5, 5, None), (5, 4, None)]:
    T = rng.normal(size=(Ns, Ns)); T = (T + T.T) / 2
    Wr, _ = np.linalg.qr(rng.normal(size=(20, Ns)))
    r = tm.run_ed(T, tm.interaction_terms(Wr, 3.0, mode="full", thr=0.0), ne, nstates=10**4, dense_limit=10**5)
    du, dd = r["ed"].up.dim, r["ed"].dn.dim
    dS2 = max(abs(tm.total_spin_ladder(r["X"][:, k].reshape(du, dd), Ns, r["n_up"], r["n_dn"])[1] - r["C"][k].sum())
              for k in range(len(r["E"])))
    tot = sum(tm.spin_flip_weight(tm.transition_amplitudes(r["X"][:, f], r["X"][:, 0], Ns, r["n_up"], r["n_dn"]))
              for f in range(len(r["E"])))
    check(f"S^2 = S^-S^+ + Sz(Sz+1) and sum rule sum_f W_i = <S_i^2>, {Ns} sites {ne} e",
          dS2 < 1e-10 and np.abs(tot - np.diag(r["C"][0])).max() < 1e-10, f"{dS2:.1e}, {np.abs(tot - np.diag(r['C'][0])).max():.1e}")
lev, lw = tm.level_weights(r)
check("fast level weights = full transition analysis",
      max(np.abs(tm.analyse_transition(r, L)["w_site"] - lw[i]).max() for i, L in enumerate(lev)) < 1e-10)
try:
    from openfermion import FermionOperator, get_sparse_operator
    for Ns, ne in [(4, 4), (5, 5), (4, 3)]:
        T = rng.normal(size=(Ns, Ns)); T = (T + T.T) / 2
        Wr, _ = np.linalg.qr(rng.normal(size=(12, Ns)))
        terms = tm.interaction_terms(Wr, 3.0, mode="full", thr=0.0)
        Watoms = rng.normal(size=(5, Ns))
        nq = 2 * Ns
        Hf = FermionOperator()
        for a in range(Ns):
            for b in range(Ns):
                for sgm in (0, 1):
                    Hf += FermionOperator(((2*a+sgm, 1), (2*b+sgm, 0)), T[a, b])
        for a, b, c, dd_, v in terms:
            Hf += FermionOperator(((2*a, 1), (2*b, 0), (2*c+1, 1), (2*dd_+1, 0)), v)
        nop = sum((FermionOperator(((q, 1), (q, 0))) for q in range(nq)), FermionOperator())
        idx = np.where(np.abs(get_sparse_operator(nop, n_qubits=nq).diagonal().real - ne) < 1e-9)[0]
        E, V = np.linalg.eigh(get_sparse_operator(Hf, n_qubits=nq).toarray()[np.ix_(idx, idx)])
        def spin_op(coef, comp):          # sum_ab coef[a, b] S^comp_ab
            op = FermionOperator()
            for a in range(Ns):
                for b in range(Ns):
                    f = coef[a, b]
                    if f == 0:
                        continue
                    ua, da, ub, db = 2*a, 2*a+1, 2*b, 2*b+1
                    if comp == "z":
                        op += 0.5 * f * (FermionOperator(((ua, 1), (ub, 0))) - FermionOperator(((da, 1), (db, 0))))
                    elif comp == "x":
                        op += 0.5 * f * (FermionOperator(((ua, 1), (db, 0))) + FermionOperator(((da, 1), (ub, 0))))
                    else:
                        op += -0.5j * f * (FermionOperator(((ua, 1), (db, 0))) - FermionOperator(((da, 1), (ub, 0))))
            return get_sparse_operator(op, n_qubits=nq).toarray()[np.ix_(idx, idx)]
        lev_ = []
        for k in range(len(E)):
            if lev_ and abs(E[k] - E[lev_[-1][0]]) < 1e-8: lev_[-1].append(k)
            else: lev_.append([k])
        G = lev_[0]
        r = tm.run_ed(T, terms, ne, nstates=10**4, dense_limit=10**5)
        ours = {round(a_["dE"], 7): a_ for a_ in (tm.analyse_transition(r, L, W=Watoms) for L in tm.levels(r["E"], r["S"])[1:])}
        err = 0.0
        for L in lev_[1:6]:
            dE = E[L[0]] - E[G[0]]
            a_ = ours[min(ours, key=lambda x: abs(x - dE))]
            ws = np.array([sum(np.sum(np.abs(V[:, L].conj().T @ spin_op(np.diag(np.eye(Ns)[i]), c) @ V[:, G]) ** 2)
                               for c in "xyz") / len(G) for i in range(Ns)])
            wa = np.array([sum(np.sum(np.abs(V[:, L].conj().T @ spin_op(np.outer(Watoms[x], Watoms[x]), c) @ V[:, G]) ** 2)
                               for c in "xyz") / len(G) for x in range(len(Watoms))])
            err = max(err, abs(a_["dE"] - dE), np.abs(a_["w_site"] - ws).max(), np.abs(a_["w_atom"] - wa).max())
        check(f"spin-flip weights (sites and atoms) = OpenFermion, {Ns} sites {ne} e", err < 1e-10, f"max error {err:.1e}")
except ImportError:
    print("skip OpenFermion transition checks (not installed)")

# ---------------------------------------------------------------- spin-model fit: bond labelling
h4 = tm.HeisenbergED(4)
Jtrue = {(0, 1): 1.0, (1, 2): 0.15, (2, 3): 1.0, (0, 3): 0.4}
pairs4, groups4 = [(0, 1), (1, 2), (2, 3), (0, 3)], [0, 1, 0, 2]
E4, X4 = np.linalg.eigh(h4.matrix(Jtrue)); S4 = h4.spectrum(Jtrue)[1]
C4 = np.array([[[X4[:, k] @ h4.matrix({tuple(sorted((a, b))): 1.0}) @ X4[:, k] if a != b else 0.75 for b in range(4)]
                for a in range(4)] for k in range(6)])
Jfit, *_ = tm.fit_spin_model(E4[:6], S4[:6], 4, pairs4, groups4, C_ref=C4)
check("spin-model fit with correlations recovers the J's (not a relabelled solution)",
      np.allclose(Jfit, [1.0, 0.15, 0.4], atol=1e-5) and tm.fit_spin_model.corr_err < 1e-5, str(np.round(Jfit, 5)))

# the real case: the energies of the goblet chain (2 cells) are fitted exactly by a wrong
# assignment (singlets on the Tu-Tu bonds); the correlations must give the ED sign pattern
cl2 = tm.build_cluster(H, st, m, tb.fit(H, st, m)["params"], ncells=2, periodic=False, U=3.0,
                       EF=tb.fit(H, st, m)["EF"])
r2 = tm.solve_cluster(cl2, nstates=8)
pos = np.array([st.xy[a].mean(axis=0) for a in m.site_atom])
sp_sites, sp_pairs, sp_groups, _ = tm.spin_pairs(m, 2, False, ["Tu", "Td"], 1, pos, st.T, tb.symmetry_ops(st))
Cs = r2["C"][:6][:, sp_sites][:, :, sp_sites]
Jg = tm.fit_spin_model(r2["E"][:6], r2["S"][:6], 4, sp_pairs, sp_groups, C_ref=Cs)[0]
Jd = {}
for pr, g in zip(sp_pairs, sp_groups):
    Jd[pr] = Jd.get(pr, 0.0) + Jg[g]
_, Xg = np.linalg.eigh(h4.matrix(Jd))
signs_ok = all(np.sign(Xg[:, 0] @ h4.matrix({pr: 1.0}) @ Xg[:, 0]) == np.sign(Cs[0][pr]) for pr in sp_pairs
               if abs(Cs[0][pr]) > 0.1)
check("goblet chain spin fit: Heisenberg correlations have the ED signs (AFM Tu-Td, FM Tu-Tu)", signs_ok,
      "J (meV) = " + str(np.round(1e3 * Jg, 4)))

# ---------------------------------------------------------------- symmetry and filling
sym = tb.symmetry_ops(st)
check("real goblet chain: glide, mirror and C2 found", sorted((o[0], o[1]) for o in sym["ops"]) == [(-1, -1), (-1, 1), (1, -1)])
pos = np.array([st.xy[a].mean(axis=0) for a in m.site_atom])
_, _, _, labs = tm.spin_pairs(m, 2, False, ["Tu", "Td"], 1, pos, st.T, sym)
check("glide makes Tu-Tu and Td-Td one coupling", "Td-Td/Tu-Tu d = 17.2 A" in labs, str(labs))
used = {a for x in m.site_atom for a in x}
st1 = tb.remove_atoms(st, [st.xy[[i for i in range(st.N) if i not in used][40]]])
m1 = tb.parse_eff(open(os.path.join(os.path.dirname(HERE), "examples", "real_goblet_chain.eff")).read(), st1)
n1, per1 = tm.neutral_filling(m1, tb.fit(tb.hoppings(st1), st1, m1), 2)
check("neutral filling with an odd electron count: 3 per cell in 4 model bands", n1 == 6, f"{per1:.3f} per cell")

# ---------------------------------------------------------------- band crossings
pa = tb.from_hexmap("O.\n", "polyacene")
pa_eff = "\n".join([f"site {x} atom {i}" for i, x in enumerate("ABCD")] + [f"onsite {x} e" for x in "ABCD"] +
                    ["hop A B 0 t", "hop B C 0 t", "hop C D 0 t", "hop A B -1 t", "hop D C -1 t"])
Hp = tb.hoppings(pa)                                   # t1, t2, t3: the HOMO/LUMO crossing moves to k ~ 0.92 pi
mp = tb.parse_eff(pa_eff, pa)
dl, mk = tb.band_diagnostics(Hp, pa, mp, tb.fit(Hp, pa, mp))
check("polyacene: mirror-protected HOMO/LUMO crossing found (k ~ 0.92 pi), NN model crosses at pi instead",
      any(m_[0] == "cross" and abs(m_[1] / np.pi - 0.923) < 0.01 for m_ in mk) and any("cross at k = 1.000" in x for x in dl))
Hb = tb.hoppings(pa); Hb[0] = Hb[0].copy(); Hb[0][0, 0] += 0.1        # break the mirror: the crossing opens
Eb = np.array([np.linalg.eigvalsh(tb.Hk(Hb, k)) for k in np.linspace(0, np.pi, 2001)])
check("breaking the mirror turns the crossing into an avoided crossing", (Eb[:, 2] - Eb[:, 1]).min() > 0.05,
      f"gap {(Eb[:, 2] - Eb[:, 1]).min():.3f} eV")
pp = tb.load_structure(os.path.join(os.path.dirname(HERE), "examples", "polyparaphenylene.cdxml"))
Hpp = tb.hoppings(pp)
txt = open(os.path.join(os.path.dirname(HERE), "examples", "polyparaphenylene.eff")).read()
m_idx = tb.parse_eff(txt.replace("bands auto", ""), pp)
m_auto = tb.parse_eff(txt, pp)
d_idx = tb.band_diagnostics(Hpp, pp, m_idx, tb.fit(Hpp, pp, m_idx))[0]
d_auto = tb.band_diagnostics(Hpp, pp, m_auto, tb.fit(Hpp, pp, m_auto))[0]
check("polyparaphenylene: index window flagged (flat band inside), 'bands auto' clean",
      d_idx[0].startswith("WARNING") and not any(x.startswith("WARNING") for x in d_auto))
dg = tb.band_diagnostics(H, st, m, tb.fit(H, st, m))[1]
check("real goblet chain: glide makes the bands stick in pairs at k = pi", sum(1 for x in dg if x[0] == "zone") >= 2)

# ---------------------------------------------------------------- U from the mean field (tab 5)
pvg = tb.fit(H, st, m)["params"]
Utrue = np.array([0.8, 0.8, 0.3, 0.3])
syn = []
for nu_, nd_, n0_ in [(3, 1, [[1, 1, .5, .5], [0, 0, .5, .5]]), (2, 2, [[1, 0, .5, .5], [0, 1, .5, .5]])]:
    s_ = tm.model_mfh(m, pvg, Utrue, nu_, nd_, 24, 0.005, n0=np.array(n0_, float))
    syn.append(dict(ks=s_["ks"], E=[s_["bands"][0] + 0.1, s_["bands"][1] + 0.1], n_site=s_["n"],
                    m_site=s_["n"][0] - s_["n"][1], n_up_mod=nu_, n_dn_mod=nd_))
fu = tm.fit_U_to_mfh(m, pvg, syn, [0, 0, 1, 1], [0.3, 0.3], 0.005)
check("U fit recovers the U of a synthetic reduced-model mean field (0.8, 0.3 eV; offset 0.1 eV)",
      np.allclose(fu["U_group"], [0.8, 0.3], atol=1e-4) and np.allclose(fu["offsets"], 0.1, atol=1e-5),
      str(np.round(fu["U_group"], 5)))
cfg = tm.mfh_configurations(H, st, [m.site_atom[0], m.site_atom[1]], 3.0, 24, 0.005, which=("fm",))
rf = tm.mfh_reference(H, st, m, cfg["fm"], 3.0, 0.005)
check("atomistic FM mean field projected on the effective orbitals: 3 up / 1 down electrons in the model bands,"
      " moments on the heads", np.allclose(rf["n_mod"], [3, 1], atol=0.01) and rf["m_site"][:2].min() > 0.9,
      str(np.round(rf["n_mod"], 3)))

# ---------------------------------------------------------------- charge excitations (tab 6)
def _fake_cluster(T, terms):
    Mu, Md = tm.ph_shift(terms, len(T))
    return dict(Tup=T + Mu, Tdn=T + Md, terms=terms)
def _full_res(cl, Ns, ne):
    ed = tm.SpinHubbardED(cl["Tup"], cl["Tdn"], cl["terms"], (ne + ne % 2) // 2, ne // 2)
    E, X = ed.solve(ed.dim, 10**6)
    X = tm.clean_multiplets(E, X, Ns, ed.up.n, ed.dn.n)
    return dict(E=E, X=X, S=np.array([ed.total_spin(X[:, k])[0] for k in range(len(E))]), ed=ed,
                n_up=ed.up.n, n_dn=ed.dn.n)
for Ns, ne, U in [(6, 6, 2.0), (5, 5, 2.0), (6, 6, 0.0)]:
    T = rng.normal(size=(Ns, Ns)) * 0.5; T = (T + T.T) / 2
    cl_ = _fake_cluster(T, [(a, a, a, a, U) for a in range(Ns)] if U else [])
    r_ = _full_res(cl_, Ns, ne)
    ch = tm.charge_excitations(r_, cl_, nstates=10**4, dense_limit=10**6)
    g_ = tm.levels(r_["E"], r_["S"])[0]
    nd_ = np.mean([sum(r_["ed"].density(r_["X"][:, i])) for i in g_], axis=0)
    err = 0.0
    for a in range(Ns):
        Ew, Wt, kd = tm.tip_peaks(ch, np.eye(Ns)[a])
        err = max(err, abs(Wt[kd == "remove"].sum() - nd_[a]), abs(Wt[kd == "add"].sum() - (2 - nd_[a])))
    extra = ""
    if U == 0:
        eps = np.linalg.eigvalsh(cl_["Tup"])
        Ew, Wt, kd = tm.tip_peaks(ch, np.ones(Ns) / np.sqrt(Ns))
        a_ = np.unique(np.round(Ew[(kd == "add") & (Wt > 1e-9)], 8))
        err = max(err, np.abs(a_ - np.round(eps[ne // 2:], 8)).max())
        extra = ", U = 0: peaks at the single-particle levels"
    check(f"charge excitations {Ns} sites {ne} e U={U}: sum rules sum_f A = <n_a>, 2 - <n_a>{extra}", err < 1e-9,
          f"{err:.1e}")
try:
    from openfermion import FermionOperator, get_sparse_operator
    Ns, ne = 4, 4
    T = rng.normal(size=(Ns, Ns)) * 0.5; T = (T + T.T) / 2
    cl_ = _fake_cluster(T, [(a, a, a, a, 1.7) for a in range(Ns)])
    r_ = _full_res(cl_, Ns, ne)
    ch = tm.charge_excitations(r_, cl_, nstates=10**4, dense_limit=10**6)
    nq = 2 * Ns
    Hf = FermionOperator()
    for a in range(Ns):
        for b in range(Ns):
            Hf += FermionOperator(((2*a, 1), (2*b, 0)), cl_["Tup"][a, b])
            Hf += FermionOperator(((2*a+1, 1), (2*b+1, 0)), cl_["Tdn"][a, b])
    for a, b, c, d, v in cl_["terms"]:
        Hf += FermionOperator(((2*a, 1), (2*b, 0), (2*c+1, 1), (2*d+1, 0)), v)
    Hm = get_sparse_operator(Hf, n_qubits=nq).toarray()
    Nd = get_sparse_operator(sum((FermionOperator(((q, 1), (q, 0))) for q in range(nq)), FermionOperator()),
                             n_qubits=nq).diagonal().real
    i0, i1 = np.where(np.abs(Nd - ne) < 1e-9)[0], np.where(np.abs(Nd - ne - 1) < 1e-9)[0]
    E0s, V0 = np.linalg.eigh(Hm[np.ix_(i0, i0)])
    E1s, V1 = np.linalg.eigh(Hm[np.ix_(i1, i1)])
    G = np.abs(E0s - E0s[0]) < 1e-8
    v = rng.normal(size=Ns); v /= np.linalg.norm(v)
    A = np.zeros(len(E1s))
    for s_ in (0, 1):
        op = get_sparse_operator(sum((FermionOperator(((2*a + s_, 1),), v[a]) for a in range(Ns)), FermionOperator()),
                                 n_qubits=nq).toarray()[np.ix_(i1, i0)]
        A += np.sum(np.abs(V1.T.conj() @ op @ V0[:, G]) ** 2, axis=1) / G.sum()
    Ew, Wt, kd = tm.tip_peaks(ch, v)
    grid = np.linspace(-1, 6, 3000)
    ref = tm.broadened(E1s - E0s[0], A, grid, 0.05)
    ours = tm.broadened(Ew[kd == "add"], Wt[kd == "add"], grid, 0.05)
    check("tip spectrum (addition) = OpenFermion in the full Fock space, random tip orbital",
          np.abs(ref - ours).max() < 1e-8, f"max diff {np.abs(ref - ours).max():.1e}")
except ImportError:
    print("skip OpenFermion tip-spectrum check (not installed)")

# ---------------------------------------------------------------- structure input
ag = tb.load_structure(os.path.join(os.path.dirname(HERE), "examples", "7AGNR.hex"))
Eg = tb.bands(tb.hoppings(ag), 101)[1]
EFg, ng_ = tb.fermi(Eg, ag.n_electrons)
check("armchair hex map: 7-AGNR has 14 atoms per cell, T = 3a, a gap (3p+1 family)",
      ag.N == 14 and abs(ag.T[0] - 3 * 1.42) < 1e-6 and Eg[:, ng_].min() - Eg[:, ng_ - 1].max() > 0.8)
flake = tb.from_hexmap("armchair\nO.O.\n.O.O\nO.O.\n.O.O\nO.O.\n", "flake", finite=True)
cands = tb.period_from_repeats(flake.xy, return_all=True)
check("a short, wide drawing that repeats along x and y gives both periods, x first",
      len(cands) == 2 and abs(cands[0][1]) < 1e-6 and abs(abs(cands[1][0])) < 1e-6,
      str([np.round(c_, 2) for c_ in cands]))
terms = tb.symmetric_terminations(ag, 4)
check("7-AGNR, 4 units: symmetric cuts include the one ending on three-ring columns (56 atoms)",
      any(d["N"] == 56 for d in terms) and all(d["symmetry"] for d in terms), str([d["N"] for d in terms]))
fin = tb.finite_chain(ag, 4, [d for d in terms if d["N"] == 56][0]["offset"], True,
                      [d for d in terms if d["N"] == 56][0]["extra"])
mf_ = tm.mfh_scan({0: tb.hoppings(fin)[0]}, fin, 3.0, 1, 0.005)
d52 = [d for d in terms if d["N"] == 52][0]
fin52 = tb.finite_chain(ag, 4, d52["offset"], True, d52["extra"])
mf52 = tm.mfh_scan({0: tb.hoppings(fin52)[0]}, fin52, 3.0, 1, 0.005)
check("finite mean field without a period: the 7-AGNR piece with three-ring (zigzag) ends carries end "
      "moments, the one with two-ring ends does not (U = 3 eV)",
      all(o["sol"]["converged"] for o in mf_ + mf52) and mf_[0]["mmax"] > 0.1 and mf52[0]["mmax"] < 0.05,
      f"max |m| {mf_[0]['mmax']:.2f} / {mf52[0]['mmax']:.2f}")

# ---------------------------------------------------------------- odd chains (dimer cell)
EX = os.path.join(os.path.dirname(HERE), "examples")
ol = tb.load_structure(os.path.join(EX, "olympicene_chain.hex"))
check("olympicene chain: fused-ring units of 19 atoms linked by single bonds", tb.unit_sizes(ol) == {19})
t1, t2 = tb.symmetric_terminations(ol, 1), tb.symmetric_terminations(ol, 2)
check("symmetric ends of the olympicene chain keep whole units: 2 or 3 (n = 1), 4 or 5 (n = 2)",
      sorted((d["units"], d["N"]) for d in t1) == [(2, 38), (3, 57)]
      and sorted((d["units"], d["N"]) for d in t2) == [(4, 76), (5, 95)],
      str([(d["units"], d["N"]) for d in t1 + t2]))
mo = tb.parse_eff(open(os.path.join(EX, "olympicene_chain_hex.eff")).read(), ol)
Ho = tb.hoppings(ol)
fo = tb.fit(Ho, ol, mo)
full = tm.build_cluster(Ho, ol, mo, fo["params"], ncells=3, periodic=False, U=3.0, EF=fo["EF"])
odd = tm.build_cluster(Ho, ol, mo, fo["params"], ncells=3, periodic=False, U=3.0, EF=fo["EF"], keep_last=["A"])
k5 = odd["kept"]
check("partial last cell: 3 cells keeping A give A0 B0 A1 B1 A2, with the hoppings and U of the full cluster",
      odd["labels"] == ["A0", "B0", "A1", "B1", "A2"] and np.allclose(odd["T"], full["T"][np.ix_(k5, k5)])
      and np.allclose(odd["U_site"], full["U_site"][k5]))
n5, _ = tm.neutral_filling(mo, fo, 3, len(k5))
r5 = tm.solve_cluster(odd, n_elec=n5, nstates=10)
ref5 = tm.run_ed(odd["Tup"], odd["terms"], n5, nstates=10)
check("odd chain of 5 units: neutral = 5 electrons, S = 1/2 ground state, same spectrum as a direct ED",
      n5 == 5 and abs(r5["S"][0] - 0.5) < 1e-6 and np.allclose(r5["E"][:6], ref5["E"][:6]),
      f"S0 = {r5['S'][0]:.2f}")
sh = tm.cluster_atoms_shown(odd)
check("drawing of the odd chain: the last cell shows one whole olympicene (19 atoms)",
      sh[:2].all() and sh[2].sum() == 19, str(sh.sum(axis=1)))
posd = np.array([ol.xy[a].mean(axis=0) for a in mo.site_atom])
sp5 = tm.spin_pairs(mo, 3, False, ["A", "B"], 1, posd, ol.T, tb.symmetry_ops(ol), kept=k5)
check("spin pairs of the odd chain use the 5 kept sites only", sp5[0] == [0, 1, 2, 3, 4]
      and max(max(p_) for p_ in sp5[1]) == 4, str(sp5[3]))
ch5 = tm.charge_excitations(r5, odd, nstates=200)
I5 = np.eye(len(k5))
ok_d = len([c for c in ch5 if c["kind"] == "add"]) == 2
for c in ch5:
    for j in range(len(c["w"])):
        ref = c["mult"] * (c["D"][:, j, :] ** 2).mean(axis=0)
        ok_d &= np.allclose(tm.dyson_map(ch5, c["kind"], j, I5, spin=c["spin"]), ref)
check("odd chain: both spins computed, the Dyson map of a spin-down state uses its own sector", ok_d)
try:
    tm.cluster_keep(mo, 3, ["A"], periodic=True)
    check("a partial last cell on a ring is refused", False)
except ValueError:
    check("a partial last cell on a ring is refused", True)

# ---------------------------------------------------------------- singlet-triplet gap from the mean field
import scipy.sparse as sps
from scipy.spatial import cKDTree
def _fock(Ns):
    a_ = sps.csr_matrix(np.array([[0.0, 1.0], [0.0, 0.0]])); Z_, I_ = sps.diags([1.0, -1.0]), sps.identity(2)
    ops = []
    for mo in range(2 * Ns):
        out = None
        for o in [Z_] * mo + [a_] + [I_] * (2 * Ns - mo - 1):
            out = o if out is None else sps.kron(out, o, format="csr")
        ops.append(out.tocsr())
    return ops
Hb = np.zeros((6, 6))
for i in range(6):
    Hb[i, (i + 1) % 6] = Hb[(i + 1) % 6, i] = -1.0
c_ = _fock(6); cu_, cd_ = c_[:6], c_[6:]
num_ = lambda o: o.T @ o
S2op = (lambda X, Y, Zs: X @ X + Y @ Y + Zs @ Zs)(sum(0.5 * (cu_[i].T @ cd_[i] + cd_[i].T @ cu_[i]) for i in range(6)),
                                                 sum(-0.5j * (cu_[i].T @ cd_[i] - cd_[i].T @ cu_[i]) for i in range(6)),
                                                 sum(0.5 * (num_(cu_[i]) - num_(cd_[i])) for i in range(6)))
err = 0.0
for nu_, nd_, n0_ in ((3, 3, [[1, 0, 1, 0, 1, 0], [0, 1, 0, 1, 0, 1]]), (4, 2, [[1, 1, 1, 0, 1, 0], [0, 0, 0, 1, 0, 1]])):
    sol_ = tm.mean_field_hubbard({0: Hb}, nu_, nd_, U=6.0, periodic=False, kT=1e-6, n0=np.array(n0_, float), tol=1e-10,
                                 maxiter=5000)
    sol_.update(n_up=nu_, n_dn=nd_)
    psi = np.zeros(4 ** 6); psi[0] = 1.0
    for s_, nel, ops in ((1, nd_, cd_), (0, nu_, cu_)):
        V_ = np.linalg.eigh(Hb + np.diag(6.0 * sol_["n"][1 - s_]))[1][:, :nel]
        for k in range(nel):
            psi = sum(V_[i, k] * (ops[i].T @ psi) for i in range(6))
    psi /= np.linalg.norm(psi)
    err = max(err, abs(tm.uhf_s2(Hb, sol_, 6.0, 1e-6) - float(np.real(psi.conj() @ (S2op @ psi)))))
check("<S^2> of a mean-field determinant (S_z = 0 and 1) = brute force in the Fock space", err < 1e-10, f"{err:.1e}")
st2 = tb.Structure([[0, 0], [1.42, 0]], ["C", "C"], [0, 0], "dimer"); st2.periodic = False
H2 = np.array([[0, -0.05], [-0.05, 0]])
g2 = tm.mfh_spin_gap(H2, st2, tm.mfh_scan({0: H2}, st2, 0.566, 1, 1e-4), 0.566, 1e-4)
J2 = (np.sqrt(0.566 ** 2 + 16 * 0.05 ** 2) - 0.566) / 2
check("Hubbard dimer (t = 50 meV, U = 0.566 eV): spin-projected mean-field gap = exact J within 0.5 %",
      abs(g2["yamaguchi"] / J2 - 1) < 0.005, f"{1e3 * g2['yamaguchi']:.3f} vs {1e3 * J2:.3f} meV")
asi = np.array([[0.022384, 0.016604], [1.431865, -0.003314], [2.138802, 1.221138], [1.416812, 2.431825],
                [-0.005702, 2.466341], [-0.718765, 1.231280], [-0.445087, -1.322183], [0.709714, -2.159273],
                [1.881632, -1.347213], [3.527536, 1.503578], [3.644842, 2.924518], [2.342501, 3.506061]])
sti = tb.Structure(asi, ["C"] * 12, [0, 0], "as-indacene"); sti.periodic = False
Hi = np.zeros((12, 12))
for i, j in cKDTree(asi).query_pairs(1.62):
    Hi[i, j] = Hi[j, i] = -2.7
gi = tm.mfh_spin_gap(Hi, sti, tm.mfh_scan({0: Hi}, sti, 3.0, 1, 0.005), 3.0, 0.005)
ri = tm.run_ed(Hi - 1.5 * np.eye(12), [(k, k, k, k, 3.0) for k in range(12)], 12, nstates=3)
check("as-indacene (diradical, pentagons, t1 only, U = 3 eV): two spin centres; projected mean-field S-T gap "
      "within 10 % of the exact Hubbard ED",
      gi["n_centres"] == 2 and abs(gi["yamaguchi"] / (ri["E"][1] - ri["E"][0]) - 1) < 0.1
      and ri["S"][0] < 0.1 and abs(ri["S"][1] - 1) < 0.1,
      f"{1e3 * gi['yamaguchi']:.1f} vs exact {1e3 * (ri['E'][1] - ri['E'][0]):.1f} meV")
ring6 = 1.42 * np.c_[np.cos(np.pi / 3 * np.arange(6) + np.pi / 6), np.sin(np.pi / 3 * np.arange(6) + np.pi / 6)]
mx = np.vstack([ring6, 2 * ring6[0], 2 * ring6[2]])
stm = tb.Structure(mx, ["C"] * 8, [0, 0], "m-xylylene"); stm.periodic = False
Hm_ = np.zeros((8, 8))
for i, j in cKDTree(mx).query_pairs(1.6):
    Hm_[i, j] = Hm_[j, i] = -2.7
gm = tm.mfh_spin_gap(Hm_, stm, tm.mfh_scan({0: Hm_}, stm, 3.0, 1, 0.005), 3.0, 0.005)
rm_ = tm.run_ed(Hm_ - 1.5 * np.eye(8), [(k, k, k, k, 3.0) for k in range(8)], 8, nstates=3)
check("m-xylylene (radicals on one sublattice): mean field and exact ED both give a triplet ground state",
      gm["raw"] < 0 and abs(rm_["S"][0] - 1) < 0.1 and any("ferromagnetic" in x for x in tm.spin_gap_report(gm)),
      f"mean field {1e3 * gm['yamaguchi']:.0f} meV, exact {-1e3 * (rm_['E'][1] - rm_['E'][0]):.0f} meV")
fo4 = [d for d in tb.symmetric_terminations(ol, 2) if d["units"] == 4][0]
f4 = tb.finite_chain(ol, 2, fo4["offset"], True, fo4["extra"])
H4 = tb.hoppings(f4)[0]
g4 = tm.mfh_spin_gap(H4, f4, tm.mfh_scan({0: H4}, f4, 3.0, 1, 0.005), 3.0, 0.005)
h4_ = g4["heis"]
check("olympicene chain of 4 units: 4 radicals, Heisenberg mapping (symmetric end bonds, Ising rms < 1 meV), "
      "singlet ground state",
      g4["n_centres"] == 4 and h4_ is not None and abs(h4_["J"][0] - h4_["J"][2]) < 1e-4 and h4_["rms"] < 1e-3
      and h4_["S0"] == 0 and h4_["gap_up"] > 0, f"J = {np.round(1e3 * np.array(h4_['J']), 2)} meV, gap {1e3 * h4_['gap_up']:.2f} meV")
fo5 = [d for d in tb.symmetric_terminations(ol, 2) if d["units"] == 5][0]
f5 = tb.finite_chain(ol, 2, fo5["offset"], True, fo5["extra"])
H5 = tb.hoppings(f5)[0]
sc5 = tm.mfh_scan({0: H5}, f5, 3.0, 1, 0.005)
g5 = tm.mfh_spin_gap(H5, f5, sc5, 3.0, 0.005)
plain = min(o["sol"]["E"] for o in sc5 if o["two_sz"] == 3) - g5["E_LS"]
check("olympicene chain of 5 units: the flipped-radical starts find the S_z = 3/2 state with one domain wall "
      "(the plain scan misses it)", g5["raw"] < 0.05 and plain > 0.1,
      f"{1e3 * g5['raw']:.1f} meV vs {1e3 * plain:.1f} meV")
fa = [d for d in tb.symmetric_terminations(ag, 4) if d["N"] == 56][0]
f56 = tb.finite_chain(ag, 4, fa["offset"], True, fa["extra"])
H56 = tb.hoppings(f56)[0]
g56 = tm.mfh_spin_gap(H56, f56, tm.mfh_scan({0: H56}, f56, 3.0, 1, 0.005), 3.0, 0.005)
xs = [f56.xy[c].mean(axis=0)[0] for c in g56["centres"]]
check("7-AGNR with zigzag ends (56 atoms): the two end radicals are the spin centres, antiferromagnetic gap",
      g56["n_centres"] == 2 and xs[0] < f56.xy[:, 0].mean() < xs[1] and g56["yamaguchi"] > 0,
      f"{1e3 * g56['yamaguchi']:.1f} meV")
# ---------------------------------------------------------------- spectra and maps at any zoom
tri = tb.load_structure(os.path.join(os.path.dirname(HERE), "examples", "triangulene_chain.cdxml"))
Ht = tb.hoppings(tri)
ct = tb.bloch_cache(Ht)
kk, Ek = tb.bands(Ht, 201)
EFt, _ = tb.fermi(Ek, tri.n_electrons)
xx = np.linspace(Ek.min() - EFt - 0.5, Ek.max() - EFt + 0.5, 6000)
dall, _ = tb.cache_spectra(ct, xx, 0.02, shift=EFt)
check("DOS of the Bloch cache integrates to the number of bands", abs(np.trapezoid(dall, xx) - tri.N) < 1e-3,
      f"{np.trapezoid(dall, xx):.5f} vs {tri.N}")
eg_, Lold = tb.dos_ldos(Ht, EFt - 3, EFt + 3, 0.02, ne=600, nk=401)
dnew, Lnew = tb.cache_spectra(ct, eg_ - EFt, 0.02, atoms=[0, 7, 30], shift=EFt)
check("wide window (20 meV): DOS and LDOS = the plain k sum", np.abs(dnew - Lold.sum(0)).max() < 0.01 * Lold.sum(0).max()
      and np.abs(Lnew - Lold[[0, 7, 30]]).max() < 0.01 * Lold[[0, 7, 30]].max())
# zoomed on the flat bands with 0.3 meV: the k-interpolated map against a brute-force sum on 20001 k points
sg, E0_ = 3e-4, -0.03
mp = tb.cache_map(ct, E0_, sg, shift=EFt)
ref_map = np.zeros(tri.N)
for k in np.linspace(0, np.pi, 20001)[:-1] + np.pi / 40000:
    e_, v_ = np.linalg.eigh(tb.Hk(Ht, k))
    g_ = np.exp(-(e_ - EFt - E0_) ** 2 / (2 * sg ** 2)) / (np.sqrt(2 * np.pi) * sg)
    ref_map += (np.abs(v_) ** 2) @ g_
ref_map /= 20000
check("zoomed map (0.3 meV, 401 k interpolated) = brute force on 20000 k points within 1 %",
      np.abs(mp - ref_map).max() < 0.01 * ref_map.max(), f"{np.abs(mp - ref_map).max() / ref_map.max():.1e}")
check("map summed over the atoms = DOS at that energy; band map normalised",
      abs(mp.sum() / tb.cache_spectra(ct, [E0_], sg, shift=EFt)[0][0] - 1) < 1e-6 and   # |psi|^2 kept in float32
      abs(tb.cache_band_map(ct, 34).sum() - 1) < 1e-5 and abs(tb.cache_band_map(ct, 34, k=-1.0).sum() - 1) < 1e-5)
cf = tb.bloch_cache({0: H56})
ef_, vf_ = np.linalg.eigh(H56)
g_ = np.exp(-(ef_ - 0.01) ** 2 / (2 * 0.005 ** 2)) / (np.sqrt(2 * np.pi) * 0.005)
check("finite system: map = sum_n |psi_n|^2 g(E - E_n)", np.allclose(tb.cache_map(cf, 0.01, 0.005), (np.abs(vf_) ** 2) @ g_))
check("auto broadening: window / 250 between 0.1 and 20 meV",
      tb.auto_broadening((-3, 3)) == 0.02 and abs(tb.auto_broadening((-0.05, 0.05)) - 4e-4) < 1e-12
      and tb.auto_broadening((0, 1e-3)) == 1e-4 and tb.auto_broadening((-3, 3), 0.005) == 0.005
      and tb.parse_broadening("auto") is None and tb.parse_broadening("2") == 0.002)

print("\nALL PASSED" if ok else "\nSOME CHECKS FAILED")
