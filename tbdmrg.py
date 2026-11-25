"""
tbdmrg: DMRG solver (block2) for the reduced Hubbard models of tbfit (tabs 4 and 7).

    pip install block2          (Linux, macOS, Windows wheels; brings MKL with it)

The cluster Hamiltonian of tbmany.build_cluster,

    H = sum_ab,s T^s_ab c+_as c_bs + sum_(abcd) v c+_a,up c_b,up c+_c,dn c_d,dn ,

is turned into a matrix product operator term by term: the spin-resolved hoppings Tup, Tdn and
the interaction terms are exactly the ones the ED uses (on-site U or the full projected
interaction, particle-hole form included), with U(1) x U(1) symmetry (N and S_z).

Spectrum (solve_cluster): the lowest states of the S_z sector of the ED are found one after the
other with two-site DMRG, each one orthogonal to the ones before (energy penalty). For every state: energy, total spin S from <S^2>, charges and
<S_i.S_j> from the one- and two-particle density matrices, and the transition spin densities
<f|S^z_ab|g> from the ground level (transition density matrices). The spin-flip weights of the
ED need the three components of S; by the Wigner-Eckart theorem they follow from S^z alone,

    sum_q sum_M' |<S' M'|S^q|S M>|^2 = |<S' M|S^z|S M>|^2 (2S'+1) / ((2S+1) <S M 1 0|S' M>^2),

except when the Clebsch-Gordan coefficient vanishes (S' = S with M = 0): those levels are
recomputed in the S_z = 1 sector, where it does not.

Charge excitations (charge_excitations): lowest states with N + 1 and N - 1 electrons and the
Dyson amplitudes <f|c+_a,s|0>, <f|c_a,s|0> (single-site operators as MPOs).

The results have the same form as tbmany.solve_cluster / charge_excitations (no wavefunction
X: the transition amplitudes are stored instead), so the rest of the program uses them as they
are. block2 holds the Python interpreter while it runs, so every calculation runs in a child
process: the GUI stays responsive and a crash of the solver cannot take it down. (In a script,
put the calls under  if __name__ == "__main__":  as for any multiprocessing code.)
"""

import os
import time
import shutil
import tempfile
import traceback
from math import comb

import numpy as np

DEFAULTS = dict(bond_dim=500, sweeps=24, tol=1e-10, threads=None, stack_mem=1 << 30, seed=1234,
                level_tol=1e-6, reorder=True)


_AVAILABLE = None


def available():
    """(True, version) if block2 can be imported, else (False, how to install it)."""
    global _AVAILABLE
    if _AVAILABLE is None:
        _AVAILABLE = _check_block2()
    return _AVAILABLE


def _check_block2():
    try:
        import block2  # noqa: F401
        from pyblock2.driver.core import DMRGDriver  # noqa: F401
    except Exception as e:  # ImportError, missing DLLs, ...
        return False, (f"block2 is not available ({type(e).__name__}: {e}). Install it with\n"
                       "    python -m pip install block2\n"
                       "(on Windows: in the same Python that runs tbgui, e.g. the conda prompt)")
    try:
        from importlib.metadata import version
        return True, version("block2")
    except Exception:
        return True, "?"


# ---------------------------------------------------------------------------- helpers
def _cg10(j, m, jp):
    """Clebsch-Gordan <j m 1 0|jp m>."""
    if abs(m) > j + 1e-9 or abs(m) > jp + 1e-9:
        return 0.0
    if abs(jp - j) < 1e-9:
        return m / np.sqrt(j * (j + 1)) if j > 0 else 0.0
    if abs(jp - j - 1) < 1e-9:
        return float(np.sqrt((j - m + 1) * (j + m + 1) / ((2 * j + 1) * (j + 1))))
    if abs(jp - j + 1) < 1e-9 and j > 0:
        return float(-np.sqrt((j - m) * (j + m) / (j * (2 * j + 1))))
    return 0.0


def _fiedler_order(Tup, Tdn, terms):
    """Order of the sites along the MPS from the Fiedler vector of the coupling graph (hoppings
    and interaction terms between different sites): strongly coupled sites end up close, which
    lowers the entanglement across the MPS bonds and the bond dimension needed."""
    N = len(Tup)
    A = np.abs(Tup) + np.abs(Tdn)
    for a, b, c, d, v in terms:
        idx = sorted({int(a), int(b), int(c), int(d)})
        for x in idx:
            for y in idx:
                if x != y:
                    A[x, y] += abs(v)
    np.fill_diagonal(A, 0.0)
    A = 0.5 * (A + A.T)
    if N < 3 or not A.any():
        return np.arange(N)
    Lap = np.diag(A.sum(axis=1)) - A
    w, V = np.linalg.eigh(Lap)
    f = V[:, 1]
    if f[0] > f[-1]:                     # keep the natural direction when possible
        f = -f
    return np.argsort(f, kind="stable")


def _schedule(o):
    M, ns = int(o["bond_dim"]), int(o["sweeps"])
    bd = [min(M, 100)] * 2 + [min(M, 250)] * 2 + [M] * max(ns - 4, 1)
    noise = [1e-4] * 4 + [1e-5] * 2 + [1e-6] * 2 + [0.0] * max(ns - 8, 1)
    thr = [1e-8] * 4 + [1e-10] * 4 + [1e-12] * max(ns - 8, 1)       # Davidson: tight at the end (observables)
    return bd[:ns], noise[:ns], thr[:ns]


class _Solver:
    """block2 driver with the MPO of one cluster (lives in the child process)."""

    def __init__(self, job):
        from pyblock2.driver.core import DMRGDriver, SymmetryTypes
        self.o = dict(DEFAULTS, **(job.get("opts") or {}))
        Tup, Tdn = np.asarray(job["Tup"], float), np.asarray(job["Tdn"], float)
        terms = list(job["terms"])
        self.Ns = len(Tup)
        # order of the sites along the MPS: strongly coupled sites next to each other
        self.perm = _fiedler_order(Tup, Tdn, terms) if self.o.get("reorder", True) else np.arange(self.Ns)
        self.inv = np.argsort(self.perm)                  # MPS position of every cluster site
        p, q = self.perm, self.inv
        self.Tup, self.Tdn = Tup[np.ix_(p, p)], Tdn[np.ix_(p, p)]
        self.terms = [(int(q[a]), int(q[b]), int(q[c]), int(q[d]), v) for a, b, c, d, v in terms]
        self.n_elec, self.two_sz = int(job["n_elec"]), int(job["two_sz"])
        self.scratch = tempfile.mkdtemp(prefix="tbfit_dmrg_")
        mem = int(self.o["stack_mem"]) * (4 if int(self.o["bond_dim"]) > 1000 else 1)
        self.drv = DMRGDriver(scratch=self.scratch, symm_type=SymmetryTypes.SZ,
                              n_threads=int(self.o["threads"] or os.cpu_count() or 1), stack_mem=mem)
        try:
            self.drv.bw.b.Random.rand_seed(int(self.o["seed"]))
        except Exception:
            pass
        self.drv.initialize_system(n_sites=self.Ns, n_elec=self.n_elec, spin=self.two_sz)
        b = self.drv.expr_builder()
        for T, op in ((self.Tup, "cd"), (self.Tdn, "CD")):
            for i, j in zip(*np.nonzero(np.abs(T) > 1e-14)):
                b.add_term(op, [int(i), int(j)], float(T[i, j]))
        for a, b_, c, d, v in self.terms:
            if abs(v) > 0:
                b.add_term("cdCD", [int(a), int(b_), int(c), int(d)], float(v))
        self.mpo = self.drv.get_mpo(b.finalize(adjust_order=True), iprint=0)
        self.s2mpo = None
        self.info = []
        # penalty for the states already found: larger than the width of the spectrum
        width = 2 * max(np.abs(self.Tup).sum(axis=1).max(), np.abs(self.Tdn).sum(axis=1).max())
        width += sum(abs(t[4]) for t in self.terms) / max(self.Ns, 1) * 4
        self.penalty = float(max(5.0, 4 * width))

    def close(self):
        try:
            self.drv.finalize()
        except Exception:
            pass
        shutil.rmtree(self.scratch, ignore_errors=True)

    def sector_dim(self, n, two_sz):
        nu, nd = (n + two_sz) // 2, (n - two_sz) // 2
        if not (0 <= nu <= self.Ns and 0 <= nd <= self.Ns) or (n + two_sz) % 2:
            return 0
        return comb(self.Ns, nu) * comb(self.Ns, nd)

    def lowest(self, n, two_sz, nroots, tag):
        """Lowest nroots states of the sector (n electrons, 2 S_z), one after the other: state k
        is the ground state of H + w sum_(j<k) |j><j| (the states found before are pushed up by
        the penalty w, larger than the spread of the spectrum). Every state gets the full bond
        dimension (a state-averaged run needs a much larger one for the same accuracy).
        Returns the energies (sorted) and the MPSs."""
        nroots = int(min(nroots, self.sector_dim(n, two_sz)))
        if nroots < 1:
            return np.zeros(0), []
        bd, noise, thr = _schedule(self.o)
        target = self.drv.bw.SX(int(n), int(two_sz), 0)
        E, kets, sweeps, dw, dE = [], [], [], [], []
        t0 = time.time()
        for i in range(nroots):
            k = self.drv.get_random_mps(tag=f"{tag}{i}", bond_dim=bd[0], nroots=1, target=target)
            e = self.drv.dmrg(self.mpo, k, n_sweeps=len(bd), bond_dims=bd, noises=noise, thrds=thr,
                              tol=float(self.o["tol"]), iprint=0, proj_mpss=kets or None,
                              proj_weights=[self.penalty] * len(kets) if kets else None)
            bdims, dws, Es = self.drv.get_dmrg_results()
            Es = np.asarray(Es, float).ravel()
            E.append(float(np.real(e)))
            kets.append(k)
            sweeps.append(len(Es))
            dw.append(float(np.max(dws[-2:])) if len(dws) else 0.0)
            dE.append(float(abs(Es[-1] - Es[-2])) if len(Es) > 1 else float("nan"))
        E = np.array(E)
        order = np.argsort(E, kind="stable")
        self.info.append(dict(n=int(n), two_sz=int(two_sz), nroots=nroots, sweeps=int(max(sweeps)),
                              bond_dim=int(bd[-1]), discarded=float(max(dw)),
                              dE_last=float(np.nanmax(dE)) if np.isfinite(dE).any() else float("nan"),
                              reordered=bool(np.any(order != np.arange(nroots))), time=time.time() - t0))
        return E[order], [kets[i] for i in order]

    def s2(self, a, b):
        """<a|S^2|b>."""
        if self.s2mpo is None:
            self.s2mpo = self.drv.get_spin_square_mpo(iprint=0)
        return float(np.real(self.drv.expectation(a, self.s2mpo, b)))

    def spin(self, k):
        return 0.5 * (-1 + np.sqrt(1 + 4 * max(self.s2(k, k), 0.0)))

    def densities(self, k):
        """Charges n_i and spin correlations <S_i.S_j> (diagonal <S_i^2>) of a state."""
        p1 = np.asarray(self.drv.get_1pdm(k), float)
        aa, ab, bb = (np.asarray(x, float) for x in self.drv.get_2pdm(k))
        na, nb = np.diag(p1[0]), np.diag(p1[1])
        i = np.arange(self.Ns)
        # n_is n_jt = <c+_is c+_jt c_jt c_is> (i != j); dm2[i,j,k,l] = <c+_i c+_j c_k c_l>
        uu, dd = np.einsum("ijji->ij", aa), np.einsum("ijji->ij", bb)
        ud = np.einsum("ijji->ij", ab)
        pm = -np.einsum("ijij->ij", ab)                   # S+_i S-_j = -<c+_i,up c+_j,dn c_i,dn c_j,up>
        C = 0.25 * (uu - ud - ud.T + dd) + 0.5 * (pm + pm.T)
        C[i, i] = 0.75 * (na + nb - 2 * ab[i, i, i, i])
        q = self.inv                                       # back to the order of the cluster sites
        return (na + nb)[q], C[np.ix_(q, q)]

    def trans_sz(self, f, g):
        """<f| S^z_ab |g> = 1/2 (<f|c+_a,up c_b,up|g> - <f|c+_a,dn c_b,dn|g>)."""
        t = np.asarray(self.drv.get_trans_1pdm(f, g), float)
        q = self.inv
        return (0.5 * (t[0] - t[1]))[np.ix_(q, q)]


class _States:
    """Lowest states of one sector, with the spin purified: near-degenerate states of different
    total spin come out of DMRG slightly mixed (the admixture is of the order of the residual
    over the splitting), which leaves <S^2> almost exact but moves spin-flip weight from one
    level to the other at first order. Within every group of such states, S^2 is diagonalised
    in the space they span and each state is replaced by its projection on the S^2 eigenspace
    of its own (rounded) spin, orthonormalised symmetrically (the closest pure-spin states).
    The new states are combinations sum_f R[f, f'] |f>; charges and <S_i.S_j> (spin scalars:
    no cross terms between different spins) and the energies follow with weights R^2, the
    transition densities linearly."""

    def __init__(self, solver, E, kets, window=2e-3):
        self.sv, self.kets = solver, kets
        n = len(E)
        S = np.array([solver.spin(k) for k in kets])
        Sr = np.round(2 * S) / 2
        R = np.eye(n)
        par = list(range(n))

        def root(x):
            while par[x] != x:
                par[x] = par[par[x]]
                x = par[x]
            return x
        for i in range(n):
            for j in range(i + 1, n):
                if abs(E[i] - E[j]) < window and Sr[i] != Sr[j]:
                    par[root(i)] = root(j)
        comps = {}
        for i in range(n):
            comps.setdefault(root(i), []).append(i)
        self.mixing = 0.0
        for comp in (c for c in comps.values() if len(c) > 1):
            k = len(comp)
            A = np.diag([S[i] * (S[i] + 1) for i in comp])
            for a in range(k):
                for b in range(a + 1, k):
                    A[a, b] = A[b, a] = solver.s2(kets[comp[a]], kets[comp[b]])
            w, V = np.linalg.eigh(A)
            cols = np.zeros((k, k))
            used = np.zeros(k, bool)
            for sig in sorted(set(Sr[comp])):
                idx = [a for a, i in enumerate(comp) if Sr[i] == sig]
                order = [x for x in np.argsort(np.abs(w - sig * (sig + 1))) if not used[x]][:len(idx)]
                used[order] = True
                B = V[:, order] @ V[:, order].T[:, idx]
                ev, U = np.linalg.eigh(B.T @ B)
                cols[:, idx] = B @ (U / np.sqrt(np.clip(ev, 1e-14, None))) @ U.T
            R[np.ix_(comp, comp)] = cols
            self.mixing = max(self.mixing, float(np.abs(cols - np.eye(k)).max()))
        self.R = R
        self.S = np.where([any(i in c and len(c) > 1 for c in comps.values()) for i in range(n)], Sr, S)
        self.E = (R ** 2).T @ np.asarray(E, float)
        self._tz = {}

    def densities(self):
        d = [self.sv.densities(k) for k in self.kets]
        W = self.R ** 2
        return W.T @ np.array([x[0] for x in d]), np.einsum("fp,fij->pij", W, np.array([x[1] for x in d]))

    def tz(self, fp, gp):
        """<f'|S^z_ab|g'> of the purified states f', g'."""
        out = 0.0
        for f in np.nonzero(np.abs(self.R[:, fp]) > 1e-12)[0]:
            for g in np.nonzero(np.abs(self.R[:, gp]) > 1e-12)[0]:
                if (f, g) not in self._tz:
                    self._tz[(f, g)] = self.sv.trans_sz(self.kets[f], self.kets[g])
                out = out + self.R[f, fp] * self.R[g, gp] * self._tz[(f, g)]
        return out


def _levels(E, S, tol):
    import tbmany as tm
    return tm.levels(E, S, tol=tol)


# ---------------------------------------------------------------------------- tasks (child process)
def _task_spectrum(job):
    s = _Solver(job)
    try:
        o, n, tsz = s.o, s.n_elec, s.two_sz
        warn = []
        E, kets = s.lowest(n, tsz, job["nstates"], "K")
        st = _States(s, E, kets)
        E, S = st.E, st.S
        bad = np.abs(2 * S - np.round(2 * S)) > 0.05
        if bad.any():
            warn.append(f"states {list(np.where(bad)[0])} have a non-integer spin: increase the bond dimension "
                        "or the sweeps")
        nn, C = st.densities()
        dEs = [i["dE_last"] for i in s.info if np.isfinite(i["dE_last"])]
        tol = max(float(o["level_tol"]), 10 * max(dEs)) if dEs else float(o["level_tol"])
        lev = _levels(E, S, tol)
        G = lev[0]
        m = tsz / 2
        jr = lambda x: round(2 * x) / 2
        zero = np.zeros((s.Ns, s.Ns))
        taus, need = [], []
        for li, L in enumerate(lev):
            tl = []
            for g in G:
                for f in L:
                    j, jp = jr(S[g]), jr(S[f])
                    cg = _cg10(j, m, jp)
                    if abs(jp - j) > 1.01 or (j == 0 and jp == 0):
                        z = zero
                    elif cg ** 2 < 1e-12:
                        need.append(li)
                        z = zero
                    else:
                        z = np.sqrt((2 * jp + 1) / ((2 * j + 1) * cg ** 2)) * st.tz(f, g)
                    tl.append(dict(z=z, p=zero, m=zero))
            taus.append(tl)
        need = sorted(set(need))
        if need:
            # S' = S with M = 0: the Clebsch-Gordan coefficient vanishes; use the members M = 1
            Emax = max(E[lev[li][0]] for li in need + [0])
            n1 = int(sum(1 for k in range(len(E)) if S[k] > 0.75 and E[k] <= Emax + tol)) + 2
            E1, k1 = s.lowest(n, tsz + 2, n1, "M")
            st1 = _States(s, E1, k1)
            E1, S1 = st1.E, st1.S

            def partners(L):
                """The S_z = 1 members of the multiplets of level L: same spin, nearest energies."""
                e, sp = E[L[0]], S[L[0]]
                cand = sorted((abs(E1[k] - e), k) for k in range(len(E1)) if abs(S1[k] - sp) < 0.25)
                return [k for d, k in cand[:len(L)] if d < max(10 * tol, 2e-4)]
            G1 = partners(G)
            for li in need:
                F1 = partners(lev[li])
                if len(G1) != len(G) or len(F1) != len(lev[li]):
                    warn.append(f"level {li}: its S_z = 1 members were not found (spin-flip weight set to 0); "
                                "increase the bond dimension or the sweeps")
                    continue
                tl = []
                for g in G1:
                    for f in F1:
                        j, jp = jr(S1[g]), jr(S1[f])
                        cg = _cg10(j, 1.0, jp)
                        z = np.sqrt((2 * jp + 1) / ((2 * j + 1) * cg ** 2)) * st1.tz(f, g) if cg ** 2 > 1e-12 else zero
                        tl.append(dict(z=z, p=zero, m=zero))
                taus[li] = tl
        return dict(E=E, S=S, n=nn, C=C, levels=lev, taus=taus, n_ground=len(G), info=s.info,
                    warnings=warn, level_tol=tol, spin_mixing=st.mixing)
    finally:
        s.close()


def _task_charge(job):
    s = _Solver(job)
    try:
        n, tsz = s.n_elec, s.two_sz
        nu, nd = (n + tsz) // 2, (n - tsz) // 2
        ng = int(job["n_ground"])
        E, kets = s.lowest(n, tsz, ng, "G")
        E0 = float(E[0])
        spins = (0,) if nu == nd else (0, 1)
        out = []
        for kind in ("add", "remove"):
            for sp in spins:
                dn = 1 if kind == "add" else -1
                d2 = dn if sp == 0 else -dn
                if s.sector_dim(n + dn, tsz + d2) == 0:
                    continue
                Ef, fs = s.lowest(n + dn, tsz + d2, job["nstates"], f"{kind[0]}{sp}")
                op = {("add", 0): "c", ("add", 1): "C", ("remove", 0): "d", ("remove", 1): "D"}[(kind, sp)]
                D = np.zeros((len(kets), len(fs), s.Ns))
                for a in range(s.Ns):                        # a: MPS position
                    smpo = s.drv.get_site_mpo(op=op, site_index=a, iprint=0)
                    for gi, g in enumerate(kets):
                        for k, f in enumerate(fs):
                            D[gi, k, a] = float(np.real(s.drv.expectation(f, smpo, g)))
                D = D[:, :, s.inv]                               # cluster order
                w = (Ef - E0) if kind == "add" else (E0 - Ef)
                out.append(dict(kind=kind, spin=sp, w=w, D=D, mult=2 if nu == nd else 1,
                                dim=s.sector_dim(n + dn, tsz + d2)))
        return dict(ch=out, info=s.info, E0=E0)
    finally:
        s.close()


_TASKS = dict(spectrum=_task_spectrum, charge=_task_charge)


def _child(conn, job):
    try:
        conn.send(("ok", _TASKS[job["task"]](job)))
    except BaseException:
        conn.send(("error", traceback.format_exc()))
    finally:
        conn.close()


def _run(job, inprocess=False):
    """Run a task in a child process (default) or here (tests)."""
    if inprocess:
        return _TASKS[job["task"]](job)
    import multiprocessing as mp
    ctx = mp.get_context("spawn")
    rd, wr = ctx.Pipe(duplex=False)
    p = ctx.Process(target=_child, args=(wr, job), daemon=True)
    p.start()
    wr.close()
    try:
        status, payload = rd.recv()
    except EOFError:
        p.join()
        raise RuntimeError(f"the DMRG process stopped without a result (exit code {p.exitcode}): out of memory "
                           "(try a smaller bond dimension), or, in a script, the calling code is not under "
                           "'if __name__ == \"__main__\":' (needed to start processes)") from None
    finally:
        rd.close()
    p.join()
    if status != "ok":
        raise RuntimeError("DMRG failed:\n" + payload)
    return payload


def _job(cl, task, n_elec, two_sz, opts):
    return dict(task=task, Tup=np.asarray(cl["Tup"], float), Tdn=np.asarray(cl["Tdn"], float),
                terms=[tuple(t) for t in cl["terms"]], n_elec=int(n_elec), two_sz=int(two_sz),
                opts=dict(opts or {}))


# ---------------------------------------------------------------------------- public API
def solve_cluster(cl, n_elec=None, nstates=20, two_sz=None, opts=None, inprocess=False):
    """DMRG version of tbmany.solve_cluster: the lowest nstates states of the S_z sector
    (default the lowest, 2 S_z = N mod 2). Returns the same dict (E, S, C, n, n_up, n_dn, dim)
    without X, plus levels, taus (transition spin densities for every level, as
    tbmany.inelastic_amplitudes), n_ground, solver='dmrg', dmrg_info, warnings."""
    ok, msg = available()
    if not ok:
        raise RuntimeError(msg)
    Ns = len(cl["T"])
    n_elec = Ns if n_elec is None else int(n_elec)
    two_sz = (n_elec % 2) if two_sz is None else int(two_sz)
    n_up, n_dn = (n_elec + two_sz) // 2, (n_elec - two_sz) // 2
    if not (0 <= n_dn <= Ns and 0 <= n_up <= Ns):
        raise ValueError("impossible electron number for this cluster")
    job = _job(cl, "spectrum", n_elec, two_sz, opts)
    job["nstates"] = int(nstates)
    t0 = time.time()
    r = _run(job, inprocess)
    r.update(n_up=n_up, n_dn=n_dn, dim=comb(Ns, n_up) * comb(Ns, n_dn), solver="dmrg", Ns=Ns,
             n_elec=n_elec, two_sz=two_sz, opts=dict(DEFAULTS, **(opts or {})), time=time.time() - t0,
             inprocess=bool(inprocess))
    r["dmrg_info"] = r.pop("info")
    return r


def charge_excitations(res, cl, nstates=12, inprocess=None):
    """DMRG version of tbmany.charge_excitations for a result of solve_cluster (same output).
    inprocess None: as the run that gave res."""
    job = _job(cl, "charge", res["n_elec"], res["two_sz"], res.get("opts"))
    job.update(nstates=int(nstates), n_ground=int(res["n_ground"]))
    r = _run(job, res.get("inprocess", False) if inprocess is None else inprocess)
    res.setdefault("dmrg_info_charge", r["info"])
    return r["ch"]


def report(res):
    """Text lines on the DMRG runs of a result (bond dimension, truncation, convergence)."""
    o = res.get("opts", {})
    L = [f"DMRG (block2, two-site, one state after the other; U(1) x U(1): N, S_z): bond dimension {o.get('bond_dim')}, "
         f"up to {o.get('sweeps')} sweeps, {res.get('time', 0):.1f} s"]
    for i in res.get("dmrg_info", []):
        L.append(f"  N = {i['n']}, 2 S_z = {i['two_sz']}: {i['nroots']} states, up to {i['sweeps']} sweeps, "
                 f"bond dim {i['bond_dim']}, discarded weight {i['discarded']:.1e}, "
                 f"last energy change {i['dE_last']:.1e} eV")
    worst = max([i["discarded"] for i in res.get("dmrg_info", [])] + [0])
    if worst > 1e-6:
        L.append("  WARNING: discarded weight > 1e-6: the states are not converged in the bond dimension; "
                 "increase the bond dimension")
    dEmax = max([i["dE_last"] for i in res.get("dmrg_info", []) if np.isfinite(i["dE_last"])] + [0])
    if dEmax > 1e-6:
        L.append(f"  WARNING: energies still changing by {dEmax:.1e} eV in the last sweep: increase the sweeps")
    if res.get("spin_mixing", 0) > 1e-6:
        L.append(f"  near-degenerate states of different spin were separated (S^2 diagonalised among them; largest "
                 f"correction {res['spin_mixing']:.1e})")
    if any(i.get("reordered") for i in res.get("dmrg_info", [])):
        L.append("  note: a state was found after a higher one (sorted again); if this repeats, increase the sweeps")
    L += ["  WARNING: " + w for w in res.get("warnings", [])]
    return L
