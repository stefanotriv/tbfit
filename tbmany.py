"""
tbmany: interacting electrons and spin Hamiltonians for the effective models of tbcore.

1. Projected interaction
   The effective sites are Wannier-like orbitals |w_n> (Loewdin-orthogonalised projection of
   the TB bands on the chosen sites, see tbcore.project). The atomic Hubbard interaction
   U sum_x n_x,up n_x,dn  (x = carbon pz orbitals) projected on these orbitals is exactly

       H_int = U sum_{abcd} O_abcd  c+_a,up c_b,up c+_c,dn c_d,dn ,
       O_abcd = sum_x w_a(x) w_b(x) w_c(x) w_d(x)      (real orbitals)

   The on-site part is U_a = U O_aaaa = U sum_x |w_a(x)|^4. The other terms (inter-site
   density, direct exchange, pair hopping) are kept when |U O| is above a threshold.

2. Exact diagonalisation (ED) of H = sum t_ab c+_a c_b + H_int on a finite cluster of
   unit cells (open chain or ring), in a sector of fixed N_up, N_dn. The wave function is
   stored as a matrix V[up configuration, down configuration] and every term is a product
   (up bilinear) x (down bilinear), so  (A x B) v = A V B^T  and the full Hamiltonian is
   never built. Fermion signs: modes ordered by site index inside each spin species,
   c+_i gives (-1)^(number of occupied modes below i).

3. Observables: <n_a>, local moments <S_a^2>, spin correlations <S_a.S_b>, total S from
   <S^2> = S(S+1) (also exactly S^- S^+ + Sz(Sz+1) with the ladder operators).
   Spin-flip transitions (inelastic tunnelling): W_i = sum_a sum_{M_f} |<f M_f|S_i^a|0>|^2
   from the transition spin densities <f_q|S^q_ab|0> (q = z, +, -; the M +- 1 members of
   the final multiplet are S^+-_tot |f>), per site or per atom (S_x = sum w_a w_b S_ab).

4. Spin model: H_spin = sum_(ab) J_ab S_a.S_b on the spin sites, diagonalised exactly;
   the J's are fitted to the low-energy ED spectrum (one state per spin configuration) and
   to the ground-state correlations (the spectrum alone is invariant under relabelling).
   Analytic estimates: superexchange 4t^2/U, Kondo coupling 2 tau^2 [1/D1 + 1/D2].

5. Atomistic mean-field Hubbard (collinear, unrestricted Hartree-Fock) on the full TB,
   periodic (k grid) or finite, same conventions as the `hubbard` package (DIPC):
       H_sigma = H_TB + U diag(<n_-sigma>),  E = sum_occ e - U sum_x <n_up><n_dn>.
"""

from math import comb
from itertools import combinations
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh, LinearOperator
from scipy.optimize import least_squares

import tbcore as tb


# ===========================================================================
# 1. Wannier-like orbitals and projected interaction
# ===========================================================================
def wannier_orbitals(H, st, model, nk=64, Rmax=4):
    """Real-space effective orbitals.

    Returns (w, hR, smin): w[R] is an (N_atoms x n_sites) real array, the weight of the
    orbital of site n (home cell 0) on atom i of cell R; hR[R] the projected hoppings
    <n,0|H|m,R>. Convention: Bloch sums |k> = sum_R e^{ikR}|R>, as in tbcore.
    """
    N = st.N
    G = tb.trial_orbitals(model, N)
    if G is None:
        raise ValueError("every effective site must sit on atoms to build its orbital")
    nocc = st.n_electrons // 2
    if st.n_electrons % 2:
        nocc += 1
    ks = 2 * np.pi * np.arange(nk) / nk
    Wk, Hk_eff, smin = [], [], 1.0
    for k in ks:
        E, V = np.linalg.eigh(tb.Hk(H, k))
        sel = tb.select(model, E, V, nocc, G)
        A = V[:, sel].conj().T @ G
        S = A.conj().T @ A
        w, U = np.linalg.eigh(S)
        smin = min(smin, w.min())
        Sm = U @ np.diag(np.clip(w, 1e-12, None) ** -0.5) @ U.conj().T
        Wk.append(V[:, sel] @ A @ Sm)                    # orbitals at k, atomic basis
        Hk_eff.append(Sm @ A.conj().T @ np.diag(E[sel]) @ A @ Sm)
    Wk, Hk_eff = np.array(Wk), np.array(Hk_eff)
    w = {R: np.real(np.mean(Wk * np.exp(1j * ks * R)[:, None, None], axis=0))
         for R in range(-Rmax, Rmax + 1)}
    hR = {R: np.real(np.mean(Hk_eff * np.exp(-1j * ks * R)[:, None, None], axis=0))
          for R in range(0, Rmax + 1)}
    return w, hR, smin


def orbital_spread(w):
    """(IPR = sum |w|^4, number of atoms holding 90 % of the weight) for each orbital."""
    dens = np.concatenate([w[R] ** 2 for R in sorted(w)], axis=0)
    ipr = (dens ** 2).sum(axis=0)
    n90 = [int(np.searchsorted(np.cumsum(np.sort(d)[::-1]), 0.9) + 1) for d in dens.T]
    return ipr, n90


def cluster_orbitals(w, ncells, periodic):
    """Orbitals of all cluster sites on a common atom list.

    Site (cell c, orbital n) -> column c * n_sites + n. For a ring the atom cells are
    taken modulo ncells (wrapped tails are added)."""
    Rs = sorted(w)
    Na, ns = w[Rs[0]].shape
    if periodic:
        W = np.zeros((ncells * Na, ncells * ns))
        for c in range(ncells):
            for R in Rs:
                cell = (c + R) % ncells
                W[cell * Na:(cell + 1) * Na, c * ns:(c + 1) * ns] += w[R]
    else:
        lo, hi = Rs[0], ncells - 1 + Rs[-1]
        W = np.zeros(((hi - lo + 1) * Na, ncells * ns))
        for c in range(ncells):
            for R in Rs:
                cell = c + R - lo
                W[cell * Na:(cell + 1) * Na, c * ns:(c + 1) * ns] += w[R]
    return W


def interaction_terms(W, U, mode="onsite", thr=1e-3):
    """Projected Hubbard interaction on the cluster: list of (a, b, c, d, value) meaning
    value * c+_a,up c_b,up c+_c,dn c_d,dn.

    mode "onsite": only a = b = c = d (U_a = U sum |w_a|^4)
    mode "full"  : all terms with |value| > thr (eV)
    U may be a number or an array of per-site values (then used for the on-site terms
    directly, e.g. edited by hand; the inter-site terms use U_atomic scaling U[a]/O_aaaa).
    """
    Ns = W.shape[1]
    D = W ** 2
    Oaaaa = (D ** 2).sum(axis=0)
    terms = []
    if np.ndim(U) == 0:
        Uons = U * Oaaaa
        Uat = np.full(Ns, float(U))
    else:
        Uons = np.asarray(U, float)
        Uat = np.where(Oaaaa > 0, Uons / Oaaaa, 0.0)
    for a in range(Ns):
        if abs(Uons[a]) > 0:
            terms.append((a, a, a, a, float(Uons[a])))
    if mode == "full":
        idx = np.arange(Ns)
        O = np.einsum("xa,xb,xc,xd->abcd", W[:, idx], W[:, idx], W[:, idx], W[:, idx], optimize=True)
        scale = np.mean(Uat[idx]) if len(idx) else 0.0
        for a, b, c, d in zip(*np.nonzero(np.abs(scale * O) > thr)):
            if a == b == c == d:
                continue
            terms.append((int(idx[a]), int(idx[b]), int(idx[c]), int(idx[d]), float(scale * O[a, b, c, d])))
    return terms


def cluster_hopping(model, pv, ncells, periodic, shift=0.0):
    """One-body matrix of the effective model on ncells cells (open chain or ring)."""
    ns = len(model.sites)
    Ns = ns * ncells
    T = np.zeros((Ns, Ns))
    for i, j, R, p, s in model.terms:
        v = s * pv[p]
        for c in range(ncells):
            a = c * ns + i
            if i == j and R == 0:
                T[a, a] += v - shift
                continue
            c2 = c + R
            if periodic:
                c2 %= ncells
            elif not 0 <= c2 < ncells:
                continue
            b = c2 * ns + j
            T[a, b] += v
            T[b, a] += v
    labels = [f"{model.sites[n]}{c}" for c in range(ncells) for n in range(ns)]
    return T, labels


# ===========================================================================
# 2. exact diagonalisation
# ===========================================================================
def _popcount(x, nbits):
    x = np.asarray(x, np.int64)
    c = np.zeros_like(x)
    for b in range(nbits):
        c += (x >> b) & 1
    return c


class SpeciesBasis:
    """Configurations of n fermions of one spin species on Ns modes, with the sparse
    matrices of the bilinears c+_i c_j (cached)."""

    def __init__(self, Ns, n):
        self.Ns, self.n = Ns, n
        self.states = np.array(sorted(sum(1 << b for b in c) for c in combinations(range(Ns), n)),
                               dtype=np.int64)
        self.dim = len(self.states)
        self.occ = ((self.states[:, None] >> np.arange(Ns)[None, :]) & 1).astype(float)
        self._cache = {}

    def index(self, s):
        return np.searchsorted(self.states, s)

    def E(self, i, j):
        """Sparse matrix of c+_i c_j in this basis."""
        key = (i, j)
        if key in self._cache:
            return self._cache[key]
        s = self.states
        if i == j:
            M = sp.diags(self.occ[:, i]).tocsr()
        else:
            ok = ((s >> j) & 1).astype(bool) & ~((s >> i) & 1).astype(bool)
            s0 = s[ok]
            s1 = s0 ^ (1 << j)
            sign = (-1.0) ** _popcount(s0 & ((1 << j) - 1), self.Ns)
            s2 = s1 | (1 << i)
            sign = sign * (-1.0) ** _popcount(s1 & ((1 << i) - 1), self.Ns)
            M = sp.csr_matrix((sign, (self.index(s2), np.where(ok)[0])), shape=(self.dim, self.dim))
        self._cache[key] = M
        return M

    def one_body(self, T):
        M = sp.csr_matrix((self.dim, self.dim))
        for i, j in zip(*np.nonzero(T)):
            M = M + T[i, j] * self.E(i, j)
        return M.tocsr()


class HubbardED:
    """H = sum_ab T_ab c+_a c_b (both spins) + sum terms value c+_a,up c_b,up c+_c,dn c_d,dn."""

    def __init__(self, T, terms, n_up, n_dn):
        self.T = np.asarray(T, float)
        self.Ns = len(T)
        self.up, self.dn = SpeciesBasis(self.Ns, n_up), SpeciesBasis(self.Ns, n_dn)
        self.dim = self.up.dim * self.dn.dim
        self.Tu, self.Td = self.up.one_body(self.T), self.dn.one_body(self.T)
        diag = np.zeros((self.up.dim, self.dn.dim))
        general = {}
        for a, b, c, d, v in terms:
            if a == b and c == d:                          # density-density: diagonal
                diag += v * np.outer(self.up.occ[:, a], self.dn.occ[:, c])
            else:
                general.setdefault((a, b), []).append((c, d, v))
        self.diag = diag
        self.general = [(self.up.E(a, b), sum(v * self.dn.E(c, d) for c, d, v in lst).tocsr())
                        for (a, b), lst in general.items()]

    def matvec(self, v):
        V = v.reshape(self.up.dim, self.dn.dim)
        out = self.Tu @ V + (self.Td @ V.T).T + self.diag * V
        for A, B in self.general:
            out += A @ (B @ V.T).T
        return out.ravel()

    def solve(self, nstates=20, dense_limit=2500):
        k = min(nstates, self.dim)
        if self.dim <= dense_limit:
            Hm = np.column_stack([self.matvec(e) for e in np.eye(self.dim)])
            E, X = np.linalg.eigh(0.5 * (Hm + Hm.T))
            return E[:k], X[:, :k]
        op = LinearOperator((self.dim, self.dim), matvec=self.matvec, dtype=float)
        E, X = eigsh(op, k=k, which="SA", tol=1e-10, ncv=min(self.dim, max(2 * k + 20, 60)))
        o = np.argsort(E)
        return E[o], X[:, o]

    # ------------------------------------------------------------------ observables
    def _expect(self, v, A, B):
        V = v.reshape(self.up.dim, self.dn.dim)
        return float(np.sum(V * (A @ (B @ V.T).T)))

    def density(self, v):
        P = v.reshape(self.up.dim, self.dn.dim) ** 2
        nu = P.sum(axis=1) @ self.up.occ
        nd = P.sum(axis=0) @ self.dn.occ
        return nu, nd

    def spin_correlations(self, v):
        """Matrix C_ab = <S_a . S_b> (diagonal: <S_a^2>)."""
        V = v.reshape(self.up.dim, self.dn.dim)
        P = V ** 2
        Ns = self.Ns
        C = np.zeros((Ns, Ns))
        ou, od = self.up.occ, self.dn.occ
        for a in range(Ns):
            for b in range(Ns):
                if a == b:
                    # S_a^2 = 3/4 (n_up + n_dn - 2 n_up n_dn)
                    C[a, a] = 0.75 * np.sum(P * (ou[:, a][:, None] + od[:, a][None, :]
                                                 - 2 * np.outer(ou[:, a], od[:, a])))
                    continue
                za = ou[:, a][:, None] - od[:, a][None, :]
                zb = ou[:, b][:, None] - od[:, b][None, :]
                szsz = 0.25 * np.sum(P * za * zb)
                # S_a^+ S_b^- = c+_a,up c_a,dn c+_b,dn c_b,up = - (c+_a,up c_b,up)(c+_b,dn c_a,dn)
                pm = -self._expect(v, self.up.E(a, b), self.dn.E(b, a))
                C[a, b] = szsz + pm        # <S+S-> is real here; (pm + pm*)/2 = pm
        return C

    def total_spin(self, v, C=None):
        C = self.spin_correlations(v) if C is None else C
        s2 = C.sum()
        return 0.5 * (-1 + np.sqrt(1 + 4 * max(s2, 0))), s2


# ---------------------------------------------------------------------------
# spin ladder operators and spin-flip (inelastic) transitions
# ---------------------------------------------------------------------------
_BASES = {}


def _basis(Ns, n):
    """Cached SpeciesBasis (one spin species, n fermions on Ns modes)."""
    if (Ns, n) not in _BASES:
        _BASES[(Ns, n)] = SpeciesBasis(Ns, n)
    return _BASES[(Ns, n)]


def _annihilator(Ns, n, i):
    """Sparse matrix of c_i from the n-particle to the (n-1)-particle basis of one species,
    sign (-1)^(number of occupied modes below i). Its transpose is c+_i (n-1 -> n)."""
    b0, b1 = _basis(Ns, n), _basis(Ns, n - 1)
    key = ("a", i)
    if key not in b0._cache:
        s = b0.states
        ok = ((s >> i) & 1).astype(bool)
        sign = (-1.0) ** _popcount(s[ok] & ((1 << i) - 1), Ns)
        b0._cache[key] = sp.csr_matrix((sign, (b1.index(s[ok] ^ (1 << i)), np.where(ok)[0])),
                                       shape=(b1.dim, b0.dim))
    return b0._cache[key]


def _rmul(V, M):
    """Dense V times sparse M (returned as a dense array)."""
    return np.asarray((M.T @ V.T).T)


def spin_raise(V, Ns, n_up, n_dn, sites=None):
    """S^+ = sum_a c+_a,up c_a,dn (over `sites`, default all) applied to the wave function
    V[up configuration, down configuration] of the sector (n_up, n_dn). Returns the matrix of
    the sector (n_up + 1, n_dn - 1), or None if that sector does not exist.
    Fermion order: all up modes before all down modes (as in HubbardED)."""
    if n_up >= Ns or n_dn <= 0:
        return None
    sites = range(Ns) if sites is None else sites
    out = 0
    for a in sites:
        out = out + _annihilator(Ns, n_up + 1, a).T @ _rmul(V, _annihilator(Ns, n_dn, a).T)
    return (-1.0) ** n_up * out


def spin_lower(V, Ns, n_up, n_dn, sites=None):
    """S^- = sum_a c+_a,dn c_a,up; sector (n_up, n_dn) -> (n_up - 1, n_dn + 1) (None if absent)."""
    if n_dn >= Ns or n_up <= 0:
        return None
    sites = range(Ns) if sites is None else sites
    out = 0
    for a in sites:
        out = out + _rmul(_annihilator(Ns, n_up, a) @ V, _annihilator(Ns, n_dn + 1, a))
    return (-1.0) ** (n_up - 1) * out


def total_spin_ladder(V, Ns, n_up, n_dn):
    """<S^2> = |S^+ v|^2 + Sz (Sz + 1) (exact identity S^2 = S^- S^+ + Sz^2 + Sz)."""
    sz = 0.5 * (n_up - n_dn)
    P = spin_raise(V, Ns, n_up, n_dn)
    s2 = (0.0 if P is None else float(np.sum(P * P))) + sz * (sz + 1)
    return 0.5 * (-1 + np.sqrt(1 + 4 * max(s2, 0))), s2


def clean_multiplets(E, X, Ns, n_up, n_dn, tol=1e-8):
    """Inside every group of degenerate states (|dE| < tol eV) rotate the eigenvectors so
    that each one has a definite total spin (a Lanczos / eigh basis of a degenerate level
    can mix a singlet and a triplet component, which gives S between 0 and 1 and spoils the
    selection rules)."""
    X = X.copy()
    k, n = 0, len(E)
    d_up, d_dn = comb(Ns, n_up), comb(Ns, n_dn)
    while k < n:
        g = [k]
        while g[-1] + 1 < n and abs(E[g[-1] + 1] - E[k]) < tol:
            g.append(g[-1] + 1)
        if len(g) > 1:
            P = [spin_raise(X[:, j].reshape(d_up, d_dn), Ns, n_up, n_dn) for j in g]
            if P[0] is not None:
                P = np.array([p.ravel() for p in P])
                w, R = np.linalg.eigh(P @ P.T)
                X[:, g] = X[:, g] @ R
        k = g[-1] + 1
    return X


def transition_amplitudes(F, G, Ns, n_up, n_dn):
    """Transition spin densities between the eigenstates G (initial) and F (final) of the
    same sector, for the components q = z, +, - of the spin operator
        S^q_ab = 1/2 (c+_a,up c_b,up - c+_a,dn c_b,dn),  c+_a,up c_b,dn,  c+_a,dn c_b,up .
    The + and - components connect G to the members M + 1 and M - 1 of the multiplet of F
    (obtained as S^+_tot F and S^-_tot F, normalised; |S^+ F|^2 = S(S+1) - M(M+1) >= 2 when the
    member exists, so a smaller norm is numerical noise and means no member). Returns dict z, p, m of (Ns x Ns)
    matrices tau[a, b] = <f_q| S^q_ab |g> (zeros where the member does not exist).
    A local spin operator on atom x is S^q_x = sum_ab w_a(x) w_b(x) S^q_ab."""
    du, dd = comb(Ns, n_up), comb(Ns, n_dn)
    F, G = F.reshape(du, dd), G.reshape(du, dd)
    out = {}
    # z: <c_a,s f | c_b,s g> for each spin (the constant Jordan-Wigner signs cancel)
    tz = np.zeros((Ns, Ns))
    if n_up > 0:
        A = [(_annihilator(Ns, n_up, a) @ F).ravel() for a in range(Ns)]
        B = [(_annihilator(Ns, n_up, b) @ G).ravel() for b in range(Ns)]
        tz += 0.5 * np.array(A) @ np.array(B).T
    if n_dn > 0:
        A = [_rmul(F, _annihilator(Ns, n_dn, a).T).ravel() for a in range(Ns)]
        B = [_rmul(G, _annihilator(Ns, n_dn, b).T).ravel() for b in range(Ns)]
        tz -= 0.5 * np.array(A) @ np.array(B).T
    out["z"] = tz
    # +: <f+| c+_a,up c_b,dn |g> = (-1)^n_up <c_a,up f+ | c_b,dn g>
    Fp = spin_raise(F, Ns, n_up, n_dn)
    tp = np.zeros((Ns, Ns))
    if Fp is not None and np.sum(Fp * Fp) > 1e-6:
        Fp = Fp / np.sqrt(np.sum(Fp * Fp))
        A = [(_annihilator(Ns, n_up + 1, a) @ Fp).ravel() for a in range(Ns)]
        B = [_rmul(G, _annihilator(Ns, n_dn, b).T).ravel() for b in range(Ns)]
        tp = (-1.0) ** n_up * np.array(A) @ np.array(B).T
    out["p"] = tp
    # -: <f-| c+_a,dn c_b,up |g> = (-1)^(n_up - 1) <c_a,dn f- | c_b,up g>
    Fm = spin_lower(F, Ns, n_up, n_dn)
    tm_ = np.zeros((Ns, Ns))
    if Fm is not None and np.sum(Fm * Fm) > 1e-6:
        Fm = Fm / np.sqrt(np.sum(Fm * Fm))
        A = [_rmul(Fm, _annihilator(Ns, n_dn + 1, a).T).ravel() for a in range(Ns)]
        B = [(_annihilator(Ns, n_up, b) @ G).ravel() for b in range(Ns)]
        tm_ = (-1.0) ** (n_up - 1) * np.array(A) @ np.array(B).T
    out["m"] = tm_
    return out


def spin_flip_weight(tau, W=None):
    """Rotation-averaged spin-flip weight sum_{a=x,y,z} sum_{M_f} |<f M_f| S^a |g>|^2
    = |t_z|^2 + (|t_+|^2 + |t_-|^2) / 2 . By the Wigner-Eckart theorem this does not depend
    on the member of the initial multiplet, so it is the weight for an unpolarised tip at
    zero field. W=None: one value per effective site (operator S_a); W (N_atoms x Ns): one
    value per atom (operator S_x projected on the effective orbitals)."""
    if W is None:
        return (np.diag(tau["z"]) ** 2 + 0.5 * np.diag(tau["p"]) ** 2 + 0.5 * np.diag(tau["m"]) ** 2)
    val = lambda t: np.einsum("xa,ab,xb->x", W, t, W, optimize=True)
    return val(tau["z"]) ** 2 + 0.5 * val(tau["p"]) ** 2 + 0.5 * val(tau["m"]) ** 2


def levels(E, S, tol=1e-8):
    """Group the states into levels: same energy (within tol eV) and same total spin."""
    out, used = [], np.zeros(len(E), bool)
    for k in range(len(E)):
        if used[k]:
            continue
        g = [j for j in range(k, len(E)) if not used[j] and abs(E[j] - E[k]) < tol and abs(S[j] - S[k]) < 0.25]
        used[g] = True
        out.append(g)
    return out


def level_weights(res):
    """Spin-flip weight per effective site for every level (list from `levels`), from the
    ground level: fast version of analyse_transition (site weights only), used for the
    overview of all transitions. Returns (levels, array n_levels x Ns)."""
    E, S, X = res["E"], res["S"], res["X"]
    Ns, nu, nd = res["ed"].Ns, res["n_up"], res["n_dn"]
    du, dd = comb(Ns, nu), comb(Ns, nd)
    lev = levels(E, S)
    ou = res["ed"].up.occ
    od = res["ed"].dn.occ
    pre = []
    for g in lev[0]:
        G = X[:, g].reshape(du, dd)
        Z = [0.5 * (ou[:, i][:, None] - od[:, i][None, :]) * G for i in range(Ns)]
        P = [spin_raise(G, Ns, nu, nd, [i]) for i in range(Ns)]
        M = [spin_lower(G, Ns, nu, nd, [i]) for i in range(Ns)]
        pre.append((Z, P, M))
    out = np.zeros((len(lev), Ns))
    for li, L in enumerate(lev):
        for f in L:
            F = X[:, f].reshape(du, dd)
            Fp, Fm = spin_raise(F, Ns, nu, nd), spin_lower(F, Ns, nu, nd)
            Fp = Fp / np.sqrt(np.sum(Fp * Fp)) if Fp is not None and np.sum(Fp * Fp) > 1e-6 else None
            Fm = Fm / np.sqrt(np.sum(Fm * Fm)) if Fm is not None and np.sum(Fm * Fm) > 1e-6 else None
            for Z, P, M in pre:
                for i in range(Ns):
                    w = np.sum(F * Z[i]) ** 2
                    if Fp is not None:
                        w += 0.5 * np.sum(Fp * P[i]) ** 2
                    if Fm is not None:
                        w += 0.5 * np.sum(Fm * M[i]) ** 2
                    out[li, i] += w / len(lev[0])
    return lev, out


def analyse_transition(res, f_level, W=None):
    """Spin-flip analysis of the transition ground level -> f_level (list of state indices)
    for the ED result of solve_cluster. Returns dict with dE (eV), S0, Sf, dS, site and atom
    weights (averaged over the ground level, summed over the final level), dC, dn, dS2,
    charge moved, 'allowed' (Delta S rule) and a label."""
    E, S, X = res["E"], res["S"], res["X"]
    Ns, nu, nd = res["ed"].Ns, res["n_up"], res["n_dn"]
    g_level = levels(E, S)[0]
    ws, wa = np.zeros(Ns), None if W is None else np.zeros(W.shape[0])
    for g in g_level:
        for f in f_level:
            tau = transition_amplitudes(X[:, f], X[:, g], Ns, nu, nd)
            ws += spin_flip_weight(tau) / len(g_level)
            if W is not None:
                wa += spin_flip_weight(tau, W) / len(g_level)
    C0, Cf = res["C"][g_level].mean(axis=0), res["C"][f_level].mean(axis=0)
    n0, nf = res["n"][g_level].mean(axis=0), res["n"][f_level].mean(axis=0)
    S0, Sf = float(np.mean(S[g_level])), float(np.mean(S[f_level]))
    dS = round(2 * (Sf - S0)) / 2
    moved = 0.5 * np.abs(nf - n0).sum()
    rule = abs(dS) <= 1 and not (round(2 * S0) == 0 and round(2 * Sf) == 0)
    return dict(dE=float(E[f_level[0]] - E[g_level[0]]), S0=S0, Sf=Sf, dS=dS, w_site=ws, w_atom=wa,
                total=float(ws.sum()), dC=Cf - C0, dn=nf - n0, dS2=np.diag(Cf) - np.diag(C0),
                moved=float(moved), allowed=bool(rule), kind=classify(nf - n0, np.diag(Cf) - np.diag(C0)),
                g_level=g_level, f_level=list(f_level))


def classify(dn, dS2, tol=0.05):
    """Label of an excitation from the change of the site charges dn and local moments dS2:
    a spin excitation keeps the charges (only the spin arrangement changes), a charge
    excitation moves electrons (doubly occupied / empty sites)."""
    moved = 0.5 * np.abs(dn).sum()
    if moved >= tol:
        return f"charge excitation ({moved:.2f} e moved)"
    if np.abs(dS2).max() >= tol:
        return "spin excitation, local moments change"
    return "spin excitation"


def run_ed(T, terms, n_elec, nstates=20, sz=None, dense_limit=2500):
    """ED in the lowest |Sz| sector (or the given 2*Sz). Returns dict with energies, total
    spin of every state, densities, local moments and spin correlations of every state."""
    Ns = len(T)
    two_sz = (n_elec % 2) if sz is None else int(sz)
    n_up = (n_elec + two_sz) // 2
    n_dn = n_elec - n_up
    if not (0 <= n_dn <= Ns and 0 <= n_up <= Ns):
        raise ValueError("impossible electron number for this cluster")
    ed = HubbardED(T, terms, n_up, n_dn)
    E, X = ed.solve(nstates, dense_limit)
    X = clean_multiplets(E, X, Ns, n_up, n_dn)
    S, C, n = [], [], []
    for k in range(len(E)):
        Ck = ed.spin_correlations(X[:, k])
        S.append(ed.total_spin(X[:, k], Ck)[0])
        C.append(Ck)
        nu, nd = ed.density(X[:, k])
        n.append(nu + nd)
    return dict(E=E, X=X, S=np.array(S), C=np.array(C), n=np.array(n), dim=ed.dim, ed=ed,
                n_up=n_up, n_dn=n_dn)


# ===========================================================================
# 3. spin model
# ===========================================================================
class HeisenbergED:
    """H = sum_(a<b) J_ab S_a.S_b for spins 1/2, in a sector of fixed Sz."""

    def __init__(self, n, two_sz=None):
        self.n = n
        two_sz = n % 2 if two_sz is None else two_sz
        nup = (n + two_sz) // 2
        self.basis = SpeciesBasis(n, nup)            # bit = spin up
        s = self.basis.states
        self.sz = self.basis.occ - 0.5               # (dim, n)
        self.flip = {}
        for a in range(n):
            for b in range(a + 1, n):
                ok = ((s >> a) & 1) != ((s >> b) & 1)
                s2 = s[ok] ^ ((1 << a) | (1 << b))
                self.flip[(a, b)] = sp.csr_matrix((np.full(ok.sum(), 0.5), (self.basis.index(s2), np.where(ok)[0])),
                                                  shape=(self.basis.dim, self.basis.dim))

    def matrix(self, J):
        """J: dict (a, b) -> value with a < b."""
        d = np.zeros(self.basis.dim)
        M = sp.csr_matrix((self.basis.dim, self.basis.dim))
        for (a, b), v in J.items():
            if a > b:
                a, b = b, a
            d += v * self.sz[:, a] * self.sz[:, b]
            M = M + v * self.flip[(a, b)]
        return (M + sp.diags(d)).toarray()

    def spectrum(self, J, with_spin=True):
        E, X = np.linalg.eigh(self.matrix(J))
        if not with_spin:
            return E, None
        # total spin from S^2 = sum_ab S_a.S_b = 3n/4 + 2 sum_{a<b} S_a.S_b
        S2op = self.matrix({(a, b): 2.0 for a in range(self.n) for b in range(a + 1, self.n)}) \
            + 0.75 * self.n * np.eye(self.basis.dim)
        s2 = np.einsum("ik,ij,jk->k", X, S2op, X)
        return E, 0.5 * (-1 + np.sqrt(1 + 4 * np.clip(s2, 0, None)))


def fit_spin_model(E_ref, S_ref, n_spins, pairs, groups, J0=None, C_ref=None, corr_weight=0.3,
                   corr_states="ground"):
    """Fit J's so that the Heisenberg model reproduces the low-energy ED states.

    pairs : list of (a, b) spin-site pairs; groups: parameter index of every pair.
    The reference states are matched to the model states of the same total spin, in order
    of energy, and the energies (relative to the ground state) are fitted.
    C_ref (optional, n_ref x n_spins x n_spins): the ED spin correlations <S_a.S_b> on the
    spin sites. The spectrum alone does not fix which bond carries which J (relabelling the
    spins leaves it unchanged, so a perfect fit can put the singlets on the wrong bonds);
    with C_ref the correlations are fitted too, with weight corr_weight x (width of the
    reference spectrum) per unit of <S.S>: of the ground state (corr_states="ground",
    default) or of every non-degenerate matched state ("all"). They are normalised to full
    moments (x 0.75 / sqrt(<S_a^2><S_b^2>)). Each start is first fitted to the energies only,
    then refined with the correlations.
    Returns (J values, model energies, model spins, rms of the matched energies); the rank of
    the Jacobian is in fit_spin_model.rank, the largest correlation error (ground state) in
    fit_spin_model.corr_err and the number of ED states without a model state of the same
    spin in fit_spin_model.unmatched.
    """
    hed = HeisenbergED(n_spins)
    ng = max(groups) + 1
    Eref = np.asarray(E_ref) - E_ref[0]
    Sref = np.round(2 * np.asarray(S_ref)) / 2
    scale = max(float(Eref.max()), 1e-9)
    iu = np.triu_indices(n_spins, 1)
    ops = [hed.matrix({(a, b): 1.0}) for a, b in zip(*iu)]
    S2op = hed.matrix({(a, b): 2.0 for a in range(n_spins) for b in range(a + 1, n_spins)}) \
        + 0.75 * n_spins * np.eye(hed.basis.dim)
    Cn = None
    if C_ref is not None:
        C_ref = np.asarray(C_ref, float)
        d = np.sqrt(np.clip(np.einsum("kaa->ka", C_ref), 1e-12, None))
        Cn = 0.75 * C_ref / (d[:, :, None] * d[:, None, :])

    iso = [i for i in range(len(Eref))
           if all(abs(Eref[j] - Eref[i]) > 0.05 * scale for j in np.where(Sref == Sref[i])[0] if j != i)]
    if corr_states == "ground":
        iso = [i for i in iso if i == 0]

    def match(S):
        """Model state of every reference state (same total spin, in order of energy; -1 if
        the model has fewer states of that spin)."""
        k = np.full(len(Sref), -1)
        for s in np.unique(Sref):
            ir, im = np.where(Sref == s)[0], np.where(S == s)[0]
            k[ir[:len(im)]] = im[:len(ir)]
        return k

    def model(x):
        J = {}
        for (a, b), g in zip(pairs, groups):
            J[(a, b)] = J.get((a, b), 0.0) + x[g]
        E, X = np.linalg.eigh(hed.matrix(J))
        s2 = np.einsum("ik,ij,jk->k", X, S2op, X)
        S = np.round(2 * 0.5 * (-1 + np.sqrt(1 + 4 * np.clip(s2, 0, None)))) / 2
        return E - E[0], S, X

    def resid(x, with_corr=True):
        """Fixed length: one energy per reference state (+ correlations of the isolated ones)."""
        E, S, X = model(x)
        k = match(S)
        r = [Eref[i] - E[k[i]] if k[i] >= 0 else 1.0 for i in range(len(Eref))]   # 1 eV: missing multiplet
        if Cn is not None and with_corr:
            for i in iso:
                if k[i] >= 0:
                    cm = np.array([X[:, k[i]] @ o @ X[:, k[i]] for o in ops])
                    r.extend(corr_weight * scale * (cm - Cn[i][iu]))
                else:
                    r.extend(np.full(len(ops), corr_weight * scale))
        return np.array(r)

    starts = [np.full(ng, v * scale) for v in (0.5, -0.5, 2.0, -2.0)]
    rng = np.random.default_rng(0)
    starts += [scale * rng.normal(size=ng) for _ in range(6)]
    if J0 is not None:
        starts.insert(0, np.asarray(J0, float))
    best = None
    for x0 in starts:
        # energies first (smooth landscape), then energies + correlations from there: the
        # relabelled solutions all fit the energies, the correlations pick the right one
        x1 = least_squares(lambda x: resid(x, False), x0, x_scale=scale).x
        sol = least_squares(resid, x1, x_scale=scale)
        if best is None or sol.cost < best.cost:
            best = sol
    E, S, X = model(best.x)
    # which J's are determined: rank of the Jacobian at the solution
    Jac = best.jac
    rank = np.linalg.matrix_rank(Jac, tol=1e-6 * max(1e-12, np.abs(Jac).max())) if Jac.size else 0
    fit_spin_model.rank = int(rank)
    fit_spin_model.corr_err = float("nan")
    if Cn is not None:
        cm = np.array([X[:, 0] @ o @ X[:, 0] for o in ops])
        fit_spin_model.corr_err = float(np.abs(cm - Cn[0][iu]).max())
    k = match(S)
    er = np.array([Eref[i] - E[k[i]] for i in range(len(Eref)) if k[i] >= 0])
    fit_spin_model.unmatched = int(np.sum(k < 0))       # ED states whose spin the model cannot give
    return best.x, E, S, float(np.sqrt(np.mean(er ** 2))) if len(er) else float("nan")


def superexchange(t, U):
    """Kinetic superexchange between two singly occupied sites: J = 4 t^2 / U."""
    return 4 * t ** 2 / U


def kondo_coupling(tau, eps_f, U, EF=0.0):
    """Schrieffer-Wolff Kondo coupling of a local moment hybridised (tau) with a band:
    J_K = 2 tau^2 [1/(EF - eps_f) + 1/(eps_f + U - EF)]; 8 tau^2/U at particle-hole symmetry."""
    return 2 * tau ** 2 * (1 / (EF - eps_f) + 1 / (eps_f + U - EF))


# ===========================================================================
# 4. atomistic mean-field Hubbard
# ===========================================================================
def mean_field_hubbard(H, n_up, n_dn, U=3.0, nk=1, kT=0.005, n0=None, mix=0.5, tol=1e-7,
                       maxiter=1000, periodic=True, history=6):
    """Collinear mean-field Hubbard on the atomistic TB (hubbard-package conventions).

    H: dict {R: H_R} of tbcore (periodic) or {0: H0} (finite). n_up, n_dn: electrons of each
    spin per cell (fixed). U: one value or one per orbital. n0: initial densities (2, N).
    Occupations: Fermi-Dirac with kT (eV) per spin channel on a uniform k grid of nk points. Pulay (DIIS) mixing of the
    densities with the given history (0: linear mixing).
    Returns dict(n=(2,N), E=energy per cell, Ef=(2,), bands per spin, iterations).
    """
    N = H[0].shape[0]
    ks = 2 * np.pi * np.arange(nk) / nk if periodic and nk > 1 else np.array([0.0])
    Hks = [tb.Hk(H, k) if periodic else H[0].astype(complex) for k in ks]
    nkp = len(ks)
    rng = np.random.default_rng(1)
    if n0 is None:
        n = np.array([np.full(N, n_up / N), np.full(N, n_dn / N)]) + 0.05 * rng.standard_normal((2, N))
    else:
        n = np.array(n0, float)
    q = (n_up, n_dn)

    def occupations(E, nel):
        """Occupations for nel electrons per cell in the (nk, N) eigenvalues."""
        e = np.sort(E.ravel())
        m = int(round(nel * nkp))
        if kT <= 0 or m == 0 or m == e.size:
            f = np.zeros(E.size)
            f[np.argsort(E.ravel(), kind="stable")[:m]] = 1.0
            return f.reshape(E.shape), (e[m - 1] if m else e[0])
        lo, hi = e[max(m - 1, 0)] - 20 * kT, e[min(m, e.size - 1)] + 20 * kT
        for _ in range(100):
            mu = 0.5 * (lo + hi)
            f = 0.5 * (1 - np.tanh(0.5 * (E - mu) / kT))
            if f.sum() > nel * nkp:
                hi = mu
            else:
                lo = mu
            if hi - lo < 1e-12:
                break
        return f, mu

    def step(n):
        new, Etot, Ef, bands = np.zeros((2, N)), 0.0, [], []
        for s in (0, 1):
            Es, Vs = [], []
            for Hk_ in Hks:
                e, v = np.linalg.eigh(Hk_ + np.diag(U * n[1 - s]))
                Es.append(e)
                Vs.append(v)
            Es = np.array(Es)
            f, mu = occupations(Es, q[s])
            for kk in range(nkp):
                new[s] += (np.abs(Vs[kk]) ** 2) @ f[kk]
                Etot += Es[kk] @ f[kk]
            Ef.append(mu)
            bands.append(Es)
        return new / nkp, Etot / nkp, Ef, bands

    xs, rs = [], []
    for it in range(maxiter):
        new, Etot, Ef, bands = step(n)
        res = (new - n).ravel()
        dn = np.abs(res).max()
        if dn < tol:
            break
        xs.append(n.ravel().copy())
        rs.append(res)
        xs, rs = xs[-history:], rs[-history:]
        if history and len(rs) > 1:
            # Pulay / DIIS: minimise |sum c_i r_i| with sum c_i = 1
            m = len(rs)
            B = np.empty((m + 1, m + 1))
            B[:m, :m] = np.array(rs) @ np.array(rs).T
            B[m, :], B[:, m], B[m, m] = 1.0, 1.0, 0.0
            rhs = np.zeros(m + 1)
            rhs[m] = 1.0
            try:
                c = np.linalg.solve(B, rhs)[:m]
                x = c @ np.array(xs) + mix * (c @ np.array(rs))
            except np.linalg.LinAlgError:
                x = n.ravel() + mix * res
        else:
            x = n.ravel() + mix * res
        n = np.clip(x.reshape(2, N), 0.0, 1.0)
    # energy of the converged densities: sum_occ e - U sum n_up n_dn (hubbard package)
    E = Etot - np.sum(np.asarray(U) * new[0] * new[1])   # U may be one value per atom
    return dict(n=new, E=E, Ef=np.array(Ef), bands=bands, ks=ks, iterations=it + 1, converged=dn < tol)


def polarized_start(N, n_up, n_dn, up_atoms=(), dn_atoms=(), amp=0.5):
    """Initial densities: uniform plus +amp spin up on up_atoms and on dn_atoms spin down."""
    n = np.array([np.full(N, n_up / N), np.full(N, n_dn / N)])
    for a in up_atoms:
        n[0, a] += amp
        n[1, a] -= amp
    for a in dn_atoms:
        n[0, a] -= amp
        n[1, a] += amp
    return np.clip(n, 0, 1)


# ===========================================================================
# 5. workflow: effective model -> interacting cluster -> ED -> spin model
# ===========================================================================
def ph_shift(terms, Ns):
    """One-body correction that writes the interaction in particle-hole symmetric form,
    i.e. relative to a half-filled reference (<c+_c c_d> = delta_cd / 2):
        U (n_up - 1/2)(n_dn - 1/2)  for the on-site terms, and in general
        - 1/2 sum_c v_abcc c+_a,up c_b,up - 1/2 sum_a v_aacd c+_c,dn c_d,dn  (+ constant).
    The fitted TB parameters already contain the average electrostatics of the half-filled
    system, so this avoids double counting it. Returns (M_up, M_dn) to add to the hopping."""
    Mu, Md = np.zeros((Ns, Ns)), np.zeros((Ns, Ns))
    for a, b, c, d, v in terms:
        if c == d:
            Mu[a, b] -= 0.5 * v
        if a == b:
            Md[c, d] -= 0.5 * v
    return Mu, Md


class SpinHubbardED(HubbardED):
    """HubbardED with (possibly) different one-body matrices for up and down spins."""

    def __init__(self, Tup, Tdn, terms, n_up, n_dn):
        super().__init__(Tup, terms, n_up, n_dn)
        self.Td = self.dn.one_body(np.asarray(Tdn, float))


def cluster_keep(model, ncells, keep_last=None, periodic=False):
    """Cluster indices (c * n_sites + n) of the sites kept when the last cell holds only the
    sites named in keep_last (None or empty = all): odd chains of a model with two sites per
    cell (e.g. A B A B A for a dimerised chain). Open chains only."""
    ns = len(model.sites)
    idx = list(range(ns * ncells))
    if keep_last:
        keep_last = list(keep_last)
        bad = [n for n in keep_last if n not in model.sites]
        if bad:
            raise ValueError(f"unknown site(s) {', '.join(bad)} for the last cell (sites: {', '.join(model.sites)})")
        if periodic:
            raise ValueError("a partial last cell needs an open chain (a ring must close on whole cells)")
        if ncells < 2:
            raise ValueError("a partial last cell needs at least 2 cells")
        c = ncells - 1
        idx = [i for i in idx if i // ns < c or model.sites[i % ns] in keep_last]
    return idx


def build_cluster(H, st, model, pv, ncells=2, periodic=True, U=3.0, mode="onsite",
                  thr=1e-3, EF=0.0, U_override=None, ph=True, keep_last=None):
    """Everything needed for ED of the effective model on a cluster of unit cells.

    U_override: dict site name -> on-site U (eV), replaces U sum|w|^4.
    keep_last: names of the sites kept in the last cell (odd chains; see cluster_keep).
    Returns dict(T, Tup, Tdn, terms, labels, U_site, w, spread, overlap, ns, ncells).
    """
    w, hR, smin = wannier_orbitals(H, st, model)
    ipr, n90 = orbital_spread(w)
    ns = len(model.sites)
    W = cluster_orbitals(w, ncells, periodic)
    T, labels = cluster_hopping(model, pv, ncells, periodic, shift=EF)
    Uons = np.tile(U * ipr, ncells)
    if U_override:
        for c in range(ncells):
            for n, name in enumerate(model.sites):
                if name in U_override:
                    Uons[c * ns + n] = U_override[name]
    kept = cluster_keep(model, ncells, keep_last, periodic)
    W, T, Uons = W[:, kept], T[np.ix_(kept, kept)], Uons[kept]
    labels = [labels[i] for i in kept]
    Uarg = Uons if (U_override or mode == "onsite") else U
    terms = interaction_terms(W, Uarg, mode, thr)
    Mu, Md = ph_shift(terms, len(T)) if ph else (0, 0)
    U_site = np.zeros(len(T))
    for a, b, c, d, v in terms:
        if a == b == c == d:
            U_site[a] = v
    return dict(T=T, Tup=T + Mu, Tdn=T + Md, terms=terms, labels=labels, U_site=U_site, w=w,
                spread=(ipr, n90), overlap=smin, ns=ns, ncells=ncells, periodic=periodic,
                kept=kept, site_of=[(i // ns, i % ns) for i in kept],
                W=W, atom_xy=cluster_atom_positions(st, w, ncells, periodic))


def cluster_atom_positions(st, w, ncells, periodic):
    """Positions of the rows of cluster_orbitals (atoms of the cells the orbitals reach;
    for a ring, the cells 0 .. ncells - 1)."""
    Rs = sorted(w)
    cells = range(ncells) if periodic else range(Rs[0], ncells + Rs[-1])
    return np.concatenate([st.xy + c * np.asarray(st.T) for c in cells])


def cluster_atoms_shown(cl):
    """Which atoms of the cells 0 .. ncells - 1 belong to the cluster, shape (ncells, N_atoms):
    all of them for whole cells; with a partial last cell, an atom is kept when most of its
    weight in the (full-cell) orbitals is on the kept sites."""
    w, nc, per = cl["w"], cl["ncells"], cl["periodic"]
    Rs = sorted(w)
    Na = w[Rs[0]].shape[0]
    out = np.ones((nc, Na), bool)
    if len(cl["kept"]) == nc * cl["ns"]:
        return out
    Wf = cluster_orbitals(w, nc, per)
    lo = 0 if per else Rs[0]
    rows = slice(-lo * Na, (nc - lo) * Na)                   # atoms of the cells 0 .. nc - 1
    tot = (Wf[rows] ** 2).sum(axis=1)
    kep = (Wf[rows][:, cl["kept"]] ** 2).sum(axis=1)
    P = cl["atom_xy"][rows]
    # decide per fused-ring unit when the chain is made of units linked by single bonds (atoms
    # on nodes of the orbitals have no weight of their own); otherwise per atom, with the
    # weight summed over the atom and its neighbours up to 3 A
    lab, sizes, _ = tb.ring_units(P)
    big = np.flatnonzero(sizes >= 6)
    keep = np.zeros(len(P), bool)
    from scipy.spatial import cKDTree
    nb = cKDTree(P).query_ball_point(P, 3.0)
    for i in range(len(P)):
        keep[i] = kep[nb[i]].sum() > 0.5 * max(tot[nb[i]].sum(), 1e-30)
    if len(big) > 1:
        for u in big:
            m = lab == u
            keep[m] = kep[m].sum() > 0.5 * max(tot[m].sum(), 1e-30)
    return keep.reshape(nc, Na)


def neutral_filling(model, fitres, ncells=1, nsites=None):
    """Electrons in the model bands for the neutral structure: 2 x (states of the fitted TB
    bands below E_F), per cell times ncells. Equal to one per site only if half of the model
    bands are occupied (not true e.g. with an odd number of electrons per cell).
    nsites: number of cluster sites when the last cell is partial (the count is then
    per_cell x nsites / sites per cell, rounded)."""
    E = fitres["E_ref"] - fitres["EF"]
    per_cell = 2 * np.mean(np.sum(E < -1e-9, axis=1) + 0.5 * np.sum(np.abs(E) <= 1e-9, axis=1))
    cells = ncells if nsites is None else nsites / len(model.sites)
    return int(round(per_cell * cells)), per_cell


def solve_cluster(cl, n_elec=None, nstates=20, two_sz=None, dense_limit=2500):
    """ED of a cluster from build_cluster. Default: half filling (one electron per site)."""
    Ns = len(cl["T"])
    n_elec = Ns if n_elec is None else n_elec
    two_sz = (n_elec % 2) if two_sz is None else two_sz
    n_up = (n_elec + two_sz) // 2
    n_dn = n_elec - n_up
    ed = SpinHubbardED(cl["Tup"], cl["Tdn"], cl["terms"], n_up, n_dn)
    E, X = ed.solve(nstates, dense_limit)
    X = clean_multiplets(E, X, Ns, n_up, n_dn)
    out = dict(E=E, dim=ed.dim, n_up=n_up, n_dn=n_dn, ed=ed, S=[], C=[], n=[])
    for k in range(len(E)):
        C = ed.spin_correlations(X[:, k])
        out["S"].append(ed.total_spin(X[:, k], C)[0])
        out["C"].append(C)
        nu, nd = ed.density(X[:, k])
        out["n"].append(nu + nd)
    out["S"], out["C"], out["n"] = np.array(out["S"]), np.array(out["C"]), np.array(out["n"])
    out["X"] = X
    return out


def bond_classes(pos, sym, max_d=4, tol=0.3):
    """Equivalence classes of the bonds (i, j, d) between the effective sites (site j in the
    cell d cells away from site i) under the translations and the point operations of the
    structure (tbcore.symmetry_ops). A site is mapped onto the site at the image of its
    position (centroid of its atoms). Returns a function (i, j, d) -> class id."""
    u, v, L, ops = sym["u"], sym["v"], sym["L"], sym["ops"]
    s0, p0 = pos @ u, pos @ v
    n = len(pos)

    def locate(s, p):
        for k in range(n):
            c = (s - s0[k]) / L
            if abs(p - p0[k]) < tol and abs(c - round(c)) * L < tol:
                return k, int(round(c))
        return None

    def canon(i, j, d):
        return min((i, j, d), (j, i, -d))

    parent = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i in range(n):
        for j in range(n):
            for d in range(-max_d, max_d + 1):
                if i == j and d == 0:
                    continue
                t = canon(i, j, d)
                find(t)
                for ss, sp_, a, b in ops:
                    A = locate(ss * s0[i] + a, sp_ * p0[i] + b)
                    B = locate(ss * (s0[j] + d * L) + a, sp_ * p0[j] + b)
                    if A is None or B is None:
                        continue
                    img = canon(A[0], B[0], B[1] - A[1])
                    if abs(img[2]) <= max_d:
                        parent[find(img)] = find(t)
    return lambda i, j, d: find(canon(i, j, d))


def spin_pairs(model, ncells, periodic, spin_names, max_dR=1, pos=None, T=None, sym=None, kept=None):
    """Pairs of spin sites on the cluster and their grouping into couplings J.

    With pos (site positions in cell 0, A), T (period) and sym (tbcore.symmetry_ops): bonds
    related by a translation or a point operation of the structure (mirror, glide, C2) share
    one J. With pos and T only: bonds between the same kinds of site with the same length
    share one J (a guess, wrong when the structure lacks the symmetry). Without them:
    grouping by translation only.
    On a ring, a pair separated by dR cells is also separated by dR - ncells: when both are
    within max_dR the two couplings act on the same pair and are one group.
    kept: the cluster's kept sites (cluster_keep) when the last cell is partial.
    Returns (cluster indices of the spin sites, pairs as indices into that list, group of every
    pair, group labels)."""
    ns = len(model.sites)
    sites = [c * ns + n for c in range(ncells) for n, name in enumerate(model.sites) if name in spin_names]
    if kept is not None:
        sites = [x for x in sites if x in kept]
    info = {s: (s // ns, s % ns) for s in sites}
    pairs, groups, labels, key2g, seen = [], [], [], {}, []
    cls = bond_classes(np.asarray(pos, float), sym, max_d=max(max_dR, ncells)) if (pos is not None and sym is not None) else None
    for ia, a in enumerate(sites):
        for ib, b in enumerate(sites):
            if ib <= ia:
                continue
            (ca, ia_), (cb, ib_) = info[a], info[b]
            na, nb = model.sites[ia_], model.sites[ib_]
            if periodic:
                images = sorted({d for d in (cb - ca, cb - ca - ncells, cb - ca + ncells) if abs(d) <= max_dR})
            else:
                images = [cb - ca] if abs(cb - ca) <= max_dR else []
            if not images:
                continue
            names = tuple(sorted((na, nb)))
            if pos is not None and sym is not None:
                key = tuple(sorted({cls(ia_, ib_, d) for d in images}))
                dists = tuple(sorted(round(float(np.linalg.norm(pos[ib_] + d * np.asarray(T) - pos[ia_])), 1)
                                     for d in images))
                lab = f"{names[0]}-{names[1]} d = " + " & ".join(f"{x:.1f}" for x in dists) + " A"
            elif pos is not None:
                dists = tuple(sorted(round(float(np.linalg.norm(pos[ib_] + d * np.asarray(T) - pos[ia_])), 1)
                                     for d in images))
                key = (names, dists)
                lab = f"{names[0]}-{names[1]} d = " + " & ".join(f"{x:.1f}" for x in dists) + " A"
            else:
                if len(images) == 1:
                    d = images[0]
                    key = min((na, nb, d), (nb, na, -d))
                    lab = f"{key[0]}(c)-{key[1]}(c{key[2]:+d})" if key[2] else f"{key[0]}(c)-{key[1]}(c)"
                else:
                    d = max(abs(x) for x in images)
                    key = (names[0], names[1], f"+-{d}")
                    lab = f"{names[0]}(c)-{names[1]}(c+-{d})"
            if len(images) > 1:
                lab += " [ring: both sides]"
            if key not in key2g:
                key2g[key] = len(labels)
                labels.append(lab)
                seen.append({names})
            seen[key2g[key]].add(names)
            pairs.append((ia, ib))
            groups.append(key2g[key])
    # a symmetry can relate bonds between different kinds of site (e.g. a glide Tu -> Td):
    # name all of them; labels must be unique (they name the J's in the text box)
    out = []
    for lab, nm in zip(labels, seen):
        if len(nm) > 1 and " d = " in lab:
            lab = "/".join(f"{a}-{b}" for a, b in sorted(nm)) + " d = " + lab.split(" d = ", 1)[1]
        base, k = lab, 2
        while lab in out:
            lab, k = f"{base} #{k}", k + 1
        out.append(lab)
    if kept is not None:
        sites = [list(kept).index(x) for x in sites]
    return sites, pairs, groups, out


# ===========================================================================
# 6. U of the effective model from the atomistic mean field (t fixed)
# ===========================================================================
def mfh_configurations(H, st, site_atoms, U=3.0, nk=24, kT=0.005, which=("fm", "af", "pm")):
    """Atomistic mean-field Hubbard in the standard magnetic states.

    site_atoms: atom lists of the sites that carry the moments. FM: all of them up. AFM: by
    sublattice of their first atom (alternating the sites if all are on one sublattice).
    S_z per cell: number of moment sites (FM) or the sublattice imbalance (AFM), with the
    parity of the electron count. Returns dict name -> result of mean_field_hubbard, with
    'n_up', 'n_dn' (electrons of each spin per cell) added."""
    N, ne = st.N, st.n_electrons
    nsp = len(site_atoms)
    allowed = lambda s: s if (ne - s) % 2 == 0 else s - 1
    out = {}
    if "fm" in which:
        s = allowed(nsp)
        up = [a for atoms in site_atoms for a in atoms]
        nu, nd = (ne + s) // 2, (ne - s) // 2
        out["fm"] = mean_field_hubbard(H, nu, nd, U=U, nk=nk, kT=kT, n0=polarized_start(N, nu, nd, up, []))
        out["fm"].update(n_up=nu, n_dn=nd)
    if "af" in which:
        col = tb.sublattice_colors(st)
        cols = [col[atoms[0]] if col is not None else k % 2 for k, atoms in enumerate(site_atoms)]
        if len(set(cols)) < 2:
            cols = [k % 2 for k in range(nsp)]
        up = [a for c, atoms in zip(cols, site_atoms) if c == 0 for a in atoms]
        dn = [a for c, atoms in zip(cols, site_atoms) if c == 1 for a in atoms]
        s = allowed(abs(sum(1 if c == 0 else -1 for c in cols)))
        nu, nd = (ne + s) // 2, (ne - s) // 2
        out["af"] = mean_field_hubbard(H, nu, nd, U=U, nk=nk, kT=kT, n0=polarized_start(N, nu, nd, up, dn))
        out["af"].update(n_up=nu, n_dn=nd)
    if "pm" in which:
        nu, nd = ne // 2, ne - ne // 2
        out["pm"] = mean_field_hubbard(H, nu, nd, U=U, nk=nk, kT=kT, n0=np.full((2, N), ne / (2 * N)))
        out["pm"].update(n_up=nu, n_dn=nd)
    return out


def model_H_dict(model, pv, onsite_shift=None):
    """The effective model as {R: H_R} (R >= 0) in the convention of tbcore.Hk.
    onsite_shift: optional array added to the on-site energies."""
    n = len(model.sites)
    Hm = {0: np.zeros((n, n))}
    for i, j, R, p, s in model.terms:
        v = s * pv[p]
        if i == j and R == 0:
            Hm[0][i, i] += v
        elif R == 0:
            Hm[0][i, j] += v
            Hm[0][j, i] += v
        else:
            Hm.setdefault(R, np.zeros((n, n)))
            Hm[R][i, j] += v
    if onsite_shift is not None:
        Hm[0] = Hm[0] + np.diag(onsite_shift)
    return Hm


def _occupations(E, nel, kT):
    """Fermi-Dirac occupations of the (nk, nb) eigenvalues for nel electrons per cell
    (same rule as mean_field_hubbard)."""
    nkp = E.shape[0]
    e = np.sort(E.ravel())
    m = int(round(nel * nkp))
    if kT <= 0 or m == 0 or m == e.size:
        f = np.zeros(E.size)
        f[np.argsort(E.ravel(), kind="stable")[:m]] = 1.0
        return f.reshape(E.shape)
    lo, hi = e[max(m - 1, 0)] - 20 * kT, e[min(m, e.size - 1)] + 20 * kT
    for _ in range(200):
        mu = 0.5 * (lo + hi)
        f = 0.5 * (1 - np.tanh(0.5 * (E - mu) / kT))
        if f.sum() > nel * nkp:
            hi = mu
        else:
            lo = mu
        if hi - lo < 1e-13:
            break
    return f


def mfh_reference(H, st, model, sol, U, kT):
    """What the effective model has to reproduce from an atomistic mean-field solution.

    For each spin: the energies of the bands of the model window (tbcore.select, same rule
    as the fit of tab 2, with the non-magnetic band count) on the k grid of the solution,
    and the occupation of every effective orbital n_i,s = sum_occ |<w_i,k|psi>|^2 / nk
    (w_i,k: Loewdin orbitals of the non-magnetic TB). Also the electrons of each spin in
    the model bands (n_up - n_dn of the whole cell sits in them)."""
    G = tb.trial_orbitals(model, st.N)
    if G is None:
        raise ValueError("every effective site must sit on atoms")
    nocc = st.n_electrons // 2 + st.n_electrons % 2
    ks, n = sol["ks"], sol["n"]
    ns = len(model.sites)
    Wk = []
    for k in ks:
        e, V = np.linalg.eigh(tb.Hk(H, k))
        sel = tb.select(model, e, V, nocc, G)
        A = V[:, sel].conj().T @ G
        w, Uv = np.linalg.eigh(A.conj().T @ A)
        Wk.append(V[:, sel] @ A @ Uv @ np.diag(np.clip(w, 1e-12, None) ** -0.5) @ Uv.conj().T)
    E_sel, occ, smin = [], np.zeros((2, ns)), 1.0
    for s_ in (0, 1):
        nel = sol["n_up"] if s_ == 0 else sol["n_dn"]
        Es, Vs = [], []
        for k in ks:
            e, V = np.linalg.eigh(tb.Hk(H, k) + np.diag(np.asarray(U) * n[1 - s_]))
            Es.append(e)
            Vs.append(V)
        Es = np.array(Es)
        f = _occupations(Es, nel, kT)
        rows = []
        for kk in range(len(ks)):
            sel = tb.select(model, Es[kk], Vs[kk], nocc, G)
            rows.append(Es[kk, sel])
            P = np.abs(Wk[kk].conj().T @ Vs[kk]) ** 2            # (ns, nbands)
            occ[s_] += P @ f[kk] / len(ks)
            A = Vs[kk][:, sel].conj().T @ Wk[kk]
            smin = min(smin, np.linalg.eigvalsh(A.conj().T @ A).min())
        E_sel.append(np.array(rows))
    nmod = occ.sum(axis=1)
    return dict(ks=ks, E=E_sel, n_site=occ, m_site=occ[0] - occ[1], n_mod=nmod, overlap=smin,
                E_tot=sol["E"], n_up=sol["n_up"], n_dn=sol["n_dn"])


def model_mfh(model, pv, U_site, n_up, n_dn, nk, kT, n0=None, ph=True, tol=1e-9):
    """Self-consistent mean field of the effective model,
        H_s(k) = H_t(k) + sum_i U_i (<n_i,-s> - 1/2) n_i,s        (ph=True),
    with n_up, n_dn electrons per cell (may be fractional). Returns mean_field_hubbard's dict."""
    U_site = np.asarray(U_site, float)
    Hm = model_H_dict(model, pv, onsite_shift=-0.5 * U_site if ph else None)
    return mean_field_hubbard(Hm, n_up, n_dn, U=U_site, nk=nk, kT=kT, n0=n0, tol=tol, maxiter=3000)


def fit_U_to_mfh(model, pv, refs, groups, U0, kT, moment_weight=0.1, ph=True):
    """Fit the on-site U_i of the effective model (hoppings and energies fixed) so that its
    self-consistent mean field reproduces the atomistic mean-field references.

    refs: list of mfh_reference results (one per magnetic state), with 'n_up_mod',
    'n_dn_mod' (electrons of each spin per cell in the model) set by the caller.
    groups: group index of every site (sites of a group share one U). U0: start per group.
    Residuals: spin-resolved band energies of every state (both sorted at every k) minus a
    free rigid offset per state (the Hartree shift of the whole spectrum), plus
    moment_weight (eV per unit moment) x (m_i model - m_i atomistic).
    Returns dict(U_group, U_site, offsets, rms per state (eV), moments, model solutions)."""
    groups = np.asarray(groups)
    ng = groups.max() + 1
    nk = len(refs[0]["ks"])
    warm = [np.clip(r["n_site"] / np.maximum(r["n_site"].sum(axis=1, keepdims=True), 1e-12)
                    * np.array([[r["n_up_mod"]], [r["n_dn_mod"]]]), 0, 1) for r in refs]
    last = {}

    def solve(x):
        Us = np.clip(x[:ng], 0, None)[groups]
        sols = []
        for c, r in enumerate(refs):
            # always start from the projected atomistic densities: the mean field can have several
            # solutions (e.g. the sign of a small induced moment) and this keeps the model on the
            # branch of the reference state
            sol = model_mfh(model, pv, Us, r["n_up_mod"], r["n_dn_mod"], nk, kT, n0=warm[c], ph=ph, tol=1e-11)
            sols.append(sol)
        return Us, sols

    def resid(x):
        Us, sols = solve(x)
        out = []
        for c, (r, sol) in enumerate(zip(refs, sols)):
            off = x[ng + c]
            for s_ in (0, 1):
                out.append((r["E"][s_] - off - sol["bands"][s_]).ravel())
            if moment_weight:
                out.append(moment_weight * ((sol["n"][0] - sol["n"][1]) - r["m_site"]))
        last["sols"] = sols
        return np.concatenate(out)

    offs0 = []
    for c, r in enumerate(refs):         # start: align the centres of the spectra
        Us0 = np.asarray(U0, float)[groups]
        s0 = model_mfh(model, pv, Us0, r["n_up_mod"], r["n_dn_mod"], nk, kT, n0=warm[c], ph=ph)
        offs0.append(np.mean([r["E"][s_].mean() - s0["bands"][s_].mean() for s_ in (0, 1)]))
    lb = np.concatenate([np.zeros(ng), np.full(len(refs), -np.inf)])
    # the mean field can switch between solutions as U changes (e.g. the sign of a small
    # induced moment), so the cost has several minima: several starts, the best is kept
    U0 = np.asarray(U0, float)
    starts = [U0, 0.5 * U0, 2.0 * U0]
    for g in range(ng):
        for f_ in (0.3, 3.0):
            u = U0.copy()
            u[g] *= f_
            starts.append(u)
    sol = None
    for u in starts:
        # the mean field is converged to 1e-11, so finite differences of 1e-4 are clean; the
        # tolerances are tight because weakly polarised sites change the bands by < 1 meV
        trial = least_squares(resid, np.concatenate([u, offs0]), bounds=(lb, np.inf), diff_step=1e-4,
                              ftol=1e-14, xtol=1e-12, gtol=1e-14,
                              x_scale=np.concatenate([np.full(ng, 0.1), np.full(len(refs), 0.1)]))
        if sol is None or trial.cost < sol.cost:
            sol = trial
    resid(sol.x)
    Us, sols = solve(sol.x)
    rms, moments = [], []
    for c, (r, s) in enumerate(zip(refs, sols)):
        d = np.concatenate([(r["E"][s_] - sol.x[ng + c] - s["bands"][s_]).ravel() for s_ in (0, 1)])
        rms.append(float(np.sqrt(np.mean(d ** 2))))
        moments.append((r["m_site"], s["n"][0] - s["n"][1]))
    J = sol.jac[:, :ng]
    rank = int(np.linalg.matrix_rank(J, tol=1e-6 * max(1e-12, np.abs(J).max()))) if J.size else 0
    # sensitivity: rms change of the fitted quantities for a change of 0.1 eV of each U
    sens = 0.1 * np.sqrt(np.mean(J ** 2, axis=0))
    return dict(U_group=sol.x[:ng], U_site=Us, offsets=sol.x[ng:], rms=rms, moments=moments,
                sols=sols, rank=rank, cost=float(sol.cost), sensitivity=sens)


# ===========================================================================
# 7. charge excitations (N +- 1) and tip spectra
# ===========================================================================
def _add_electron(V, Ns, nu, nd, a, spin):
    """c+_a,spin |V> in the sector (nu + 1, nd) or (nu, nd + 1)."""
    if spin == 0:
        return _annihilator(Ns, nu + 1, a).T @ V
    return (-1.0) ** nu * _rmul(V, _annihilator(Ns, nd + 1, a))


def _remove_electron(V, Ns, nu, nd, a, spin):
    """c_a,spin |V> in the sector (nu - 1, nd) or (nu, nd - 1)."""
    if spin == 0:
        return _annihilator(Ns, nu, a) @ V
    return (-1.0) ** nu * _rmul(V, _annihilator(Ns, nd, a).T)


def charge_excitations(res, cl, nstates=12, dense_limit=2500):
    """Lowest states with one electron more / less than the ground state and their Dyson
    amplitudes d_f(a) = <f| c+_a,s |0>  (addition)  or  <f| c_a,s |0>  (removal).

    Excitation energies relative to the Fermi level of the model (the Hamiltonian is written
    around E_F): addition w = E_f(N+1) - E_0(N) (> 0 above E_F), removal w = E_0(N) - E_f(N-1).
    The tip spectrum at an orbital v (v_a: weights on the effective sites, e.g. w_a(x) for an
    atom x) is A_v(w) = sum_f |sum_a v_a d_f(a)|^2 delta(w - w_f) (both spins, averaged over
    the ground level). When the ground state has S_z = 0 the two spins are equivalent and only
    spin up is computed (weights doubled).
    Returns list of dict(kind, spin, w (k,), D (n_ground, k, Ns), mult, dim)."""
    Ns, nu, nd = res["ed"].Ns, res["n_up"], res["n_dn"]
    du, dd = comb(Ns, nu), comb(Ns, nd)
    g_level = levels(res["E"], res["S"])[0]
    E0 = res["E"][g_level[0]]
    G = [res["X"][:, g].reshape(du, dd) for g in g_level]
    spins = (0,) if nu == nd else (0, 1)
    out = []
    for kind in ("add", "remove"):
        for s in spins:
            nu2 = nu + (1 if kind == "add" else -1) * (s == 0)
            nd2 = nd + (1 if kind == "add" else -1) * (s == 1)
            if not (0 <= nu2 <= Ns and 0 <= nd2 <= Ns):
                continue
            ed = SpinHubbardED(cl["Tup"], cl["Tdn"], cl["terms"], nu2, nd2)
            E, X = ed.solve(min(nstates, ed.dim), dense_limit)
            D = np.zeros((len(G), len(E), Ns))
            for gi, V in enumerate(G):
                op = _add_electron if kind == "add" else _remove_electron
                Phi = np.array([op(V, Ns, nu, nd, a, s).ravel() for a in range(Ns)])   # (Ns, dim2)
                D[gi] = X.T @ Phi.T
            w = (E - E0) if kind == "add" else (E0 - E)
            out.append(dict(kind=kind, spin=s, w=w, D=D, mult=2 if nu == nd else 1, dim=ed.dim))
    return out


def tip_peaks(ch, v):
    """Peaks (energy, weight) of the single-particle tip spectrum at the orbital v (weights on
    the effective sites) from charge_excitations. Weight: sum over spins, average over the
    ground level; the sum over all states of one spin gives <n_v,s> (removal) and
    1 - <n_v,s> (addition) for a normalised v."""
    v = np.asarray(v, float)
    E, Wt, kind = [], [], []
    for c in ch:
        amp = np.einsum("gka,a->gk", c["D"], v)
        wt = c["mult"] * np.mean(amp ** 2, axis=0)
        E.extend(c["w"])
        Wt.extend(wt)
        kind.extend([c["kind"]] * len(wt))
    return np.array(E), np.array(Wt), np.array(kind)


def dyson_map(ch, kind, index, W, spin=None):
    """Weight of the charge excitation (kind, spin sector, index of the state in its sector) on
    every atom: |sum_a W[x, a] d_f(a)|^2 (ground level averaged, times 2 when only spin up is
    computed) - the STM map at that peak. spin: 0 (up) or 1 (down) = the sector of the state;
    it matters when both spins are computed (odd electron number). None: the first sector of
    that kind (the only one when N_up = N_dn)."""
    tot = 0.0
    for c in ch:
        if c["kind"] != kind or index >= len(c["w"]) or (spin is not None and c["spin"] != spin):
            continue
        amp = np.einsum("xa,ga->gx", W, c["D"][:, index, :])
        tot = tot + c["mult"] * np.mean(amp ** 2, axis=0)
        break
    return tot


def inelastic_amplitudes(res):
    """Transition spin densities of every level from the ground level (for spin-flip weights at
    any tip position): list over levels of lists of tau dicts (one per ground/final state)."""
    Ns, nu, nd = res["ed"].Ns, res["n_up"], res["n_dn"]
    lev = levels(res["E"], res["S"])
    g_level = lev[0]
    out = []
    for L in lev:
        out.append([transition_amplitudes(res["X"][:, f], res["X"][:, g], Ns, nu, nd) for g in g_level for f in L])
    return lev, out, len(g_level)


def inelastic_weights_at(taus, n_ground, v):
    """Spin-flip weight of every level for a tip on the orbital v (site weights)."""
    v = np.asarray(v, float)[None, :]
    return np.array([sum(float(spin_flip_weight(t, v)[0]) for t in tl) / n_ground for tl in taus])


def broadened(E, Wt, grid, width, shape="gauss"):
    """Sum of peaks on a grid."""
    x = grid[:, None] - np.asarray(E)[None, :]
    if shape == "lorentz":
        k = width / np.pi / (x ** 2 + width ** 2)
    else:
        k = np.exp(-0.5 * (x / width) ** 2) / (np.sqrt(2 * np.pi) * width)
    return k @ np.asarray(Wt)


def iets_dIdV(dE, W, bias, width):
    """Inelastic spin-flip conductance: a step of height W_f at |eV| = dE_f (both polarities),
    smoothed with a Fermi-like width (same units as dE)."""
    b = np.abs(np.asarray(bias))[:, None]
    return (0.5 * (1 + np.tanh((b - np.asarray(dE)[None, :]) / (2 * width)))) @ np.asarray(W)


def mfh_scan(H, st, U=3.0, nk=24, kT=0.005, two_sz=None):
    """Atomistic mean-field Hubbard without choosing any site: a non-magnetic solution and,
    for every total 2 S_z per cell in two_sz (default: the smallest allowed, the sublattice
    imbalance |N_A - N_B| of Lieb's theorem and one more unit), two starting spin patterns,
    spins by sublattice (antiparallel on the two sublattices) and all spins up; the lower
    solution is kept. Returns a list of dict(label, two_sz, start, sol) sorted by energy;
    sol is the result of mean_field_hubbard with 'n_up', 'n_dn' added."""
    N, ne = st.N, st.n_electrons
    periodic = bool(getattr(st, "periodic", True)) and np.linalg.norm(st.T) > 0
    Hm = H if periodic else {0: H[0]}
    col = tb.sublattice_colors(st)
    if col is None:
        col = np.arange(N) % 2
    A, B = list(np.where(col == 0)[0]), list(np.where(col == 1)[0])
    par = ne % 2
    lieb = abs(len(A) - len(B))
    fix = lambda s: s if (s - par) % 2 == 0 else s + 1
    if two_sz is None:
        two_sz = sorted({par, fix(lieb), fix(lieb) + 2})
    two_sz = [int(s) for s in two_sz if 0 <= s <= ne and (s - par) % 2 == 0]
    out = []
    nu, nd = ne // 2 + par, ne // 2
    pm = mean_field_hubbard(Hm, nu, nd, U=U, nk=nk, kT=kT, n0=np.full((2, N), ne / (2 * N)), periodic=periodic)
    pm.update(n_up=nu, n_dn=nd)
    out.append(dict(label=f"non-magnetic (S_z = {par / 2:g})", two_sz=par, start="uniform", sol=pm))
    for s in two_sz:
        nu, nd = (ne + s) // 2, (ne - s) // 2
        best = None
        for name, up, dn in (("sublattice", A, B), ("all up", list(range(N)), [])):
            sol = mean_field_hubbard(Hm, nu, nd, U=U, nk=nk, kT=kT, periodic=periodic,
                                     n0=polarized_start(N, nu, nd, up, dn, amp=0.3))
            if best is None or sol["E"] < best[1]["E"] - 1e-9:
                best = (name, sol)
        best[1].update(n_up=nu, n_dn=nd)
        out.append(dict(label=f"S_z = {s / 2:g} ({best[0]} start)", two_sz=s, start=best[0], sol=best[1]))
    for o in out:
        o["mmax"] = float(np.abs(o["sol"]["n"][0] - o["sol"]["n"][1]).max())
    out.sort(key=lambda o: o["sol"]["E"])
    return out


def midgap(sol):
    """Energy reference of a mean-field solution: the middle between the highest occupied and
    the lowest empty state of the two spins (the chemical potential of a gapped solution is
    anywhere in the gap; the Fermi-Dirac search returns a value near its top)."""
    homo, lumo = -np.inf, np.inf
    for s_, nel in ((0, sol["n_up"]), (1, sol["n_dn"])):
        e = np.sort(np.asarray(sol["bands"][s_]).ravel())
        m = int(round(nel * len(sol["ks"])))
        if m > 0:
            homo = max(homo, e[m - 1])
        if m < e.size:
            lumo = min(lumo, e[m])
    if not np.isfinite(homo):
        return lumo
    if not np.isfinite(lumo):
        return homo
    return 0.5 * (homo + lumo)


# ===========================================================================
# 8. singlet-triplet gap from the mean field of a finite system
# ===========================================================================
def uhf_density_matrices(H0, sol, U, kT=0.005):
    """One-body density matrices P_up, P_dn (N x N) of a converged finite mean-field solution
    (mean_field_hubbard with periodic=False): the occupied orbitals of H_s = H0 + U <n_-s>,
    filled with the same rule as the solver."""
    n = sol["n"]
    out = []
    for s_, nel in ((0, sol["n_up"]), (1, sol["n_dn"])):
        E, V = np.linalg.eigh(H0 + np.diag(np.asarray(U) * n[1 - s_]))
        f = _occupations(E[None, :], nel, kT)[0]
        out.append((V * f) @ V.T)
    return out


def uhf_s2(H0, sol, U, kT=0.005):
    """<S^2> of the unrestricted Hartree-Fock (mean-field) determinant:
        <S^2> = S_z (S_z + 1) + N_dn - sum_{i occ up, j occ dn} |<phi_i|phi_j>|^2
              = S_z (S_z + 1) + N_dn - Tr(P_up P_dn)
    (orthonormal pz basis). A pure singlet has 0, a triplet 2; a broken-symmetry determinant of
    two separated radicals has 1 (an equal mixture of singlet and triplet)."""
    Pu, Pd = uhf_density_matrices(H0, sol, U, kT)
    nu, nd = np.trace(Pu), np.trace(Pd)
    sz = 0.5 * (nu - nd)
    return float(sz * (sz + 1) + nd - np.sum(Pu * Pd))


def bond_units(xy, cut=tb.A_CC + 0.2):
    """Units of a pi system: the connected pieces left when the bridges of the bond graph
    (bonds in no ring, e.g. the single bond between two olympicenyl units) are cut. Rings of
    any size stay whole. Returns (labels per atom, bonds as an (n, 2) array)."""
    from scipy.spatial import cKDTree
    xy = np.asarray(xy, float)
    N = len(xy)
    bonds = cKDTree(xy).query_pairs(cut, output_type="ndarray")
    adj = [[] for _ in range(N)]
    for k, (i, j) in enumerate(bonds):
        adj[i].append((j, k))
        adj[j].append((i, k))
    disc, low, bridge, t = [-1] * N, [0] * N, set(), 0
    for r in range(N):                                   # iterative Tarjan bridge search
        if disc[r] >= 0:
            continue
        disc[r] = low[r] = t
        t += 1
        stack = [(r, -1, iter(adj[r]))]
        while stack:
            v, pe, it = stack[-1]
            for w, k in it:
                if k == pe:
                    continue
                if disc[w] < 0:
                    disc[w] = low[w] = t
                    t += 1
                    stack.append((w, k, iter(adj[w])))
                    break
                low[v] = min(low[v], disc[w])
            else:
                stack.pop()
                if stack:
                    u = stack[-1][0]
                    low[u] = min(low[u], low[v])
                    if low[v] > disc[u]:
                        bridge.add(pe)
    par = list(range(N))

    def find(x):
        while par[x] != x:
            par[x] = par[par[x]]
            x = par[x]
        return x
    for k, (i, j) in enumerate(bonds):
        if k not in bridge:
            par[find(i)] = find(j)
    _, lab = np.unique([find(i) for i in range(N)], return_inverse=True)
    return lab, bonds


def spin_centres(xy, m, thr=0.2, min_moment=0.3):
    """Groups of atoms that carry a local moment in a collinear mean-field solution with spin
    density m (n_up - n_dn per atom).

    Units linked by single bonds (bond_units: the bridges of the bond graph are cut, e.g. the
    olympicenyl radicals of a chain) are separate: a unit with |sum m| >= min_moment is one centre. Inside a fused unit whose
    moments cancel (e.g. the two zigzag ends of a graphene ribbon) the centres are the bonded
    pieces of atoms with |m| > thr max|m| that carry |sum m| >= min_moment.
    Returns the list of atom-index arrays, ordered along the long axis of the structure."""
    xy = np.asarray(xy, float)
    m = np.asarray(m, float)
    if len(xy) == 0 or np.abs(m).max() < 1e-6:
        return []
    from scipy.spatial import cKDTree
    lab, bonds = bond_units(xy)
    sizes = np.bincount(lab)
    tree = cKDTree(xy)
    # local net moment: the spin density alternates between the sublattices inside a radical,
    # so the sum over an atom and its first and second neighbours gives the sign of the
    # radical it belongs to
    near = tree.query_ball_point(xy, np.sqrt(3) * tb.A_CC + 0.2)
    loc = np.array([m[k].sum() for k in near])
    active = np.abs(loc) > thr * np.abs(loc).max()
    centres = []
    for u in range(len(sizes)):
        atoms = np.flatnonzero(lab == u)
        if len(atoms) < 3:                                 # dangling atoms are not radicals
            continue
        if abs(m[atoms].sum()) >= min_moment:
            centres.append(atoms)
            continue
        # fused unit whose moments cancel (e.g. a ribbon with two zigzag ends): bonded pieces of
        # atoms with a large local net moment of the same sign
        sel = set(int(a) for a in atoms if active[a])
        par = {a: a for a in sel}

        def find(x):
            while par[x] != x:
                par[x] = par[par[x]]
                x = par[x]
            return x
        for i, j in bonds:
            if i in sel and j in sel and loc[i] * loc[j] > 0:
                par[find(i)] = find(j)
        groups = {}
        for a in sel:
            groups.setdefault(find(a), []).append(a)
        for g in groups.values():
            g = np.array(sorted(g))
            if abs(loc[g].sum()) >= min_moment and len(g) >= 3:
                centres.append(g)
    if not centres:
        return []
    c = xy - xy.mean(axis=0)
    axis = np.linalg.svd(c, full_matrices=False)[2][0]
    centres.sort(key=lambda g: float(xy[g].mean(axis=0) @ axis))
    return centres


def _spin_patterns(n, two_sz, ref, max_patterns):
    """Sign patterns (+-1 per centre) with sum = two_sz, the closest to the reference pattern
    ref (or to its global flip) first, at most max_patterns."""
    from itertools import product
    if (n - two_sz) % 2 or abs(two_sz) > n:
        return []
    pats = [np.array(p) for p in product((1, -1), repeat=n) if sum(p) == two_sz]
    ref = np.asarray(ref)
    pats.sort(key=lambda p: min(np.sum(p != ref), np.sum(p != -ref)))
    return pats[:max_patterns]


def mfh_spin_gap(H0, st, results, U, kT=0.005, patterns=True, max_patterns=40, centres=None):
    """Singlet-triplet gap (in general: lowest S_z -> S_z + 1) from the mean field of a finite
    system, H0 = its TB Hamiltonian, results = mfh_scan output.

    The mean field gives broken-symmetry determinants, not spin eigenstates. Three estimates:
      raw        E(S_z + 1) - E(S_z), the energy of the mean-field states;
      Yamaguchi  spin projection with the <S^2> of the two determinants (assumes E linear in
                 <S^2>, exact for two spins with a Heisenberg coupling):
                     Delta = [E_HS - E_LS] [S_h(S_h + 1) - S_l(S_l + 1)] / (<S^2>_HS - <S^2>_LS)
                 for two separated radicals <S^2>_BS = 1 and Delta = 2 (E_T - E_BS) (Noodleman);
                 for a closed-shell singlet <S^2> = 0 and Delta = E_T - E_S;
      Heisenberg the collinear states are mapped on an Ising model of the spin centres,
                 E = E0 + sum_<ab> J_ab s_a s_b (s = +-1/2, nearest centres along the chain),
                 and H = sum J_ab S_a.S_b is diagonalised exactly. Needed with three or more
                 radicals, where the two-state projection is not valid.
    patterns: also converge the mean field from the lowest state with the moments of some
    centres flipped (every sign pattern with the two S_z, closest first, at most max_patterns
    each), which finds the low-lying collinear states the sublattice / all-up starts miss.
    Returns a dict (energies in eV) or None if the scan has no state with S_z + 1."""
    N, ne = st.N, st.n_electrons
    par = ne % 2
    sols = [dict(two_sz=o["two_sz"], sol=o["sol"], label=o["label"]) for o in results]
    s_lo = min(o["two_sz"] for o in sols)
    s_lo = s_lo if (s_lo - par) % 2 == 0 else par

    def lowest(two_sz):
        c = [o for o in sols if o["two_sz"] == two_sz and o["sol"]["converged"]]
        return min(c, key=lambda o: o["sol"]["E"]) if c else None

    ls = lowest(s_lo)
    if ls is None:
        return None
    ref = ls if np.abs(ls["sol"]["n"][0] - ls["sol"]["n"][1]).max() > 0.05 else lowest(s_lo + 2)
    if centres is None:
        centres = spin_centres(st.xy, ref["sol"]["n"][0] - ref["sol"]["n"][1]) if ref is not None else []
    nc = len(centres)
    mom = lambda sol: np.array([float((sol["n"][0] - sol["n"][1])[g].sum()) for g in centres])
    added = 0
    if patterns and 2 <= nc <= 10 and ref is not None:
        base = ref["sol"]["n"]
        s_ref = np.sign(mom(ref["sol"]))
        for t in (s_lo, s_lo + 2):
            nu, nd = (ne + t) // 2, (ne - t) // 2
            for p in _spin_patterns(nc, t, s_ref, max_patterns):
                n0 = base.copy()
                for g, s_new, s_old in zip(centres, p, s_ref):
                    if s_new != s_old:
                        n0[:, g] = n0[::-1][:, g]          # swap up and down on that centre
                sol = mean_field_hubbard({0: H0}, nu, nd, U=U, kT=kT, n0=np.clip(n0, 0, 1), periodic=False)
                sol.update(n_up=nu, n_dn=nd)
                lab = "".join("+" if x > 0 else "-" for x in p)
                sols.append(dict(two_sz=t, sol=sol, label=f"S_z = {t / 2:g} (pattern {lab})"))
                added += 1
    ls, hs = lowest(s_lo), lowest(s_lo + 2)
    if hs is None:
        return None
    Sl, Sh = s_lo / 2, s_lo / 2 + 1
    s2l, s2h = uhf_s2(H0, ls["sol"], U, kT), uhf_s2(H0, hs["sol"], U, kT)
    raw = hs["sol"]["E"] - ls["sol"]["E"]
    den = s2h - s2l
    yam = raw * (Sh * (Sh + 1) - Sl * (Sl + 1)) / den if den > 0.05 else float("nan")
    closed = np.abs(ls["sol"]["n"][0] - ls["sol"]["n"][1]).max() < 0.05
    # <S^2> of the low-spin determinant if the centres were ideal spins 1/2 (no overlap):
    # S_z (S_z + 1) + number of centres pointing down; a smaller value means delocalised moments
    s2_ideal = float("nan")
    if nc >= 2 and not closed:
        Ml = np.sign(mom(ls["sol"]))
        s2_ideal = Sl * (Sl + 1) + float(min(np.sum(Ml > 0), np.sum(Ml < 0)))
    out = dict(two_sz=(s_lo, s_lo + 2), LS=ls, HS=hs, E_LS=ls["sol"]["E"], E_HS=hs["sol"]["E"], s2_LS=s2l,
               s2_HS=s2h, raw=float(raw), yamaguchi=float(yam), centres=centres, n_centres=nc, added=added,
               closed_shell=bool(closed), s2_ideal=s2_ideal, heis=None)
    # Heisenberg mapping of the collinear states
    if nc >= 2:
        E, rows, seen = [], [], set()
        bonds = [(a, a + 1) for a in range(nc - 1)]
        m_ref = np.abs(mom(ref["sol"]))
        for o in sols:
            if not o["sol"]["converged"]:
                continue
            M = mom(o["sol"])
            if np.any(np.abs(M) < 0.5 * m_ref):            # not a collinear state of the same centres
                continue
            sg = tuple(int(x) for x in np.sign(M))
            key = min(sg, tuple(-x for x in sg))
            if key in seen:
                continue
            seen.add(key)
            E.append(o["sol"]["E"])
            rows.append([1.0] + [0.25 * sg[a] * sg[b] for a, b in bonds])
        A, E = np.array(rows), np.array(E)
        if len(E) >= 2:
            rank = np.linalg.matrix_rank(A)
            uniform = rank < A.shape[1]
            if uniform:                                    # not enough states: one J for all bonds
                A = np.c_[A[:, 0], A[:, 1:].sum(axis=1)]
            if np.linalg.matrix_rank(A) == A.shape[1]:
                x, *_ = np.linalg.lstsq(A, E, rcond=None)
                Jb = [x[1]] * len(bonds) if uniform else list(x[1:])
                rms = float(np.sqrt(np.mean((A @ x - E) ** 2)))
                hed = HeisenbergED(nc)
                Eh, Shs = hed.spectrum({b: j for b, j in zip(bonds, Jb)})
                Eh = Eh - Eh[0]
                S0 = round(2 * Shs[0]) / 2
                up = [k for k in range(1, len(Eh)) if abs(Shs[k] - (S0 + 1)) < 0.25]
                dn = [k for k in range(1, len(Eh)) if abs(Shs[k] - (S0 - 1)) < 0.25]
                out["heis"] = dict(J=[float(j) for j in Jb], bonds=bonds, uniform=uniform, rms=rms,
                                   n_states=len(E), E=Eh[:12], S=np.round(2 * Shs[:12]) / 2, S0=S0,
                                   gap_up=float(Eh[up[0]]) if up else float("nan"),
                                   gap_dn=float(Eh[dn[0]]) if dn else float("nan"),
                                   first=float(Eh[1]) if len(Eh) > 1 else float("nan"),
                                   S_first=float(np.round(2 * Shs[1]) / 2) if len(Eh) > 1 else float("nan"))
    out["states"] = sols
    return out


def spin_gap_report(g):
    """Text lines of mfh_spin_gap."""
    if g is None:
        return ["singlet-triplet gap: no state with S_z + 1 in the scan (add it to 'S_z to try')"]
    sl, sh = g["two_sz"][0] / 2, g["two_sz"][1] / 2
    name = "singlet-triplet" if g["two_sz"][0] == 0 else f"S = {sl:g} -> {sh:g}"
    L = [f"{name} gap from the mean field (lowest S_z = {sl:g} and {sh:g} states):",
         f"  E(S_z = {sh:g}) - E(S_z = {sl:g}) = {1e3 * g['raw']:+.3f} meV (raw mean-field energies)",
         f"  <S^2> of the determinants: {g['s2_LS']:.3f} (S_z = {sl:g}; pure S = {sl:g}: {sl * (sl + 1):g}), "
         f"{g['s2_HS']:.3f} (S_z = {sh:g}; pure: {sh * (sh + 1):g})"]
    if g["closed_shell"]:
        L.append(f"  the lowest S_z = {sl:g} solution has no moments (closed shell): the gap is the raw one, "
                 f"{1e3 * g['raw']:+.3f} meV (if an open-shell S_z = {sl:g} state exists the mean field did not find it)")
        return L
    L.append(f"  spin-projected (Yamaguchi, two spin centres): {1e3 * g['yamaguchi']:+.3f} meV"
             + ("   <- use this one (two radicals)" if g["n_centres"] in (0, 2) else ""))
    h = g["heis"]
    if g["n_centres"]:
        L.append(f"  spin centres (radicals) found: {g['n_centres']}" +
                 (f"; {g['added']} extra mean-field runs with the moments of some centres flipped" if g["added"] else ""))
    if np.isfinite(g.get("s2_ideal", np.nan)) and abs(g["s2_LS"] - g["s2_ideal"]) > 0.15 * max(g["s2_ideal"], 1):
        L.append(f"  <S^2> = {g['s2_LS']:.3f} instead of {g['s2_ideal']:g} for {g['n_centres']} separated spins: the "
                 "moments overlap (delocalised), so the Ising mapping below overestimates J; for two radicals "
                 "the Yamaguchi value accounts for it")
    if h is not None:
        J = ", ".join(f"{1e3 * j:+.2f}" for j in h["J"])
        L.append(f"  Heisenberg mapping of {h['n_states']} collinear states, nearest centres along the chain"
                 + (" (one J for all bonds)" if h["uniform"] else "") + f": J = {J} meV"
                 + f" (Ising fit rms {1e3 * h['rms']:.3f} meV)")
        L.append(f"    exact Heisenberg: ground S = {h['S0']:g}; gap to S = {h['S0'] + 1:g}: {1e3 * h['gap_up']:.3f} meV"
                 + (f"; first excitation {1e3 * h['first']:.3f} meV (S = {h['S_first']:g})"
                    if abs(h['first'] - h['gap_up']) > 1e-9 else "")
                 + ("   <- use this one (more than two radicals)" if g["n_centres"] > 2 else ""))
    if g["n_centres"] > 2:
        L.append("  (with more than two radicals the two-state projection is not valid: it assumes the "
                 "energy depends only on <S^2>)")
    L.append("  sign: positive = the lower-spin state is the ground state (antiferromagnetic coupling)")
    if g["raw"] < 0:
        L.append("  NOTE: the higher-spin state is lower (ferromagnetic coupling, as when the radicals sit on the "
                 "same sublattice, Lieb). The sign is reliable; the size is not when the radical orbitals share "
                 "atoms (non-disjoint): the mean field then overestimates it several times (m-xylylene, U = 3 eV: "
                 "-1.08 eV projected, -0.27 eV exact). Use the ED for the size")
    return L


def mfh_states(results, gap=None, tol=1e-6):
    """The states of mfh_scan plus the extra ones found by mfh_spin_gap (moments of some
    centres flipped), sorted by energy, without duplicates (same S_z and energy)."""
    allst = list(results)
    if gap is not None:
        allst += [o for o in gap["states"] if "pattern" in o["label"] and o["sol"]["converged"]]
    out = []
    for o in sorted(allst, key=lambda o: o["sol"]["E"]):
        if any(p["two_sz"] == o["two_sz"] and abs(p["sol"]["E"] - o["sol"]["E"]) < tol for p in out):
            continue
        o = dict(o)
        o.setdefault("mmax", float(np.abs(o["sol"]["n"][0] - o["sol"]["n"][1]).max()))
        out.append(o)
    return out
