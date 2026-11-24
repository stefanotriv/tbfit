"""
tbcore: tight binding of 1D-periodic pi-conjugated structures + effective-model fitting.

Structure input
  .hex / .txt   hexagon map ('O' = hexagon, "O.O" = neighbours, rows shifted by one char;
                the width of the map is the period).  Optional line "period <n_chars>".
                "remove <x> <y>" deletes the atom at (x, y) in A (sp3 CH2 carbon, vacancy);
                the coordinates are those shown in the GUI when clicking an atom.
  .cdxml        ChemDraw XML (in ChemDraw: File > Save As > CDXML).
  .mol / .sdf   MDL molfile V2000.
  .xyz          element x y (z); the comment line may contain  period=Tx,Ty  or
                Lattice="Tx Ty 0 ..." (extended xyz, first vector = period).
  sp3 carbons (no pz orbital) are recognised and left out: a carbon with 4 neighbours
  counting its hydrogens (explicit H atoms, a ChemDraw 'CH2' label, or H closer than 1.65 A
  in an .xyz file), or, in a Kekule drawing, a carbon with no double bond and no radical.
  For .cdxml / .mol / .xyz without a period: mark the periodic bonds with star atoms
  ('*', 'R', 'X' or ChemDraw generic nickname / attachment point). Each star sits where
  the atom of the NEXT (or previous) unit cell would be; the translation vector is
  deduced from them. Hydrogens are ignored, drawings are rescaled to C-C = 1.42 A.

Model
  H(k) = sum_R H_R e^{ikR},   H_{-R} = H_R^T,   k in [-pi, pi] (units of 1/period)
  hoppings by distance shells: 1st a, 2nd sqrt(3)a, 3rd 2a  (a = 1.42 A).

Effective model file (.eff), one statement per line, '#' starts a comment:
  site  <name>  atom <i> [<j> ...]         effective site on atom i (or the symmetric
                                          combination of atoms i, j, ...; indices as in the GUI)
  site  <name>  near <x> <y> [<x> <y> ...] same, with the atoms closest to the given points (A)
      ... weights <w1> <w2> ...           optional signed weights of the atoms (default all 1),
                                          e.g. an antisymmetric edge state: weights 1 -1
  site  <name>                     site without atom (no projection, fit only)
  onsite <site> <param>            <site|H|site>              = param
  hop   <s1> <s2> <R> <param>      <s1, cell 0|H|s2, cell R>  = param   (param may be '-name')
  param <name> [value] [fix]       optional initial value; 'fix' keeps it fixed in the fit
  bands <lo> <hi>                  TB bands to fit, relative to the highest occupied band
                                   (0 = HOMO band, 1 = LUMO band). Default: centred on E_F.
  bands auto                       at every k take the TB states with the largest weight on
                                   the model sites (for bands that cross other bands)
"""

import re
import xml.etree.ElementTree as ET
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial import cKDTree

A_CC = 1.42
SHELLS = np.array([1.0, np.sqrt(3.0), 2.0]) * A_CC
T_DEFAULT = (-2.7, -0.20, -0.18)

# pz on-site energies (eV) and pi electrons per atom. Hueckel (Streitwieser) values
# h * beta with beta = -2.7 eV; edit if you use heteroatoms.
ONSITE = {"C": 0.0, "N": -1.35, "B": 2.7, "O": -2.7, "S": -1.35}
ELECTRONS = {"C": 1, "N": 1, "B": 0, "O": 2, "S": 2}
STAR_LABELS = {"*", "R", "X", "R#", "A", "Q", "STAR"}
Z2SYM = {0: "*", 1: "H", 5: "B", 6: "C", 7: "N", 8: "O", 9: "F", 16: "S",
         17: "Cl", 35: "Br", 53: "I"}


# ===========================================================================
# structure
# ===========================================================================
class Structure:
    """Atoms of one unit cell (pz sites) and the translation vector T (A)."""

    def __init__(self, xy, elements, T, name=""):
        self.xy = np.asarray(xy, float)
        self.el = list(elements)
        self.T = np.asarray(T, float)
        self.name = name

    @property
    def N(self):
        return len(self.xy)

    @property
    def n_electrons(self):
        return int(sum(ELECTRONS.get(e, 1) for e in self.el))

    def fold(self, cut_in_gap=False):
        """Put every atom in one cell along T and remove duplicates.

        cut_in_gap: place the cell boundary in the widest empty gap along T
        (so rings are not cut in half); otherwise the cell is [0, |T|).
        """
        L = np.linalg.norm(self.T)
        s = (self.xy @ self.T) / L ** 2                 # position along T in units of T
        cut = 0.0
        if cut_in_gap and len(s) > 1:
            f = np.sort(s % 1.0)
            gaps = np.diff(np.r_[f, f[0] + 1])
            g = np.argmax(gaps)
            cut = (f[g] + gaps[g] / 2) % 1.0
        xy = self.xy - np.floor(s - cut + 1e-6)[:, None] * self.T - cut * self.T
        tree = cKDTree(xy)
        keep, dup = [], set()
        for i in range(len(xy)):
            if i in dup:
                continue
            keep.append(i)
            for sgn in (0, 1, -1):
                for j in tree.query_ball_point(xy[i] + sgn * self.T, 0.3):
                    if j > i:
                        dup.add(j)
        self.xy = xy[keep]
        self.el = [self.el[i] for i in keep]
        return self


def from_hexmap(text, name="hexmap", finite=False):
    """Structure from a text map of hexagons ('O'), period along x = width of the map.

    Default (zigzag along x): pointy-top hexagons, neighbours in a row are 'O.O' and every
    row is shifted by one character; the period is a multiple of sqrt(3) a (zigzag
    direction). With a line 'armchair': flat-top hexagons, each character is a column of
    the ribbon and each text row half a hexagon down; a hexagon has neighbours two rows above
    and below in the same column and one row above and below in the next columns, so the
    'O's form a checkerboard (row + column even). The period is a multiple of 3a/2 (armchair
    direction), e.g. the 7-AGNR is the two columns 'O.' '.O' 'O.' '.O' 'O.'.
    Other lines: 'period W' (width in characters), 'remove x y' (sp3 carbon / vacancy)."""
    lines = [l.rstrip() for l in text.splitlines()]
    period = None
    armchair = False
    rows, removed = [], []
    for l in lines:
        w = l.split("#")[0].split()
        if w and w[0].lower() in ("armchair", "flat"):
            armchair = True
        elif w and w[0].lower() == "period":
            period = int(w[1])
        elif w and w[0].lower() == "remove":          # sp3 carbon / vacancy: no pz orbital
            removed.append((float(w[1]), float(w[2])))
        elif l.strip() and not l.strip().startswith("#"):
            rows.append(l.split("#")[0].rstrip())
    if not rows:
        raise ValueError("empty hexagon map")
    W = period or max(len(r) for r in rows)
    if W % 2 and not finite:
        raise ValueError("the period (map width) must be an even number of characters")
    if armchair:
        # flat-top hexagons: x in units of a/2 (3 per column), y in units of sqrt(3) a / 2
        cen = [(3 * c, -r) for r, row in enumerate(rows) for c, ch in enumerate(row) if ch in "Oo"]
        if len({(x // 3 - y) % 2 for x, y in cen}) > 1:
            raise ValueError("armchair map: the hexagons must form a checkerboard (row + column even, or "
                             "all odd): a hexagon's neighbours are two rows above / below in its column "
                             "and one row above / below in the next columns")
        corners = [(2, 0), (-2, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]
        Wx, scale, L = 3 * W, [A_CC / 2, np.sqrt(3) / 2 * A_CC], W * 1.5 * A_CC
    else:
        cen = [(c, -3 * r) for r, row in enumerate(rows) for c, ch in enumerate(row) if ch in "Oo"]
        if len({(c + (-y // 3)) % 2 for c, y in cen}) > 1:
            raise ValueError("hexagon rows are misaligned: neighbours are 'O.O' and every row "
                             "is shifted by one character relative to the row above")
        corners = [(0, 2), (0, -2), (1, 1), (1, -1), (-1, 1), (-1, -1)]
        Wx, scale, L = W, [np.sqrt(3) / 2 * A_CC, A_CC / 2], W * np.sqrt(3) / 2 * A_CC
    atoms = sorted({((c + dx) if finite else (c + dx) % Wx, y + dy) for c, y in cen for dx, dy in corners},
                   key=lambda p: (-p[1], p[0]))
    xy = np.array(atoms, float) * scale
    keep = np.ones(len(xy), bool)
    for p in removed:
        d = np.linalg.norm(xy - p, axis=1)
        d = np.minimum.reduce([d] + [np.linalg.norm(xy + c * np.array([L, 0]) - p, axis=1) for c in (-1, 1)])
        i = int(np.argmin(d))
        if d[i] > 0.3:
            raise ValueError(f"'remove {p[0]} {p[1]}': no atom there (nearest is {d[i]:.2f} A away)")
        keep[i] = False
    xy = xy[keep]
    return Structure(xy, ["C"] * len(xy), [L, 0.0], name)


def _period_from_stars(xy, stars, anchors=None, tol=0.35):
    """Translation vector from star atoms: each star is an image (+T or -T) of a real atom.

    anchors: indices (in xy) of the atoms bonded to star atoms. In a polymer drawing the
    atom a star stands for is itself bonded to a star, so those are tried first.
    """
    if not stars:
        return None
    for pool in ([anchors] if anchors else []) + [list(range(len(xy)))]:
        try:
            return _vote(xy, stars, pool, tol)
        except ValueError as e:
            err = e
    raise err


def _vote(xy, stars, pool, tol):
    tree = cKDTree(xy)
    cands = [p - xy[q] for p in stars for q in pool]
    best, best_score = None, -1
    for v in cands:
        if np.linalg.norm(v) < 2.0:
            continue
        # a real period never puts an atom on top of another atom
        if np.min(tree.query(xy + v)[0]) < 0.5:
            continue
        score = 0
        for p in stars:
            d1, _ = tree.query(p - v)
            d2, _ = tree.query(p + v)
            score += (min(d1, d2) < tol)
        # prefer more stars explained, then the shortest vector
        if score > best_score or (score == best_score and np.linalg.norm(v) < np.linalg.norm(best) - 1e-6):
            best, best_score = v, score
    if best_score < len(stars):
        raise ValueError(f"could not find one translation vector matching all {len(stars)} "
                         f"star atoms (matched {best_score})")
    # refine: average over stars
    vs = []
    for p in stars:
        for sgn in (1, -1):
            d, j = tree.query(p - sgn * best)
            if d < tol:
                vs.append(sgn * (p - xy[j]))
    T = np.mean(vs, axis=0)
    return T if T[0] >= -1e-9 else -T


def period_from_repeats(xy, tol=0.3, min_len=2.0, return_all=False):
    """Period of a drawing that shows two or more repeat units, found automatically.

    The period is the shortest shift v that maps EVERY atom onto an atom, except the atoms
    whose image would fall beyond the end of the drawing. Requiring an exact match (not
    "most atoms") rejects plain lattice shifts that only fail at irregular edges.
    Both axes of the drawing are searched: a short, wide drawing (e.g. two columns of three
    stacked hexagons) can repeat along both, and then the direction is ambiguous. The
    candidates are ordered with the one closest to the horizontal of the drawing first
    (ribbons are usually drawn horizontally).
    Returns v (or the list of candidates with return_all=True), None / [] if not periodic.
    """
    N = len(xy)
    if N < 4:
        return [] if return_all else None
    c = xy - xy.mean(axis=0)
    axes = np.linalg.svd(c, full_matrices=False)[2]
    tree = cKDTree(xy)
    found = []
    for u in axes[:2]:
        s = xy @ u
        span = s.max() - s.min()
        refs = np.argsort(s)[:3]
        cands = {}
        for r in refs:
            for j in range(N):
                v = xy[j] - xy[r]
                if v @ u > min_len and v @ u <= 0.6 * span:
                    cands[tuple(np.round(v / 0.05).astype(int))] = v
        for v in sorted(cands.values(), key=np.linalg.norm):
            L = v @ u
            inside = s + L <= s.max() - tol
            if inside.sum() < 0.3 * N:
                continue
            d, _ = tree.query(xy[inside] + v)
            if np.all(d < tol):
                if not any(abs(v[0] * w[1] - v[1] * w[0]) < 0.1 * np.linalg.norm(v) * np.linalg.norm(w) for w in found):
                    found.append(v)
                break
    found = [v if v[0] > 1e-9 or (abs(v[0]) <= 1e-9 and v[1] > 0) else -v for v in found]
    found.sort(key=lambda v: (round(abs(v[1]) / np.linalg.norm(v), 3), np.linalg.norm(v)))
    if return_all:
        return found
    return found[0] if found else None


def reduce_period(xy, T, tol=0.3, nmax=12):
    """Shortest period T/n (n <= nmax) that still maps the periodic structure onto itself."""
    L2 = T @ T
    fold = lambda p: p - np.floor((p @ T) / L2 + 1e-9)[:, None] * T
    f = fold(xy)
    tree = cKDTree(np.concatenate([f - T, f, f + T]))
    for n in range(nmax, 1, -1):
        d, _ = tree.query(fold(xy + T / n))
        if np.all(d < tol):
            return T / n
    return T


def _neighbour_count(n_atoms, bonds, nH=None, xyz=None, T=None, star_idx=(), cutoff=1.65):
    """Number of bonded neighbours of every atom, hydrogens included.

    From the drawn bonds plus hydrogen counts stored on the atoms (nH, e.g. a ChemDraw
    'CH2' label); without a bond list, from the 3D distances (< cutoff A), including the
    periodic images when the period T is known."""
    deg = np.zeros(n_atoms, int)
    if bonds:
        for i, j in bonds:
            deg[i] += 1
            deg[j] += 1
    elif xyz is not None:
        P = np.asarray(xyz, float)
        real = np.array([i for i in range(n_atoms) if i not in star_idx], int)
        tree = cKDTree(P[real])
        shifts = [np.zeros(3)]
        if T is not None:
            t3 = np.r_[np.asarray(T, float)[:2], 0.0]
            shifts += [t3, -t3]
        for k, i in enumerate(real):
            for sh in shifts:
                for j in tree.query_ball_point(P[i] + sh, cutoff):
                    if not (np.allclose(sh, 0) and j == k):
                        deg[i] += 1
    if nH is not None:
        deg = deg + np.asarray(nH, int)
    return deg


def _finish(xy, el, bonds, star_idx, T, name, rescale=True, orders=None, radicals=(), finite=False, prefer=None,
            nH=None, xyz=None):
    """Common part of the molecular loaders.

    Drops H and halogens, rescales the drawing to C-C = 1.42 A, removes sp3 carbons, and finds
    the period: from two or more drawn repeat units, else from star atoms / dashed
    continuation bonds. A carbon is sp3 (no pz orbital) if it has 4 neighbours counting its
    hydrogens (explicit H atoms, a 'CH2' label, or H within 1.65 A in an xyz file), or, in a
    Kekule drawing, if it has neither a double bond nor a radical mark.
    """
    xy = np.asarray(xy, float)
    deg = _neighbour_count(len(el), bonds, nH, xyz, T, star_idx)
    real = [i for i in range(len(el)) if i not in star_idx and el[i] != "H"]
    if rescale:
        bl = [np.linalg.norm(xy[i] - xy[j]) for i, j in bonds
              if i in real and j in real and el[i] in ONSITE and el[j] in ONSITE]
        if not bl:  # no bond list: use nearest-neighbour distances
            pz = xy[[i for i in real if el[i] in ONSITE]]
            d, _ = cKDTree(pz).query(pz, k=2)
            bl = d[:, 1]
        sc = A_CC / np.median(bl)
        if abs(sc - 1) > 0.1:
            xy = xy * sc
            if T is not None:
                T = np.asarray(T) * sc
    pz = [i for i in real if el[i] in ONSITE]

    # sp3 carbons: in a Kekule drawing every sp2 carbon has a double (or aromatic) bond or a
    # radical mark. A carbon with neither is sp3 (CH2, CH, CH3) and has no pz orbital.
    sp3 = [i for i in pz if el[i] == "C" and deg[i] >= 4]          # saturated (hydrogenated) carbons
    pz = [i for i in pz if i not in sp3]
    if orders is not None and len(orders) == len(bonds):
        pi = set()
        for (i, j), o in zip(bonds, orders):
            if o in (2, 3, 4, 1.5):
                pi.update((i, j))
        good = [i for i in pz if i in pi or i in radicals]
        if len(good) >= 0.5 * len(pz) and len(pi) > 0:
            extra = [i for i in pz if i not in pi and i not in radicals and el[i] == "C"]
            sp3 += extra
            pz = [i for i in pz if i not in extra]
    xy = xy - xy[pz].mean(axis=0)                     # centre the drawing at the origin
    skipped = sorted({el[i] for i in real if el[i] not in ONSITE})
    how = "given" if T is not None else None
    P = xy[pz]
    keep = np.arange(len(pz))
    if finite:
        T, star_idx = None, set()
    alternatives = []
    if T is None and not finite:
        cand = period_from_repeats(P, return_all=True)
        if prefer is not None and cand:
            pr = np.asarray(prefer, float) / np.linalg.norm(prefer)
            cand.sort(key=lambda v: -abs(v @ pr) / np.linalg.norm(v))
        T = cand[0] if cand else None
        alternatives = cand[1:]
        if T is not None:
            how = "repeat units"
            # keep one period from the middle of the drawing
            u = T / np.linalg.norm(T)
            s = P @ u
            c = 0.5 * (s.min() + s.max())
            L = np.linalg.norm(T)
            keep = np.where((s >= c - L / 2) & (s < c + L / 2))[0]
    if T is None and star_idx:
        pos = {i: k for k, i in enumerate(pz)}
        anchors = sorted({pos[a] for i, j in bonds for s_, a in ((i, j), (j, i))
                          if s_ in star_idx and a in pos})
        try:
            T = _period_from_stars(P, [xy[i] for i in star_idx], anchors)
            how = "star atoms / dashed bonds"
        except ValueError:
            T = None
    if T is not None and how != "given":
        T0 = T
        T = reduce_period(P[keep], T) if how != "repeat units" else T
        if not np.allclose(T, T0):
            how += f" (reduced to 1/{int(round(np.linalg.norm(T0) / np.linalg.norm(T)))})"
    st = Structure(P[keep], [el[pz[k]] for k in keep], T if T is not None else [0, 0], name)
    st.skipped = skipped
    st.periodic = T is not None
    st.period_source = how
    st.period_alternatives = [np.asarray(v, float) for v in alternatives]
    st.n_sp3 = len(sp3)
    return st


def _is_dashed(display):
    d = (display or "").lower()
    return "dash" in d or "dot" in d or "hash" in d


def read_cdxml(path, finite=False, prefer=None):
    root = ET.parse(path).getroot()
    xy, el, ids, star_idx, bonds, orders, radicals, nH = [], [], {}, set(), [], [], set(), []
    for n in root.iter("n"):
        if "p" not in n.attrib:
            continue
        ntype = n.attrib.get("NodeType", "")
        if ntype in ("Fragment", "Nickname", "Unspecified") and n.find("fragment") is not None:
            continue
        x, y = map(float, n.attrib["p"].split()[:2])
        label = "".join(s.text or "" for s in n.iter("s")).strip()
        z = int(n.attrib.get("Element", "6"))
        sym = Z2SYM.get(z, "X?")
        is_star = (ntype == "GenericNickname"
                   or n.attrib.get("GenericNickname", "") in STAR_LABELS
                   or label in STAR_LABELS or z == 0)
        ids[n.attrib.get("id")] = len(xy)
        if is_star:
            star_idx.add(len(xy))
        if n.attrib.get("Radical"):
            radicals.add(len(xy))
        h = n.attrib.get("NumHydrogens")
        mh = re.fullmatch(r"C(H(\d*))?", label)
        if h is None and mh and mh.group(1):
            h = mh.group(2) or "1"
        nH.append(int(h) if h else 0)
        xy.append((x, -y))                          # ChemDraw y points down
        el.append("*" if is_star else sym)
    dashed = {}
    for b in root.iter("b"):
        i, j = ids.get(b.attrib.get("B")), ids.get(b.attrib.get("E"))
        if i is None or j is None:
            continue
        bonds.append((i, j))
        orders.append(float(b.attrib.get("Order", "1")))
        dsh = _is_dashed(b.attrib.get("Display"))
        for a in (i, j):
            dashed.setdefault(a, []).append(dsh)
    # an atom attached only through dashed bonds marks the continuation of the chain
    for a, d in dashed.items():
        if all(d) and len(d) == 1:
            star_idx.add(a)
            el[a] = "*"
    return _finish(xy, el, bonds, star_idx, None, path, orders=orders, radicals=radicals, finite=finite, nH=nH,
                   prefer=prefer)


def read_mol(path, finite=False, prefer=None):
    lines = open(path).read().splitlines()
    na, nb = int(lines[3][0:3]), int(lines[3][3:6])
    xy, el, star_idx, bonds, orders, radicals = [], [], set(), [], [], set()
    for i in range(na):
        l = lines[4 + i]
        x, y = float(l[0:10]), float(l[10:20])
        sym = l[31:34].strip()
        if sym in STAR_LABELS:
            star_idx.add(i)
            sym = "*"
        xy.append((x, y))
        el.append(sym)
    for i in range(nb):
        l = lines[4 + na + i]
        bonds.append((int(l[0:3]) - 1, int(l[3:6]) - 1))
        orders.append(float(l[6:9]))
    for l in lines[4 + na + nb:]:
        if l.startswith("M  RAD"):
            w = l.split()[3:]
            radicals.update(int(w[k]) - 1 for k in range(0, len(w), 2))
    return _finish(xy, el, bonds, star_idx, None, path, orders=orders, radicals=radicals, finite=finite,
                   prefer=prefer)


def read_xyz(path, finite=False):
    lines = open(path).read().splitlines()
    n = int(lines[0].split()[0])
    comment = lines[1]
    T = None
    m = re.search(r'period\s*=\s*([-\d.eE]+)\s*[, ]\s*([-\d.eE]+)', comment)
    if m:
        T = [float(m.group(1)), float(m.group(2))]
    m = re.search(r'Lattice\s*=\s*"([^"]+)"', comment)
    if m and T is None:
        v = [float(x) for x in m.group(1).split()]
        T = v[:2]
    xy, el, star_idx, z = [], [], set(), []
    for i in range(n):
        p = lines[2 + i].split()
        sym = p[0]
        if sym in STAR_LABELS:
            star_idx.add(i)
            sym = "*"
        xy.append((float(p[1]), float(p[2])))
        z.append(float(p[3]) if len(p) > 3 else 0.0)
        el.append(sym)
    xyz = np.c_[np.array(xy), np.array(z)]
    return _finish(xy, el, [], star_idx, None if finite else T, path, rescale=T is None, finite=finite,
                   xyz=xyz)


def load_structure(path, finite=False, prefer=None):
    """Read a structure. finite=True: a finite molecule (no period search, no folding).
    prefer: a direction (vector) for the period when a drawing repeats along more than one
    (see period_from_repeats; the other candidates are in st.period_alternatives)."""
    p = path.lower()
    if p.endswith(".cdxml"):
        st = read_cdxml(path, finite, prefer)
    elif p.endswith((".mol", ".sdf")):
        st = read_mol(path, finite, prefer)
    elif p.endswith(".xyz"):
        st = read_xyz(path, finite)
    elif p.endswith(".cdx"):
        raise ValueError("binary .cdx is not supported: in ChemDraw use File > Save As > .cdxml")
    else:
        st = from_hexmap(open(path).read(), path, finite)
        st.skipped, st.periodic = [], not finite
    if finite:
        st.T, st.periodic = np.zeros(2), False
    st.name = path.replace("\\", "/").split("/")[-1]
    if getattr(st, "periodic", True):
        st.fold(cut_in_gap=not p.endswith((".hex", ".txt")) and "." in p)
    return st


def write_cdxml(st, path):
    """Export a structure to ChemDraw (with star atoms on the periodic bonds)."""
    H = hoppings(st, t=(1.0,))
    s = 14.4 / A_CC                                   # ChemDraw default bond length (pt)
    pts = lambda p: f"{p[0] * s + 200:.2f} {-p[1] * s + 200:.2f}"
    out = ['<?xml version="1.0" encoding="UTF-8" ?>',
           '<!DOCTYPE CDXML SYSTEM "http://www.cambridgesoft.com/xml/cdxml.dtd" >',
           '<CDXML BondLength="14.4"><page id="1"><fragment id="2">']
    nid = 10
    for i, p in enumerate(st.xy):
        out.append(f'<n id="{nid + i}" p="{pts(p)}" Element="{ {"C": 6, "N": 7, "B": 5, "O": 8, "S": 16}[st.el[i]] }"/>')
    bid = nid + st.N
    for i, j in zip(*np.nonzero(np.triu(H.get(0, 0 * H[0])))):
        out.append(f'<b id="{bid}" B="{nid + i}" E="{nid + j}"/>')
        bid += 1
    star = bid + 1000
    for R in [r for r in H if r > 0]:
        for i, j in zip(*np.nonzero(H[R])):
            for a, p in ((i, st.xy[j] + R * st.T), (j, st.xy[i] - R * st.T)):
                out.append(f'<n id="{star}" p="{pts(p)}" NodeType="GenericNickname" '
                           f'GenericNickname="*"><t><s>*</s></t></n>')
                out.append(f'<b id="{star + 1}" B="{nid + a}" E="{star}"/>')
                star += 2
    out.append("</fragment></page></CDXML>")
    open(path, "w").write("\n".join(out))


# ===========================================================================
# tight binding
# ===========================================================================
def hoppings(st, t=T_DEFAULT, tol=0.15):
    """Dict {R: H_R} for R >= 0 (H_0 includes on-site energies)."""
    N, xy, T = st.N, st.xy, st.T
    shells = SHELLS[:len(t)]
    rmax = shells[-1] + tol
    L = np.linalg.norm(T)
    extent = np.ptp(xy @ (T / L)) if (N and L > 0) else 0
    Rmax = int(np.ceil((extent + rmax) / L)) if L > 0 else 0
    H = {}
    for R in range(0, Rmax + 1):
        D = np.linalg.norm(xy[None, :, :] + R * T - xy[:, None, :], axis=2)
        M = np.zeros((N, N))
        for tn, dn in zip(t, shells):
            M[np.abs(D - dn) < tol] = tn
        if R == 0:
            np.fill_diagonal(M, [ONSITE.get(e, 0.0) for e in st.el])
        if R == 0 or np.any(M):
            H[R] = M
    return H


def Hk(H, k):
    out = H[0].astype(complex)
    for R, M in H.items():
        if R:
            out = out + M * np.exp(1j * k * R) + M.T * np.exp(-1j * k * R)
    return out


def bands(H, nk=201):
    ks = np.linspace(-np.pi, np.pi, nk)
    return ks, np.array([np.linalg.eigvalsh(Hk(H, k)) for k in ks])


def fermi(E, n_electrons):
    """E_F for spin-degenerate filling: mid-gap for an even number of electrons, the middle
    of the half-filled band for an odd number (e.g. after removing one atom)."""
    nocc = n_electrons // 2
    if n_electrons % 2 and nocc < E.shape[1]:
        return float(np.median(E[:, nocc])), nocc + 1
    lo = E[:, nocc - 1].max() if nocc > 0 else E.min()
    hi = E[:, nocc].min() if nocc < E.shape[1] else E.max()
    return 0.5 * (lo + hi), nocc


# ===========================================================================
# effective model
# ===========================================================================
class EffModel:
    def __init__(self):
        self.sites, self.site_atom, self.site_w = [], [], []
        self.terms = []                 # (i, j, R, pname, sign); i == j and R == 0: on-site
        self.pinit, self.fixed = {}, set()
        self.window = None

    @property
    def params(self):
        names = []
        for *_, p, _ in self.terms:
            if p not in names:
                names.append(p)
        return names

    def H(self, k, pv):
        n = len(self.sites)
        h = np.zeros((n, n), complex)
        for i, j, R, p, s in self.terms:
            v = s * pv[p]
            if i == j and R == 0:
                h[i, i] += v
            else:
                ph = np.exp(1j * k * R)
                h[i, j] += v * ph
                h[j, i] += v * np.conj(ph)
        return h

    def text(self, pv=None):
        """The model as a .eff text, optionally with fitted values."""
        pv = pv or {}
        out = []
        if self.window == "auto":
            out.append("bands auto")
        elif self.window:
            out.append(f"bands {self.window[0]} {self.window[1]}")
        for n, a, wt in zip(self.sites, self.site_atom, self.site_w):
            out.append(f"site {n}" + (" atom " + " ".join(map(str, a)) if a else "") +
                       (" weights " + " ".join(f"{x:g}" for x in wt) if wt else ""))
        for i, j, R, p, s in self.terms:
            sp = ("-" if s < 0 else "") + p
            if i == j and R == 0:
                out.append(f"onsite {self.sites[i]} {sp}")
            else:
                out.append(f"hop {self.sites[i]} {self.sites[j]} {R} {sp}")
        for p in self.params:
            v = pv.get(p, self.pinit.get(p))
            out.append(f"param {p}" + (f" {v:.6f}" if v is not None else "") +
                       (" fix" if p in self.fixed else ""))
        return "\n".join(out)


def parse_eff(text, st=None):
    m = EffModel()
    idx = {}
    for ln, raw in enumerate(text.splitlines(), 1):
        line = raw.split("#")[0].strip()
        if not line:
            continue
        w = line.split()
        key = w[0].lower()
        try:
            if key == "site":
                wts = None
                if "weights" in w:
                    k_ = w.index("weights")
                    wts = [float(x) for x in w[k_ + 1:]]
                    w = w[:k_]
                name, atoms = w[1], None
                if len(w) >= 4 and w[2] == "atom":
                    atoms = [int(x) for x in w[3:]]
                elif len(w) >= 5 and w[2] == "near":
                    if st is None:
                        raise ValueError("'near' needs a loaded structure")
                    c = np.array(w[3:], float).reshape(-1, 2)
                    atoms = [int(np.argmin(np.linalg.norm(st.xy - p, axis=1))) for p in c]
                if st is not None and atoms is not None and not all(0 <= a < st.N for a in atoms):
                    raise ValueError(f"atom index out of range (0..{st.N - 1})")
                if wts is not None and (atoms is None or len(wts) != len(atoms)):
                    raise ValueError("one weight per atom is needed")
                idx[name] = len(m.sites)
                m.sites.append(name)
                m.site_atom.append(atoms)
                m.site_w.append(wts)
            elif key == "onsite":
                p = w[2]
                s = -1.0 if p.startswith("-") else 1.0
                m.terms.append((idx[w[1]], idx[w[1]], 0, p.lstrip("+-"), s))
            elif key == "hop":
                i, j, R, p = idx[w[1]], idx[w[2]], int(w[3]), w[4]
                if R < 0:
                    i, j, R = j, i, -R
                if i == j and R == 0:
                    raise ValueError("use 'onsite' for on-site terms")
                s = -1.0 if p.startswith("-") else 1.0
                m.terms.append((i, j, R, p.lstrip("+-"), s))
            elif key == "param":
                if len(w) > 2 and w[2].lower() != "fix":
                    m.pinit[w[1]] = float(w[2])
                if "fix" in [x.lower() for x in w[2:]]:
                    m.fixed.add(w[1])
            elif key == "bands":
                m.window = "auto" if w[1].lower() == "auto" else (int(w[1]), int(w[2]))
            else:
                raise ValueError(f"unknown keyword '{w[0]}'")
        except KeyError as e:
            raise ValueError(f"line {ln}: unknown site {e}") from None
        except (IndexError, ValueError) as e:
            raise ValueError(f"line {ln}: {e}  ->  {raw.strip()}") from None
    if not m.sites:
        raise ValueError("no sites defined")
    n = len(m.sites)
    if m.window is None:                      # n bands centred on E_F
        lo = -(n // 2 - 1) if n % 2 == 0 else -((n - 1) // 2)
        m.window = (lo, lo + n - 1)
    if m.window != "auto" and m.window[1] - m.window[0] + 1 != n:
        raise ValueError(f"'bands {m.window[0]} {m.window[1]}' selects "
                         f"{m.window[1] - m.window[0] + 1} bands but the model has {n} sites")
    return m


def trial_orbitals(model, N):
    """Matrix N x n_sites of the site orbitals (None if a site has no atom)."""
    if any(a is None for a in model.site_atom):
        return None
    G = np.zeros((N, len(model.sites)))
    for n, atoms in enumerate(model.site_atom):
        wt = np.array(model.site_w[n] or [1.0] * len(atoms), float)
        G[atoms, n] = wt / np.linalg.norm(wt)
    return G


def select(model, E, V, nocc, G):
    """Indices of the TB eigenstates fitted by the model at one k."""
    n = len(model.sites)
    if model.window == "auto":
        if G is None:
            raise ValueError("'bands auto' needs every site on atoms")
        weight = np.sum(np.abs(G.T @ V) ** 2, axis=0)
        return np.sort(np.argsort(weight)[::-1][:n])
    lo, hi = model.window
    return np.arange(nocc - 1 + lo, nocc + hi)


def project(H, model, nocc, nk=64, Rmax=4):
    """Exact effective Hamiltonian on the model sites (projection + Loewdin).

    Returns ({R: h_R}, min overlap eigenvalue). Needs every site on atoms.
    """
    N = H[0].shape[0]
    G = trial_orbitals(model, N)
    if G is None:
        return None, 0.0
    ks = 2 * np.pi * np.arange(nk) / nk
    Hs, smin = [], 1.0
    for k in ks:
        E, V = np.linalg.eigh(Hk(H, k))
        sel = select(model, E, V, nocc, G)
        A = V[:, sel].conj().T @ G                      # <psi_m | site n>
        S = A.conj().T @ A
        w, U = np.linalg.eigh(S)
        smin = min(smin, w.min())
        Sm = U @ np.diag(np.clip(w, 1e-12, None) ** -0.5) @ U.conj().T
        Hs.append(Sm @ A.conj().T @ np.diag(E[sel]) @ A @ Sm)
    Hs = np.array(Hs)
    hR = {R: np.real(np.mean(Hs * np.exp(-1j * ks * R)[:, None, None], axis=0))
          for R in range(0, Rmax + 1)}
    return hR, smin


def fit(H, st, model, nk=81, use_projection=True):
    """Least-squares fit of the model bands to the TB bands. Returns a result dict."""
    ks = np.linspace(-np.pi, np.pi, nk)
    G = trial_orbitals(model, st.N)
    eig = [np.linalg.eigh(Hk(H, k)) for k in ks]
    E_tb = np.array([e for e, _ in eig])
    EF, nocc = fermi(E_tb, st.n_electrons)
    sels = [select(model, e, v, nocc, G) for e, v in eig]
    if min(x.min() for x in sels) < 0 or max(x.max() for x in sels) >= st.N:
        raise ValueError("band window outside the spectrum")
    E_ref = np.array([e[x] for (e, _), x in zip(eig, sels)])

    # isolation of the fitted bands (index window only)
    if model.window == "auto":
        gap_below = gap_above = np.nan
    else:
        a, b = sels[0][0], sels[0][-1]
        below = E_tb[:, a - 1].max() if a > 0 else -np.inf
        above = E_tb[:, b + 1].min() if b + 1 < st.N else np.inf
        gap_below, gap_above = E_ref[:, 0].min() - below, above - E_ref[:, -1].max()

    # initial values: file > projection > defaults
    hR, smin = project(H, model, nocc) if use_projection else (None, 0.0)
    init, source = {}, {}
    for p in model.params:
        vals = []
        if hR is not None:
            for i, j, R, q, s in model.terms:
                if q == p and R in hR:
                    vals.append(s * hR[R][i, j])
        if p in model.pinit:
            init[p], source[p] = model.pinit[p], "given"
        elif vals:
            init[p], source[p] = float(np.mean(vals)), "projection"
        else:
            onsite = all(i == j and R == 0 for i, j, R, q, s in model.terms if q == p)
            init[p], source[p] = (EF if onsite else 0.1), "default"

    free = [p for p in model.params if p not in model.fixed]

    def unpack(x):
        pv = dict(init)
        pv.update(zip(free, x))
        return pv

    def resid(x):
        pv = unpack(x)
        return np.concatenate([np.linalg.eigvalsh(model.H(k, pv)) - e for k, e in zip(ks, E_ref)])

    x0 = np.array([init[p] for p in free])
    r0 = resid(x0)
    sol = least_squares(resid, x0) if free else None
    pv = unpack(sol.x) if free else dict(init)
    r = resid(sol.x) if free else r0

    # projected couplings that the scheme does not contain (suggestions)
    missing = []
    if hR is not None:
        have = {(i, j, R) for i, j, R, *_ in model.terms} | {(j, i, 0) for i, j, R, *_ in model.terms if R == 0}
        n = len(model.sites)
        for R, h in hR.items():
            for i in range(n):
                for j in range(n):
                    if R == 0 and j < i:
                        continue
                    if (i, j, R) not in have and abs(h[i, j]) > 0.01:
                        missing.append((abs(h[i, j]), model.sites[i], model.sites[j], R, h[i, j]))
        missing.sort(reverse=True)

    E_eff = np.array([np.linalg.eigvalsh(model.H(k, pv)) for k in ks])
    return dict(ks=ks, E_tb=E_tb, E_ref=E_ref, sel=sels, E_eff=E_eff, EF=EF, nocc=nocc, params=pv,
                init=init, source=source, rms=np.sqrt(np.mean(r ** 2)), maxerr=np.abs(r).max(),
                rms0=np.sqrt(np.mean(r0 ** 2)), projected=hR, overlap=smin,
                gaps=(gap_below, gap_above), missing=missing)


# ===========================================================================
# building an effective model from user-chosen sites
# ===========================================================================
def symmetry_matrix(st, sym, op, k):
    """Matrix of a point operation (tbcore.symmetry_ops) acting on the Bloch states at k,
    for an operation that maps k to k (ss = +1: mirror across the axis or glide).
    Atom i goes to atom j of the cell m: (P)_ji = exp(-i k m) in the convention of Hk."""
    u, v, L = sym["u"], sym["v"], sym["L"]
    ss, sp_, a, b = op
    s, p = st.xy @ u, st.xy @ v
    P = np.zeros((st.N, st.N), complex)
    for i in range(st.N):
        s2, p2 = ss * s[i] + a, sp_ * p[i] + b
        c = (s2 - s) / L
        j = np.where((np.abs(p - p2) < 0.1) & (np.abs(c - np.round(c)) * L < 0.1))[0][0]
        P[j, i] = np.exp(-1j * k * np.round(c[j]))
    return P


def band_diagnostics(H, st, model, fitres, nk=601, tol=0.005):
    """Crossings and band-selection problems of a fit (text lines + marks for the plot).

    1. Selection: with an index window, a band of another character can take the place of
       a band of the model (e.g. a flat band crossing it): at those k a selected band has
       almost no weight on the sites while a band outside the window has much more, and the
       fit and the projection are wrong. 'bands auto' selects by weight instead.
    2. Crossings of the fitted TB bands (local minima of the gap < tol eV), classified with
       the mirror / glide symmetry of the structure: protected crossing (the two bands have
       different parity and exchange order), avoided crossing (same parity, the gap is a
       hybridisation), close approach of bands of different parity, or degeneracy at the
       zone boundary (a glide makes the bands stick in pairs at k = pi).
    3. The model at the same k: a model gap where the TB bands cross means the model couples
       states of different symmetry (or the sites break the symmetry).
    The fit itself compares the sorted energies at every k, so the order of the bands does
    not matter for it; what matters is which bands are selected and whether the model has
    the symmetry that lets bands cross."""
    G = trial_orbitals(model, st.N)
    ks = np.linspace(0, np.pi, nk)
    nocc, n = fitres["nocc"], len(model.sites)
    sym = symmetry_ops(st)
    ops = [o for o in sym["ops"] if o[0] == 1]            # operations that keep k
    lines, marks = [], []
    E, sels, weak = [], [], []
    for k in ks:
        e, V = np.linalg.eigh(Hk(H, k))
        sel = select(model, e, V, nocc, G)
        E.append(e[sel])
        sels.append((sel, V[:, sel]))
        if G is not None and model.window != "auto":
            # a selected band with almost no weight on the sites while a band outside the
            # window has much more: a band of another character took its place
            w = np.sum(np.abs(G.T @ V) ** 2, axis=0)
            out = np.setdiff1d(np.arange(len(e)), sel)
            if w[sel].min() < 0.1 * w[sel].max() and w[out].max() > 5 * w[sel].min():
                weak.append(k)
    E = np.array(E)
    if weak:
        lines.append(f"WARNING: for k in [{min(weak) / np.pi:.2f}, {max(weak) / np.pi:.2f}] pi the index window "
                     "contains a band with almost no weight on the sites (a band of another character, e.g. a flat "
                     "band of another symmetry crossing the model bands), while a band outside it has much more. "
                     "Use 'bands auto' (selection by weight) or change the sites / window.")
        marks.append(("window", min(weak), max(weak)))

    def parities(k, sel):
        if not ops:
            return None
        e, V = np.linalg.eigh(Hk(H, k))
        Vs = V[:, sel]
        out = []
        for op in ops:
            lam = np.einsum("ib,ij,jb->b", Vs.conj(), symmetry_matrix(st, sym, op, k), Vs)
            out.append(np.real(lam * np.exp(1j * k * op[2] / sym["L"])))   # glide: remove e^{-ik/2}
        return np.array(out)                                              # (n_ops, n_bands)

    model_gap = lambda k, b: np.diff(np.linalg.eigvalsh(model.H(k, fitres["params"])))[b]
    found = 0
    for b in range(E.shape[1] - 1):
        gap = E[:, b + 1] - E[:, b]
        for i in range(nk):
            left = gap[i - 1] if i > 0 else np.inf
            right = gap[i + 1] if i < nk - 1 else np.inf
            if not (gap[i] < tol and gap[i] <= left and gap[i] <= right):
                continue
            k = ks[i]
            ev = E[i, b] - fitres["EF"]
            where = f"bands {b + 1}/{b + 2} of the fit at k = {k / np.pi:.3f} pi, E - E_F = {ev:+.3f} eV"
            mg = model_gap(k, b)
            if i in (0, nk - 1):
                kname = "0" if i == 0 else "pi"
                glide = any(abs(o[2] % sym["L"] - sym["L"] / 2) < 0.1 for o in ops)
                if gap[i] < 1e-6 and i == nk - 1 and glide:
                    kind = "degenerate pair at k = pi (glide + time reversal: the bands stick together in pairs)"
                elif gap[i] < 1e-6:
                    kind = f"degenerate at k = {kname} (bands of different symmetry meet there)"
                else:
                    kind = f"closest approach at k = {kname} (gap {1e3 * gap[i]:.2f} meV)"
                tag = "zone"
            else:
                j0, j1 = max(i - 4, 0), min(i + 4, nk - 1)
                pl, pr = parities(ks[j0], sels[j0][0]), parities(ks[j1], sels[j1][0])
                if pl is None:
                    kind, tag = f"gap {1e3 * gap[i]:.3f} meV on this grid (no mirror/glide to classify it)", "unknown"
                else:
                    a0, a1, c0, c1 = pl[:, b], pl[:, b + 1], pr[:, b], pr[:, b + 1]
                    differ = np.any(np.abs(a0 - a1) > 1.0)
                    swapped = differ and np.allclose(a0, c1, atol=0.3) and np.allclose(a1, c0, atol=0.3)
                    if swapped:
                        kind, tag = "PROTECTED CROSSING (different mirror/glide parity; the bands exchange order)", "cross"
                    elif differ:
                        kind, tag = f"close approach of bands of different parity (gap {1e3 * gap[i]:.2f} meV, no coupling)", "close"
                    else:
                        kind, tag = f"avoided crossing (same parity), hybridisation gap {1e3 * gap[i]:.2f} meV", "avoided"
            note = ""
            if tag in ("cross", "zone") and mg > max(5 * gap[i], 0.002):
                mgaps = np.array([model_gap(q, b) for q in ks])
                q = int(np.argmin(mgaps))
                if mgaps[q] < 0.001:
                    note = (f"; the MODEL bands cross at k = {ks[q] / np.pi:.3f} pi instead (gap here "
                            f"{1e3 * mg:.1f} meV): the model has the symmetry but misses the terms that place "
                            "the crossing (e.g. longer-range hoppings)")
                else:
                    note = (f"; the MODEL does not cross (smallest gap {1e3 * mgaps[q]:.1f} meV at k = "
                            f"{ks[q] / np.pi:.3f} pi): it couples states that the symmetry keeps apart "
                            "(check equivalent sites / hoppings, or the band selection)")
            elif tag == "avoided" and abs(mg - gap[i]) > max(0.5 * gap[i], 0.002):
                note = f"; model gap {1e3 * mg:.1f} meV"
            lines.append(f"{where}: {kind}{note}")
            marks.append((tag, k, E[i, b] - fitres["EF"]))
            found += 1
    if not found:
        lines.append(f"no crossings or near-degeneracies (< {1e3 * tol:.0f} meV) among the fitted bands")
    return lines, marks


def isolated_windows(E, nocc, maxn=16, min_gap=0.05):
    """Groups of bands around E_F separated from the rest by more than min_gap (eV).

    Returns a list of (lo, hi, gap) with lo <= 0 < hi relative to the HOMO band (0).
    """
    N = E.shape[1]
    out = []
    for lo in range(0, -maxn, -1):
        a = nocc - 1 + lo
        if a < 0:
            break
        gb = E[:, a].min() - (E[:, a - 1].max() if a > 0 else -np.inf)
        if gb < min_gap:
            continue
        for hi in range(1, maxn + lo + 1):
            b = nocc - 1 + hi
            if b >= N:
                break
            ga = (E[:, b + 1].min() if b + 1 < N else np.inf) - E[:, b].max()
            if ga >= min_gap:
                out.append((lo, hi, min(ga, gb)))
    return sorted(out, key=lambda w: w[1] - w[0])


def optimal_weights(H, atoms, window, nocc, nk=32):
    """Best combination of the given atoms for the bands in window (lo, hi).

    It is the combination with the largest weight in those bands (top eigenvector of the
    band projector restricted to the atoms). Normalised so that the largest weight is +1.
    """
    lo, hi = window
    sel = np.arange(nocc - 1 + lo, nocc + hi)
    M = np.zeros((len(atoms), len(atoms)))
    for k in 2 * np.pi * np.arange(nk) / nk:
        _, V = np.linalg.eigh(Hk(H, k))
        W = V[np.ix_(atoms, sel)]
        M += np.real(W @ W.conj().T)
    w, U = np.linalg.eigh(M)
    v = U[:, -1]
    v = v / v[np.argmax(np.abs(v))]
    return [float(x) for x in np.round(v, 3)]


def default_window(n):
    lo = -(n // 2 - 1) if n % 2 == 0 else -((n - 1) // 2)
    return (lo, lo + n - 1)


def build_model(H, st, sites, window=None, thr=0.01, tol=0.002, Rmax=2, optimise=True, nk=64):
    """Effective model on user-chosen sites, generated from the exact projection.

    sites  : list of (name, [atoms], [weights] or None)
    window : (lo, hi), "auto" or None (n bands centred on E_F, n = number of sites)
    thr    : keep projected couplings with |value| > thr (eV)
    tol    : couplings whose |value| agree within tol share one parameter (0: no grouping)
    Returns (text of the .eff model, info dict).
    """
    n = len(sites)
    if n == 0:
        raise ValueError("pick at least one site")
    if any(not a for _, a, _ in sites):
        raise ValueError("every site needs at least one atom")
    nocc = st.n_electrons // 2
    win = default_window(n) if window is None else window
    if win != "auto" and win[1] - win[0] + 1 != n:
        raise ValueError(f"the band window {win[0]}..{win[1]} has {win[1] - win[0] + 1} bands "
                         f"but {n} sites were picked")
    m = EffModel()
    m.window = win
    wwin = default_window(n) if win == "auto" else win
    for name, atoms, wts in sites:
        if optimise and len(atoms) > 1:
            wts = optimal_weights(H, atoms, wwin, nocc)
        m.sites.append(name)
        m.site_atom.append(list(atoms))
        m.site_w.append(list(wts) if wts is not None and len(atoms) > 1 else None)
    hR, smin = project(H, m, nocc, nk, Rmax)

    onsite = [(i, hR[0][i, i]) for i in range(n)]
    hops, dropped = [], 0.0
    for R in range(Rmax + 1):
        for i in range(n):
            for j in range(n):
                if R == 0 and j <= i:
                    continue
                v = hR[R][i, j]
                if abs(v) > thr:
                    hops.append((i, j, R, v))
                else:
                    dropped = max(dropped, abs(v))

    def groups(items, key):
        """Greedy clustering of values within tol."""
        out = []
        for it in sorted(items, key=key):
            if out and tol > 0 and abs(key(it) - key(out[-1][0])) <= tol:
                out[-1].append(it)
            else:
                out.append([it])
        return out

    lines = ["# generated from the projection of the TB bands on the picked sites",
             f"# kept couplings > {1e3 * thr:.1f} meV (largest dropped: {1e3 * dropped:.1f} meV), "
             f"grouped within {1e3 * tol:.1f} meV",
             f"# min overlap of the sites with the bands: {smin:.3f}" +
             ("  (LOW: the sites miss some band states)" if smin < 0.02 else ""),
             "bands auto" if win == "auto" else f"bands {win[0]} {win[1]}"]
    for name, atoms, w in zip(m.sites, m.site_atom, m.site_w):
        lines.append(f"site {name} atom " + " ".join(map(str, atoms)) +
                     (" weights " + " ".join(f"{x:g}" for x in w) if w else ""))
    params = []
    for g, grp in enumerate(groups(onsite, key=lambda x: x[1]), 1):
        p = f"e{g}"
        val = float(np.mean([v for _, v in grp]))
        params.append((p, val))
        for i, v in grp:
            lines.append(f"onsite {m.sites[i]} {p:6s}          # {v:+.4f}")
    for g, grp in enumerate(groups(hops, key=lambda x: -abs(x[3])), 1):
        p = f"t{g}"
        ref = np.sign(grp[0][3])
        val = float(ref * np.mean([abs(v) for *_, v in grp]))
        params.append((p, val))
        for i, j, R, v in grp:
            s = "" if np.sign(v) == ref else "-"
            lines.append(f"hop {m.sites[i]} {m.sites[j]} {R} {s + p:7s}   # {v:+.4f}")
    for p, v in params:
        lines.append(f"param {p} {v:.5f}")
    info = dict(overlap=smin, n_terms=len(hops), n_params=len(params), dropped=dropped,
                weights={m.sites[i]: m.site_w[i] for i in range(n) if m.site_w[i]})
    return "\n".join(lines) + "\n", info


# ===========================================================================
# density of states and local density of states (tunnelling spectroscopy)
# ===========================================================================
def dos_ldos(H, emin, emax, sigma=0.02, ne=600, nk=101):
    """LDOS of every atom on an energy grid, Gaussian broadening sigma (eV).

    LDOS_i(E) = sum_{k,n} |psi_n(k)_i|^2 g(E - E_n(k)),  averaged over the Brillouin zone.
    H is real, so k and -k give the same |psi|^2: only 0 <= k <= pi is computed.
    Returns (E grid, L) with L of shape (N, ne) in states / eV; DOS per cell = L.sum(0).
    """
    N = H[0].shape[0]
    ks = np.linspace(0, np.pi, nk)
    w = np.full(nk, 2.0)
    w[0] = w[-1] = 1.0
    w /= w.sum()
    eg = np.linspace(emin, emax, ne)
    L = np.zeros((N, ne))
    norm = 1.0 / (np.sqrt(2 * np.pi) * sigma)
    for k, wk in zip(ks, w):
        E, V = np.linalg.eigh(Hk(H, k))
        m = (E > emin - 6 * sigma) & (E < emax + 6 * sigma)
        if not m.any():
            continue
        G = norm * np.exp(-(eg[None, :] - E[m, None]) ** 2 / (2 * sigma ** 2))
        L += wk * (np.abs(V[:, m]) ** 2) @ G
    return eg, L


# ---------------------------------------------------------------------------
# spectra and maps that stay correct at any zoom
# ---------------------------------------------------------------------------
def auto_broadening(window, sigma=None):
    """Gaussian broadening (eV) for the spectra of an energy window (lo, hi): the given sigma if
    it is a positive number, otherwise ('auto') 1/250 of the window, between 0.1 and 20 meV.
    A wide view is smooth; a zoomed view resolves bands or levels a few meV apart."""
    if sigma is not None and sigma > 0:
        return float(sigma)
    return float(np.clip((window[1] - window[0]) / 250.0, 1e-4, 0.02))


def parse_broadening(text):
    """'auto' / '' -> None, otherwise the value in meV converted to eV."""
    t = str(text).strip().lower()
    if t in ("", "auto", "a"):
        return None
    v = float(t) / 1e3
    return v if v > 0 else None


def bloch_cache(H, nk=None, max_bytes=2.5e8):
    """All the eigenstates needed for DOS, LDOS and maps at any energy and any zoom.

    Periodic H ({R: H_R} with some R != 0): eigenvalues and |psi_n(k)|^2 of every band on nk
    points 0 <= k <= pi (H is real, so k and -k give the same |psi|^2). Finite H ({0: H0}): the
    levels and |psi_n|^2. Returns dict(ks, E (nk, N), P (nk, N atoms, N bands) float32, periodic).
    The spectra (cache_spectra, cache_map) interpolate every band linearly between the k points
    (1D linear tetrahedron) and then broaden with a Gaussian: they do not depend on an energy
    grid and stay smooth with a broadening much smaller than the band dispersion between two k.
    """
    N = H[0].shape[0]
    periodic = any(R != 0 for R in H)
    if periodic:
        if nk is None:
            nk = 401 if N <= 80 else (251 if N <= 200 else (151 if N <= 400 else 101))
        nk = int(np.clip(min(nk, max_bytes / (4.0 * N * N)), 21, 4001))
        ks = np.linspace(0.0, np.pi, nk)
    else:
        ks = np.zeros(1)
    E = np.empty((len(ks), N))
    P = np.empty((len(ks), N, N), dtype=np.float32)
    for i, k in enumerate(ks):
        e, v = np.linalg.eigh(Hk(H, k) if periodic else H[0])
        E[i], P[i] = e, np.abs(v) ** 2
    return dict(ks=ks, E=E, P=P, periodic=periodic, N=N)


def _box_gauss(x, a, b, s):
    """A box of unit area on [a, b] convolved with a Gaussian of width s (arrays broadcast)."""
    from scipy.special import erf
    d = b - a
    r = 1.0 / (np.sqrt(2.0) * s)
    with np.errstate(divide="ignore", invalid="ignore"):
        box = (erf((x - a) * r) - erf((x - b) * r)) / (2.0 * d)
    g = np.exp(-((x - 0.5 * (a + b)) * r) ** 2) * (r / np.sqrt(np.pi))
    return np.where(d < 1e-4 * s, g, box)


def _band_profiles(c, n, x, sigma):
    """F (nk, len(x)): the spectrum of band n at the energies x split on the k points, with the
    Brillouin-zone weights included (sum over k and atoms of P * F = DOS of the band)."""
    e = c["E"][:, n]
    x = np.atleast_1d(np.asarray(x, float))
    if not c["periodic"]:
        return _box_gauss(x[None, :], e[0], e[0], sigma)
    a, b = np.minimum(e[:-1], e[1:]), np.maximum(e[:-1], e[1:])
    f = np.zeros((len(a), len(x)))
    m = (b >= x.min() - 8 * sigma) & (a <= x.max() + 8 * sigma)
    cols = (x >= e.min() - 8 * sigma) & (x <= e.max() + 8 * sigma)     # the band's own energy range
    if m.any() and cols.any():
        f[np.ix_(m, cols)] = _box_gauss(x[None, cols], a[m, None], b[m, None], sigma) / len(a)
    F = np.zeros((len(e), len(x)))
    F[:-1] += 0.5 * f
    F[1:] += 0.5 * f
    return F


def _bands_near(c, lo, hi):
    E = c["E"]
    return np.where((E.max(axis=0) >= lo) & (E.min(axis=0) <= hi))[0]


def cache_spectra(c, x, sigma, atoms=(), shift=0.0):
    """DOS per cell (or of the molecule) and LDOS of the given atoms (states / eV) at the
    energies x (relative to `shift`, e.g. E_F). Returns (dos, L) with L of shape (len(atoms), len(x))."""
    x = np.asarray(x, float) + shift
    atoms = list(atoms)
    dos, L = np.zeros(len(x)), np.zeros((len(atoms), len(x)))
    for n in _bands_near(c, x.min() - 8 * sigma, x.max() + 8 * sigma):
        F = _band_profiles(c, n, x, sigma)
        dos += F.sum(axis=0)
        if atoms:
            L += c["P"][:, atoms, n].T @ F
    return dos, L


def cache_map(c, E0, sigma, shift=0.0):
    """LDOS of every atom at exactly the energy E0 (relative to `shift`): the dI/dV map."""
    E0 = float(E0) + shift
    out = np.zeros(c["N"])
    for n in _bands_near(c, E0 - 8 * sigma, E0 + 8 * sigma):
        out += c["P"][:, :, n].T @ _band_profiles(c, n, [E0], sigma)[:, 0]
    return out


def cache_band_map(c, n, k=None):
    """|psi_n|^2 on the atoms: band n averaged over the Brillouin zone (k None) or at the k
    point closest to k (units 1/period; |psi(k)|^2 = |psi(-k)|^2). Sums to 1."""
    P = c["P"][:, :, n].astype(float)
    if not c["periodic"]:
        return P[0]
    if k is not None:
        kk = abs((float(k) + np.pi) % (2 * np.pi) - np.pi)
        return P[int(np.argmin(np.abs(c["ks"] - kk)))]
    w = np.full(len(c["ks"]), 1.0)
    w[0] = w[-1] = 0.5
    return (w / w.sum()) @ P


def cache_nearest_band(c, E0, k=None, shift=0.0):
    """Index of the band (level) closest to the energy E0 (relative to shift) at k (or anywhere)."""
    E0 = float(E0) + shift
    E = c["E"]
    if not c["periodic"] or k is None:
        d = np.abs(E - E0).min(axis=0)
    else:
        kk = abs((float(k) + np.pi) % (2 * np.pi) - np.pi)
        d = np.abs(E[int(np.argmin(np.abs(c["ks"] - kk)))] - E0)
    return int(np.argmin(d))


def ldos_image(st, values, cells=(-1, 0, 1), sigma=0.9, res=0.1, margin=3.0):
    """Simulated constant-height tunnelling map: every atom is a Gaussian spot of width
    sigma (A, mimics the pz orbital seen by the tip) weighted by its LDOS.
    Returns (image, extent) for imshow."""
    from scipy.ndimage import gaussian_filter
    P = np.concatenate([st.xy + c * st.T for c in cells])
    v = np.tile(values, len(cells))
    x0, y0 = P.min(axis=0) - margin
    x1, y1 = P.max(axis=0) + margin
    nx, ny = int((x1 - x0) / res) + 1, int((y1 - y0) / res) + 1
    img = np.zeros((ny, nx))
    ix = np.clip(np.rint((P[:, 0] - x0) / res).astype(int), 0, nx - 1)
    iy = np.clip(np.rint((P[:, 1] - y0) / res).astype(int), 0, ny - 1)
    np.add.at(img, (iy, ix), v)
    img = gaussian_filter(img, sigma / res, mode="constant")
    return img, (x0, x0 + (nx - 1) * res, y0, y0 + (ny - 1) * res)


# ===========================================================================
# structure editing
# ===========================================================================
def copy_structure(st):
    new = Structure(st.xy.copy(), list(st.el), st.T.copy(), st.name)
    new.skipped = list(getattr(st, "skipped", []))
    new.periodic = getattr(st, "periodic", True)
    return new


def remove_atoms(st, positions, tol=0.3):
    """Copy of st without the atoms at the given positions (A). The removal is periodic:
    the atom is removed in every unit cell (sp3 carbon or vacancy: no pz orbital)."""
    new = copy_structure(st)
    keep = np.ones(new.N, bool)
    for p in positions:
        d = np.linalg.norm(new.xy - p, axis=1)
        for c in (-1, 1):
            d = np.minimum(d, np.linalg.norm(new.xy + c * new.T - p, axis=1))
        i = int(np.argmin(d))
        if d[i] < tol:
            keep[i] = False
    new.xy = new.xy[keep]
    new.el = [e for e, k in zip(new.el, keep) if k]
    return new


def symmetry_ops(st, tol=0.1):
    """Point operations of the periodic structure (besides the translations).

    In the frame s = along T, p = across T, every candidate is (s, p) -> (ss s + a, sp p + b)
    with ss, sp = +-1 (mirror across the axis, mirror along it, C2 rotation, glide with a
    fractional translation a). An operation is kept when it maps every atom onto an atom
    (within tol A, positions modulo T). Returns dict(u, v, L, ops=[(ss, sp, a, b), ...])."""
    T = np.asarray(st.T, float)
    L = float(np.linalg.norm(T))
    u = T / L
    v = np.array([-u[1], u[0]])
    s, p = st.xy @ u, st.xy @ v
    ops = []
    for ss in (1, -1):
        for sp_ in (1, -1):
            if ss == 1 and sp_ == 1:
                continue
            b = (p.min() + p.max()) if sp_ == -1 else 0.0
            for a in np.unique(np.round((s - ss * s[0]) % L, 6)):
                s2, p2 = (ss * s + a) % L, sp_ * p + b
                ds = np.abs(((s2[:, None] - s[None, :]) + L / 2) % L - L / 2)
                if np.all(np.any((ds < tol) & (np.abs(p2[:, None] - p[None, :]) < tol), axis=1)):
                    ops.append((ss, sp_, float(a), float(b)))
    return dict(u=u, v=v, L=L, ops=ops)


def sublattice_colors(st):
    """Two-colouring (0/1 per atom) of the nearest-neighbour graph, periodic bonds included,
    or None if the lattice is not bipartite. A translation may swap the colours, both cases
    are tried."""
    H = hoppings(st, t=(1.0,))
    N = st.N
    adj = [set() for _ in range(N)]
    for R, M in H.items():
        for i, j in zip(*np.nonzero(M)):
            if R == 0 and i == j:
                continue
            adj[i].add((j, R))
            adj[j].add((i, -R))
    for par in (0, 1):
        col = [-1] * N
        ok = True
        for s in range(N):
            if col[s] >= 0:
                continue
            col[s] = 0
            stack = [s]
            while stack and ok:
                i = stack.pop()
                for j, R in adj[i]:
                    want = col[i] ^ 1 ^ ((R * par) & 1)
                    if col[j] < 0:
                        col[j] = want
                        stack.append(j)
                    elif col[j] != want:
                        ok = False
                        break
            if not ok:
                break
        if ok:
            return np.array(col)
    return None


def sublattice_balance(st):
    """(n_A, n_B) of the nearest-neighbour graph or None if it is not bipartite. For a
    bipartite pz lattice with nearest-neighbour hopping, |n_A - n_B| zero-energy bands
    exist per unit cell (Lieb)."""
    col = sublattice_colors(st)
    if col is None:
        return None
    nA = int(np.sum(col == 0))
    return nA, st.N - nA


def write_xyz(st, path):
    with open(path, "w") as f:
        f.write(f"{st.N}\nperiod={st.T[0]:.6f},{st.T[1]:.6f}  ({st.name})\n")
        for e, (x, y) in zip(st.el, st.xy):
            f.write(f"{e} {x:.6f} {y:.6f} 0.0\n")


# ===========================================================================
# finite chains (open ends)
# ===========================================================================
def prune_dangling(st, max_neighbours=1):
    """Remove, repeatedly, atoms with at most max_neighbours nearest neighbours (dangling
    atoms at cut ends). Returns (new structure, number removed)."""
    new = copy_structure(st)
    removed = 0
    while new.N:
        H = hoppings(new, t=(1.0,))
        deg = (H[0] != 0).sum(axis=1) - (np.diag(H[0]) != 0)
        for R, M in H.items():
            if R:
                deg = deg + (M != 0).sum(axis=1) + (M != 0).sum(axis=0)
        bad = deg <= max_neighbours
        if not bad.any():
            break
        new.xy, new.el = new.xy[~bad], [e for e, b in zip(new.el, bad) if not b]
        removed += int(bad.sum())
    return new, removed


def finite_chain(st, n, offset=0.0, prune=True, extra=0.0):
    """Finite piece of the periodic structure: n unit cells with open ends.

    The chain is cut at the cell boundary shifted by `offset` (A, along T), so different
    terminations can be tried; `extra` (A) lengthens the piece beyond n cells (e.g. half a
    cell, to end on the same kind of column as it starts). prune: remove atoms left with a
    single neighbour.
    """
    T = np.asarray(st.T, float)
    L = np.linalg.norm(T)
    if L == 0:
        raise ValueError("the structure has no period")
    u = T / L
    cells = range(-2, n + 3 + int(np.ceil(max(extra, 0) / L)))
    P = np.concatenate([st.xy + c * T for c in cells])
    el = [e for c in cells for e in st.el]
    s = P @ u
    s0 = (st.xy @ u).min() + offset
    keep = (s >= s0 - 1e-6) & (s < s0 + n * L + extra - 1e-6)
    fs = Structure(P[keep], [e for e, k in zip(el, keep) if k], [0.0, 0.0],
                   f"{st.name}, {n} units")
    fs.periodic = False
    fs.skipped = []
    npruned = 0
    if prune:
        fs, npruned = prune_dangling(fs)
    fs.n_pruned = npruned
    return fs


def _symmetry_of(P, tol=0.1):
    """'mirror' (perpendicular to the long axis), 'C2', or None for a finite flake."""
    c = P - P.mean(axis=0)
    u = np.linalg.svd(c, full_matrices=False)[2][0]
    v = np.array([-u[1], u[0]])
    sp = np.c_[c @ u, c @ v]
    tree = cKDTree(sp)
    for name, img in (("mirror", sp * [-1, 1]), ("C2", -sp)):
        if np.all(tree.query(img)[0] < tol):
            return name
    return None


def atoms_in_rings(xy, a=A_CC, tol=0.15):
    """True for every atom that belongs to a complete hexagon of the piece."""
    tree = cKDTree(xy)
    out = np.zeros(len(xy), bool)
    ang = np.deg2rad(np.arange(0, 360, 30))
    for i, p in enumerate(xy):
        for c in p + a * np.c_[np.cos(ang), np.sin(ang)]:
            if len(tree.query_ball_point(c, a + tol)) == 6 and len(tree.query_ball_point(c, a - tol)) == 0:
                out[i] = True
                break
    return out


def ring_units(xy, a=A_CC, tol=0.15):
    """Split a piece into its fused-ring units: bonds that belong to no hexagon (single bonds
    linking units, like the olympicene-olympicene bond of the chain) are cut. Returns
    (labels, sizes, connected): a unit label per atom, the atom count of every unit, and
    whether the whole piece is one connected molecule."""
    xy = np.asarray(xy, float)
    N = len(xy)
    tree = cKDTree(xy)
    bonds = tree.query_pairs(a + tol, output_type="ndarray")
    ang = np.deg2rad(np.arange(0, 360, 30))
    rings = set()
    for p in xy:
        for c in p + a * np.c_[np.cos(ang), np.sin(ang)]:
            idx = tree.query_ball_point(c, a + tol)
            if len(idx) == 6 and len(tree.query_ball_point(c, a - tol)) == 0:
                rings.add(tuple(sorted(idx)))
    in_ring = set()
    for r in rings:
        rs = set(r)
        for i, j in bonds:
            if i in rs and j in rs:
                in_ring.add((min(i, j), max(i, j)))

    def comps(edges):
        par = list(range(N))

        def f(x):
            while par[x] != x:
                par[x] = par[par[x]]
                x = par[x]
            return x
        for i, j in edges:
            par[f(i)] = f(j)
        return np.array([f(i) for i in range(N)])
    lab = comps(sorted(in_ring))
    _, lab, sizes = np.unique(lab, return_inverse=True, return_counts=True)
    whole = comps([tuple(b) for b in bonds])
    return lab, sizes, len(np.unique(whole)) == 1


def unit_sizes(st):
    """Atom counts of the fused-ring units of a periodic structure whose units are linked by
    single bonds (empty set when the chain is one fused ribbon, like a GNR)."""
    T = np.asarray(st.T, float)
    L = np.linalg.norm(T)
    u = T / L
    P = np.concatenate([st.xy + c * T for c in range(-3, 4)])
    lab, sizes, _ = ring_units(P)
    s = P @ u
    lo, hi = (st.xy @ u).min() - L, (st.xy @ u).max() + L
    out = set()
    for k in range(len(sizes)):
        sk = s[lab == k]
        if sk.min() > lo and sk.max() < hi:
            out.add(int(sizes[k]))
    return out if len(out) and max(out) < len(st.xy) * 3 else set()


def symmetric_terminations(st, n, prune=True, whole_rings=True):
    """All the ways to cut n (or n and a fraction) cells so that the two ends are equivalent
    (the piece is symmetric under a mirror perpendicular to the chain or a C2 rotation).
    Cuts are tried between every pair of atom columns. whole_rings: keep only pieces where
    every atom belongs to a complete hexagon (no broken rings or stray atoms at the ends; for
    units linked by single bonds, like the olympicene chain, this leaves whole units: odd
    chains are the pieces with an extra half cell). When the chain is made of fused-ring units
    linked by single bonds, the piece must also consist of whole units. Returns a list of
    dict(offset, extra, N, units, symmetry), smallest extension first, one per distinct
    structure (units = number of whole units, None for a fused ribbon)."""
    usz = unit_sizes(st) if whole_rings else set()
    T = np.asarray(st.T, float)
    L = np.linalg.norm(T)
    u = T / L
    sv = np.unique(np.round(np.mod(st.xy @ u, L), 3))
    sv = np.r_[sv, sv[0] + L]
    cuts = 0.5 * (sv[:-1] + sv[1:])                           # between atom columns, one period
    s_min = (st.xy @ u).min()
    extras = np.unique(np.round(np.mod(cuts[None, :] - cuts[:, None], L), 3))
    found, seen = [], set()
    for c1 in cuts:
        off = np.mod(c1 - s_min, L)
        for ex in extras:
            fs = finite_chain(st, n, off, prune, ex)
            if fs.N < 2:
                continue
            sym = _symmetry_of(fs.xy)
            if sym is None or (whole_rings and not atoms_in_rings(fs.xy).all()):
                continue
            nunits = None
            if usz:
                lab, sizes, conn = ring_units(fs.xy)
                if not conn or not all(int(z) in usz for z in sizes):
                    continue
                nunits = len(sizes)
            q = fs.xy - fs.xy.mean(axis=0)
            key = (fs.N, tuple(np.round(np.sort(np.linalg.norm(q, axis=1)), 1)))
            if key in seen:
                continue
            seen.add(key)
            found.append(dict(offset=float(off), extra=float(ex), N=fs.N, units=nunits,
                              symmetry=sym))
    found.sort(key=lambda d: (d["extra"], d["N"]))
    return found


def levels_ldos(H0, emin, emax, sigma=0.02, ne=600):
    """Eigenstates of a finite system and the LDOS of every atom (Gaussian broadening).
    Returns (E, V, E grid, L) with L of shape (N, ne) in states / eV."""
    E, V = np.linalg.eigh(H0)
    eg = np.linspace(emin, emax, ne)
    m = (E > emin - 6 * sigma) & (E < emax + 6 * sigma)
    G = np.exp(-(eg[None, :] - E[m, None]) ** 2 / (2 * sigma ** 2)) / (np.sqrt(2 * np.pi) * sigma)
    L = (np.abs(V[:, m]) ** 2) @ G
    return E, V, eg, L
