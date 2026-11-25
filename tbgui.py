"""
TB fit GUI: tight-binding bands of 1D-periodic structures and effective-model fitting.

Run:  python tbgui.py        (needs numpy, scipy, matplotlib; tkinter comes with Python)
"""

import os
import glob
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import numpy as np
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.collections import LineCollection

import threading
from math import comb
import tbcore as tb
import tbmany as tm
import tbdmrg as tdm

HERE = os.path.dirname(os.path.abspath(__file__))
EXAMPLES = os.path.join(HERE, "examples")
STRUCT_TYPES = [("Structures", "*.hex *.txt *.cdxml *.mol *.sdf *.xyz"), ("All files", "*.*")]
COLORS = matplotlib.rcParams["axes.prop_cycle"].by_key()["color"]
WIN_CENTRED = "centred on E_F (as many bands as sites)"
WIN_AUTO = "auto (at each k the states living on the sites)"
MAP_LDOS = "dI/dV at the map energy (LDOS)"
MAP_BAND = "clicked band: |psi_n|^2 averaged over k"
MAP_BANDK = "clicked band at the clicked k: |psi_n(k)|^2"
MAP_LEVEL = "clicked level: |psi|^2"
MAP_SPIN = "spin density n_up - n_dn"
MAP_UP = "dI/dV spin up at the map energy"
MAP_DN = "dI/dV spin down at the map energy"
MAP_UPDN = "dI/dV up - down at the map energy"
SOLVER_ED = "ED (exact diagonalisation)"
SOLVER_DMRG = "DMRG (block2)"
DMRG_MAX_AUTO_STATES = 30


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("TB bands and effective-model fitting")
        self.geometry("1400x860")
        self.st = self.H = self.bandres = self.model = self.fitres = None
        self.path = None
        self._views, self._view_jobs, self._mouse_down = {}, {}, {}

        nb = ttk.Notebook(self)
        nb.pack(fill="both", expand=True)
        self.tab1, self.tab2, self.tab3, self.tab4, self.tab5, self.tab6 = (ttk.Frame(nb) for _ in range(6))
        self.tabMF = ttk.Frame(nb)
        nb.add(self.tab1, text="  1. Structure and bands  ")
        nb.add(self.tab2, text="  2. Effective model fit  ")
        nb.add(self.tab3, text="  3. Finite chain  ")
        nb.add(self.tab4, text="  4. Spin model  ")
        nb.add(self.tabMF, text="  5. Mean-field Hubbard  ")
        nb.add(self.tab5, text="  6. U fit to mean field  ")
        nb.add(self.tab6, text="  7. ED: spectra and maps  ")
        self.nb = nb
        self.status = tk.StringVar(value="Load a structure or pick an example.")
        ttk.Label(self, textvariable=self.status, anchor="w", relief="sunken").pack(fill="x")
        self._build_tab1()
        self._build_tab2()
        self._build_tab3()
        self._build_tab4()
        self._build_tabMF()
        self._build_tab5()
        self._build_tab6()

    # ------------------------------------------------------------------ tab 1
    def _build_tab1(self):
        left = ttk.Frame(self.tab1, padding=6)
        left.pack(side="left", fill="y")

        box = ttk.LabelFrame(left, text="Structure", padding=6)
        box.pack(fill="x")
        ttk.Button(box, text="Load structure file...", command=self.load_file).pack(fill="x")
        ttk.Label(box, text="Examples:").pack(anchor="w", pady=(6, 0))
        self.ex_var = tk.StringVar()
        ex = sorted(os.path.basename(p) for p in glob.glob(os.path.join(EXAMPLES, "*"))
                    if p.lower().endswith((".hex", ".cdxml", ".mol", ".xyz")))
        cb = ttk.Combobox(box, textvariable=self.ex_var, values=ex, state="readonly", width=30)
        cb.pack(fill="x")
        cb.bind("<<ComboboxSelected>>", lambda e: self.load(os.path.join(EXAMPLES, self.ex_var.get())))
        self.info = tk.StringVar(value="-")
        ttk.Label(box, textvariable=self.info, justify="left", wraplength=260).pack(anchor="w", pady=4)

        row = ttk.Frame(box)
        row.pack(fill="x")
        ttk.Label(row, text="Period T (A):").pack(side="left")
        self.Tx, self.Ty = tk.StringVar(), tk.StringVar()
        ttk.Entry(row, textvariable=self.Tx, width=8).pack(side="left")
        ttk.Entry(row, textvariable=self.Ty, width=8).pack(side="left")
        ttk.Button(row, text="Set", width=4, command=self.set_period).pack(side="left")

        box = ttk.LabelFrame(left, text="Modify structure (sp3 / vacancies)", padding=6)
        box.pack(fill="x", pady=(6, 0))
        row = ttk.Frame(box)
        row.pack(fill="x")
        ttk.Label(row, text="click on an atom:").pack(side="left")
        self.click_mode = tk.StringVar(value="spectrum")
        ttk.Radiobutton(row, text="dI/dV", value="spectrum", variable=self.click_mode).pack(side="left")
        ttk.Radiobutton(row, text="remove (sp3)", value="remove", variable=self.click_mode).pack(side="left")
        row = ttk.Frame(box)
        row.pack(fill="x", pady=(2, 0))
        ttk.Button(row, text="Undo", width=6, command=self.undo_remove).pack(side="left")
        ttk.Button(row, text="Restore all", command=self.restore_all).pack(side="left")
        ttk.Button(row, text="Save structure as...", command=self.save_structure).pack(side="left")

        box = ttk.LabelFrame(left, text="Tight binding", padding=6)
        box.pack(fill="x", pady=6)
        self.t_vars = []
        for lab, val in zip(["t1  (1st NN, eV)", "t2  (2nd NN, eV)", "t3  (3rd NN, eV)"], tb.T_DEFAULT):
            r = ttk.Frame(box)
            r.pack(fill="x")
            ttk.Label(r, text=lab, width=18).pack(side="left")
            v = tk.StringVar(value=str(val))
            ttk.Entry(r, textvariable=v, width=10).pack(side="left")
            self.t_vars.append(v)
        ttk.Label(box, text="set t2 = t3 = 0 for nearest neighbours only",
                  foreground="gray").pack(anchor="w")
        self.nk, self.ncells = self._entry2(box, "k points", "201", "cells shown", "3")
        self.emin, self.emax = self._entry2(box, "E-E_F  min", "-3", "max (eV)", "3")
        self.show_idx = tk.BooleanVar(value=False)
        ttk.Checkbutton(box, text="show atom indices", variable=self.show_idx,
                        command=lambda: self.draw1(keep_view=True)).pack(anchor="w")
        row = ttk.Frame(box)
        row.pack(fill="x", pady=(6, 0))
        ttk.Button(row, text="Calculate bands", command=self.calc_bands).pack(side="left", fill="x", expand=True)
        ttk.Button(row, text="Export (CSV)...", command=self.export_bands).pack(side="left")

        box = ttk.LabelFrame(left, text="DOS and tunnelling LDOS (dI/dV)", padding=6)
        box.pack(fill="x")
        self.do_dos = tk.BooleanVar(value=True)
        ttk.Checkbutton(box, text="compute together with the bands", variable=self.do_dos).pack(anchor="w")
        self.broad = self._entry(box, "broadening (meV)", "auto")
        self.emap = self._entry(box, "map energy E - E_F", "0.0")
        self.tipw = self._entry(box, "tip resolution (A)", "0.9")
        row = ttk.Frame(box)
        row.pack(fill="x")
        ttk.Label(row, text="map:").pack(side="left")
        self.map1_mode = tk.StringVar(value=MAP_LDOS)
        cb = ttk.Combobox(row, textvariable=self.map1_mode, state="readonly", width=34,
                          values=[MAP_LDOS, MAP_BAND, MAP_BANDK])
        cb.pack(side="left", padx=2)
        cb.bind("<<ComboboxSelected>>", lambda e: self.draw1(keep_view=True))
        ttk.Label(box, text="click the bands/DOS: map energy (and the band)\nclick an atom: its dI/dV spectrum\n"
                            "tip resolution 0: LDOS on the atoms\n"
                            "zoom/pan with the toolbar: DOS, map and spectra are\n"
                            "recomputed for the energy window shown; broadening\n"
                            "'auto' = window / 250 (0.1 to 20 meV)", foreground="gray").pack(anchor="w")
        row = ttk.Frame(box)
        row.pack(fill="x", pady=(4, 0))
        ttk.Button(row, text="Update DOS / LDOS", command=self.calc_dos).pack(side="left", fill="x", expand=True)
        ttk.Button(row, text="Export (CSV)...", command=self.export_dos).pack(side="left")

        self.out1 = tk.Text(left, width=38, height=7, font=("Courier", 9))
        self.out1.pack(fill="both", expand=True)

        right = ttk.Frame(self.tab1)
        right.pack(side="left", fill="both", expand=True)
        self.fig1 = Figure(figsize=(10, 8))
        gs = self.fig1.add_gridspec(2, 3, width_ratios=[2.6, 1, 0.45], height_ratios=[1, 1])
        self.ax_s1 = self.fig1.add_subplot(gs[0, 0])
        self.ax_b1 = self.fig1.add_subplot(gs[0, 1])
        self.ax_d1 = self.fig1.add_subplot(gs[0, 2], sharey=self.ax_b1)
        self.ax_m1 = self.fig1.add_subplot(gs[1, 0])
        self.ax_p1 = self.fig1.add_subplot(gs[1, 1:])
        self.dosres, self.sel_atoms = None, []
        self._win1, self._band1, self._spec1 = None, None, None
        self.can1 = FigureCanvasTkAgg(self.fig1, right)
        NavigationToolbar2Tk(self.can1, right).update()
        self.can1.get_tk_widget().pack(fill="both", expand=True)
        self._on_clicks(self.can1, self.on_click1)
        self._watch_view(self.can1, "tab1", lambda: (*self.ax_b1.get_ylim(), *self.ax_p1.get_xlim()),
                         self._zoom1)

    def _entry2(self, parent, l1, v1, l2, v2):
        r = ttk.Frame(parent)
        r.pack(fill="x")
        out = []
        for lab, val, w in ((l1, v1, 10), (l2, v2, 11)):
            ttk.Label(r, text=lab, width=w).pack(side="left")
            v = tk.StringVar(value=val)
            ttk.Entry(r, textvariable=v, width=6).pack(side="left", padx=(0, 4))
            out.append(v)
        return out

    def _entry(self, parent, label, val):
        r = ttk.Frame(parent)
        r.pack(fill="x")
        ttk.Label(r, text=label, width=18).pack(side="left")
        v = tk.StringVar(value=val)
        ttk.Entry(r, textvariable=v, width=10).pack(side="left")
        return v

    def t_values(self):
        t = [float(v.get() or 0) for v in self.t_vars]
        while len(t) > 1 and t[-1] == 0:
            t.pop()
        return tuple(t)

    # ------------------------------------------------------------------ clicks and zoom
    def _on_clicks(self, canvas, handler, tol=5):
        """Call handler(press event) for a click: press and release of the same button within
        tol pixels. Clicks therefore work also while the zoom or pan tool of the toolbar is on
        (a zoom rectangle or a pan moves the mouse and is not a click)."""
        state = {}

        def press(ev):
            state["ev"] = ev
            self._mouse_down[canvas] = True

        def release(ev):
            self._mouse_down[canvas] = False
            p = state.pop("ev", None)
            if p is None or ev.button != p.button or abs(ev.x - p.x) > tol or abs(ev.y - p.y) > tol:
                return
            handler(p)

        canvas.mpl_connect("button_press_event", press)
        canvas.mpl_connect("button_release_event", release)

    def _watch_view(self, canvas, name, getter, redraw, delay=250):
        """Redraw when the user zooms or pans (toolbar zoom, pan, home, back, forward): after every
        draw of the canvas getter() (the energy window shown) is compared with the value stored
        by the last redraw in self._views[name]; when it changed, redraw() is called once the view
        has settled (no mouse button pressed for `delay` ms)."""
        def run():
            self._view_jobs.pop(name, None)
            if self._mouse_down.get(canvas):
                self._view_jobs[name] = self.after(delay, run)
                return
            try:
                redraw()
            except Exception as e:          # never leave the GUI in a broken state
                self.status.set(f"redraw after zoom failed: {e}")

        def on_draw(ev):
            old = self._views.get(name)
            if old is None:
                return
            try:
                cur = np.asarray(getter(), float)
            except Exception:
                return
            if cur.shape == np.shape(old) and np.allclose(cur, old, rtol=1e-7, atol=1e-9):
                return
            job = self._view_jobs.pop(name, None)
            if job:
                self.after_cancel(job)
            self._view_jobs[name] = self.after(delay, run)

        canvas.mpl_connect("draw_event", on_draw)

    @staticmethod
    def _get_lims(axes):
        return {ax: (ax.get_xlim(), ax.get_ylim()) for ax in axes}

    @staticmethod
    def _set_lims(saved, x=True, y=True, only=None):
        for ax, (xl, yl) in saved.items():
            if only is not None and ax not in only:
                continue
            if x:
                ax.set_xlim(*xl)
            if y:
                ax.set_ylim(*yl)

    @staticmethod
    def _new_window(old, ylim_bands, xlim_spec):
        """Energy window after a zoom: the band (level) axis or the spectra axis, whichever the
        user changed; the two are kept equal."""
        b, s = tuple(ylim_bands), tuple(xlim_spec)
        if old is not None and np.allclose(b, old, rtol=1e-7, atol=1e-9) and not np.allclose(s, old):
            return s
        return b

    def _sigma(self, var, window):
        """Broadening (eV) from an entry ('auto' or meV) for the energy window shown."""
        try:
            s = tb.parse_broadening(var.get())
        except ValueError:
            s = None
        return tb.auto_broadening(window, s), s is None

    @staticmethod
    def _sigma_text(sig, auto):
        v = 1e3 * sig
        return (f"{v:.3g} meV" if v < 10 else f"{v:.0f} meV") + (" auto" if auto else "")

    @staticmethod
    def _egrid(window, sig):
        n = int(np.clip(np.ceil(4 * (window[1] - window[0]) / sig), 801, 6001))
        return np.linspace(window[0], window[1], n)

    def _draw_map(self, ax, st, v, cells, tip, signed=False, cmap=None, unit="states/eV"):
        """Simulated constant-height map of the atomic values v: Gaussian spots of width tip (A)
        or, with tip 0, coloured atoms. signed: values of both signs (blue negative, red positive).
        Every map is scaled to its own maximum, written in the corner (a map in a gap is weak)."""
        bonds = self._chain_bonds(st, cells) if len(cells) > 1 else self._bonds(st)
        vmax = float(np.abs(v).max()) or 1.0
        ax.text(0.01, 0.98, f"max on an atom {float(np.abs(v).max()):.3g} {unit}", transform=ax.transAxes,
                fontsize=7, va="top", color="w" if tip > 0 and not signed else "k", zorder=6)
        if tip > 0:
            img, ext = tb.ldos_image(st, v, cells=cells, sigma=tip)
            if signed:
                m = float(np.abs(img).max()) or 1.0
                ax.imshow(img, origin="lower", extent=ext, cmap=cmap or "RdBu_r", vmin=-m, vmax=m, aspect="equal")
                bonds.set(color="0.3", linewidth=0.5, alpha=0.35)
            else:
                ax.imshow(img, origin="lower", extent=ext, cmap=cmap or "inferno", vmin=0, aspect="equal")
                bonds.set(color="w", linewidth=0.6, alpha=0.18)
            ax.add_collection(bonds)
        else:
            bonds.set(color="0.6", linewidth=0.6, alpha=0.8)
            ax.add_collection(bonds)
            P = np.concatenate([st.xy + c * st.T for c in cells])
            vv = np.tile(v, len(cells))
            if signed:
                ax.scatter(*P.T, c=vv, s=12 + 160 * np.abs(vv) / vmax, cmap=cmap or "RdBu_r", vmin=-vmax,
                           vmax=vmax, edgecolor="0.5", linewidth=0.3, zorder=3)
            else:
                ax.scatter(*P.T, c=vv, s=12 + 160 * vv / vmax, cmap=cmap or "YlOrRd", vmin=0,
                           edgecolor="0.5", linewidth=0.3, zorder=3)
        ax.set_aspect("equal")
        ax.autoscale_view()

    # ------------------------------------------------------------------ tab 2
    def _build_tab2(self):
        self.picks, self.cur = [], -1
        left = ttk.Frame(self.tab2, padding=6)
        left.pack(side="left", fill="y")

        pk = ttk.LabelFrame(left, text="Build a model: pick the sites on the structure", padding=6)
        pk.pack(fill="x")
        ttk.Label(pk, text="left click: add atom to the current site (+)   right click: (-)\n"
                           "click an atom of another site to select that site, again to remove it",
                  foreground="gray").pack(anchor="w")
        row = ttk.Frame(pk)
        row.pack(fill="x", pady=2)
        for txt, cmd in [("New site", self.new_site), ("Delete site", self.delete_site),
                         ("Clear all", self.clear_sites), ("Sites from editor", self.sites_from_editor)]:
            ttk.Button(row, text=txt, command=cmd).pack(side="left")
        row = ttk.Frame(pk)
        row.pack(fill="x")
        ttk.Label(row, text="name of current site:").pack(side="left")
        self.name_var = tk.StringVar()
        ttk.Entry(row, textvariable=self.name_var, width=10).pack(side="left")
        ttk.Button(row, text="Rename", command=self.rename_site).pack(side="left")
        self.picks_var = tk.StringVar(value="no sites yet: click atoms in the plot")
        ttk.Label(pk, textvariable=self.picks_var, font=("Courier", 9), justify="left").pack(anchor="w", pady=2)
        row = ttk.Frame(pk)
        row.pack(fill="x")
        ttk.Label(row, text="bands to fit:").pack(side="left")
        self.win_var = tk.StringVar(value=WIN_CENTRED)
        self.win_cb = ttk.Combobox(row, textvariable=self.win_var, width=40,
                                   values=[WIN_CENTRED, WIN_AUTO])
        self.win_cb.pack(side="left")
        self.win_cb.bind("<<ComboboxSelected>>", lambda e: self.draw_picks())
        row = ttk.Frame(pk)
        row.pack(fill="x", pady=2)
        self.thr = tk.StringVar(value="10")
        self.tol = tk.StringVar(value="2")
        self.rmax = tk.StringVar(value="2")
        for lab, var in [("keep > (meV)", self.thr), ("group within (meV)", self.tol), ("max cells R", self.rmax)]:
            ttk.Label(row, text=lab).pack(side="left")
            ttk.Entry(row, textvariable=var, width=4).pack(side="left", padx=(0, 6))
        self.opt_w = tk.BooleanVar(value=True)
        ttk.Checkbutton(pk, text="optimise the weights of multi-atom sites (finds e.g. +1 -1 states)",
                        variable=self.opt_w).pack(anchor="w")
        ttk.Button(pk, text="Build model from sites and fit", command=self.build_from_sites).pack(fill="x", pady=(4, 0))

        row = ttk.Frame(left)
        row.pack(fill="x", pady=(6, 0))
        ttk.Button(row, text="Load model...", command=self.load_eff).pack(side="left")
        ttk.Button(row, text="Save model...", command=self.save_eff).pack(side="left")
        ttk.Button(row, text="Syntax help", command=self.eff_help).pack(side="left")
        ttk.Label(left, text="Effective model (generated or written by hand; edit freely, then FIT):"
                  ).pack(anchor="w", pady=(4, 0))
        ef = ttk.Frame(left)
        ef.pack(fill="both")
        self.editor = tk.Text(ef, width=56, height=14, font=("Courier", 9), undo=True, wrap="none")
        xs = ttk.Scrollbar(ef, orient="horizontal", command=self.editor.xview)
        ys = ttk.Scrollbar(ef, orient="vertical", command=self.editor.yview)
        self.editor.configure(xscrollcommand=xs.set, yscrollcommand=ys.set)
        self.editor.grid(row=0, column=0, sticky="nsew")
        ys.grid(row=0, column=1, sticky="ns")
        xs.grid(row=1, column=0, sticky="ew")
        row = ttk.Frame(left)
        row.pack(fill="x", pady=4)
        self.use_proj = tk.BooleanVar(value=True)
        ttk.Checkbutton(row, text="initial values from projection",
                        variable=self.use_proj).pack(side="left")
        ttk.Button(row, text="Show model", command=self.show_sites).pack(side="left", padx=4)
        ttk.Button(row, text="FIT", command=self.run_fit).pack(side="left")
        self.out2 = tk.Text(left, width=56, height=12, font=("Courier", 9))
        self.out2.pack(fill="both", expand=True)

        right = ttk.Frame(self.tab2)
        right.pack(side="left", fill="both", expand=True)
        self.fig2 = Figure(figsize=(9, 7))
        gs = self.fig2.add_gridspec(1, 2, width_ratios=[2.2, 1])
        self.ax_s2 = self.fig2.add_subplot(gs[0])
        self.ax_b2 = self.fig2.add_subplot(gs[1])
        self.can2 = FigureCanvasTkAgg(self.fig2, right)
        NavigationToolbar2Tk(self.can2, right).update()
        self.can2.get_tk_widget().pack(fill="both", expand=True)
        self._on_clicks(self.can2, self.on_click2)

    # ------------------------------------------------------------------ loading
    def load_file(self):
        p = filedialog.askopenfilename(filetypes=STRUCT_TYPES, initialdir=EXAMPLES)
        if p:
            self.load(p)

    def load(self, path):
        try:
            st = tb.load_structure(path)
            alts = getattr(st, "period_alternatives", [])
            if alts:
                # the drawing repeats along more than one direction: ask which is the ribbon axis
                alt = alts[0]
                alt_st = tb.load_structure(path, prefer=alt)
                keep = messagebox.askyesno(
                    "Period direction",
                    "This drawing repeats along more than one direction (it shows too few repeat units to "
                    "tell which is the ribbon axis):\n\n"
                    f"  A) T = ({st.T[0]:.2f}, {st.T[1]:.2f}) A, {st.N} atoms per cell\n"
                    f"  B) T = ({alt_st.T[0]:.2f}, {alt_st.T[1]:.2f}) A, {alt_st.N} atoms per cell\n\n"
                    "Use A? (No = use B.)\n\nTo avoid the question, draw three or more repeat units along "
                    "the ribbon, or mark the bonds to the next cell with star atoms (*) or dashed bonds.")
                if not keep:
                    st = alt_st
        except Exception as e:
            messagebox.showerror("Cannot read structure", str(e))
            return
        self.st, self.path = st, path
        self.st_orig, self.removed = tb.copy_structure(st), []
        self.src_text = open(path).read() if path.lower().endswith((".hex", ".txt")) else None
        self.H = self.bandres = self.fitres = self.dosres = None
        self.sel_atoms = []
        self.Tx.set(f"{st.T[0]:.4f}")
        self.Ty.set(f"{st.T[1]:.4f}")
        self.update_info()
        eff = os.path.splitext(path)[0] + ".eff"
        self.picks, self.cur = [], -1
        self.update_picks_label()
        if os.path.exists(eff):
            self.editor.delete("1.0", "end")
            self.editor.insert("1.0", open(eff).read())
            self.sites_from_editor(quiet=True)
        else:
            self.editor.delete("1.0", "end")
        if not getattr(st, "periodic", True):
            messagebox.showwarning(
                "No period", "No period found. The program looks for (1) two or more drawn repeat "
                "units, (2) star atoms (*, R, X) or (3) dashed bonds at the ends of the drawing. "
                "Draw one of these, or type the translation vector T (A) and press Set.")
            self.draw1()
            return
        self.calc_bands()

    def update_info(self):
        st = self.st
        extra = f"\nignored elements: {', '.join(st.skipped)}" if getattr(st, "skipped", []) else ""
        rem = f"\nremoved (sp3): {len(self.removed)} atom(s) per cell" if self.removed else ""
        src = getattr(st, "period_source", None)
        if src and getattr(st, "periodic", True):
            rem = f"\nperiod from: {src}" + rem
        if getattr(st, "n_sp3", 0):
            rem += (f"\nsp3 carbons in the file: {st.n_sp3} (4 neighbours with H, or no double bond "
                    f"and no radical in a Kekule drawing), not in the pi system")
        bal = tb.sublattice_balance(st) if getattr(st, "periodic", True) else None
        if bal:
            d = abs(bal[0] - bal[1])
            bal = f"\nsublattices A/B = {bal[0]}/{bal[1]}" + (
                f": {d} zero-energy band(s) with NN hopping" if d else "")
        else:
            bal = ""
        self.info.set(f"{st.name}\n{st.N} pz sites, {st.n_electrons} pi electrons\n"
                      f"|T| = {np.linalg.norm(st.T):.3f} A{extra}{rem}{bal}")

    # ------------------------------------------------------------------ editing
    def apply_edits(self):
        """Rebuild the structure from the original minus the removed atoms and recompute."""
        old = self.st
        sel_pos = [old.xy[i] for i in self.sel_atoms]
        pick_pos = [[old.xy[a] for a in p["atoms"]] for p in self.picks]
        self.st = tb.remove_atoms(self.st_orig, self.removed)
        self.st.name = self.st_orig.name + (f" ({len(self.removed)} removed)" if self.removed else "")

        def find(p):
            d = np.linalg.norm(self.st.xy - p, axis=1)
            i = int(np.argmin(d))
            return i if d[i] < 0.3 else None
        self.sel_atoms = [i for i in map(find, sel_pos) if i is not None]
        for p, pos in zip(self.picks, pick_pos):
            new = [(find(q), w) for q, w in zip(pos, p["w"])]
            p["atoms"] = [i for i, w in new if i is not None]
            p["w"] = [w for i, w in new if i is not None]
        self.update_picks_label()
        self.H = self.bandres = self.fitres = self.dosres = None
        self.update_info()
        if getattr(self.st, "periodic", True):
            self.calc_bands()
        else:
            self.draw1()

    def undo_remove(self):
        if self.removed:
            self.removed.pop()
            self.apply_edits()

    def restore_all(self):
        if self.removed:
            self.removed = []
            self.apply_edits()

    def save_structure(self):
        if self.st is None:
            return
        types = ([("Hexagon map + remove lines", "*.hex")] if self.src_text is not None else []) + \
                [("xyz with period", "*.xyz"), ("ChemDraw", "*.cdxml")]
        p = filedialog.asksaveasfilename(filetypes=types, defaultextension=types[0][1][1:])
        if not p:
            return
        try:
            if p.lower().endswith(".hex"):
                if self.src_text is None:
                    raise ValueError("a .hex file needs a hexagon-map structure; use .xyz or .cdxml")
                lines = [self.src_text.rstrip(), "# atoms removed in the GUI (sp3 / vacancies)"]
                lines += [f"remove {x:.3f} {y:.3f}" for x, y in self.removed]
                open(p, "w").write("\n".join(lines) + "\n")
            elif p.lower().endswith(".cdxml"):
                tb.write_cdxml(self.st, p)
            else:
                tb.write_xyz(self.st, p)
        except Exception as e:
            messagebox.showerror("Save", str(e))
            return
        self.status.set(f"saved {p}")

    def set_period(self):
        if self.st is None:
            return
        try:
            self.st_orig.T = np.array([float(self.Tx.get()), float(self.Ty.get())])
            self.st_orig.periodic = True
            self.st_orig.fold(cut_in_gap=True)
        except Exception as e:
            messagebox.showerror("Period", str(e))
            return
        self.apply_edits()

    # ------------------------------------------------------------------ bands
    def calc_bands(self):
        if self.st is None:
            return
        try:
            self.H = tb.hoppings(self.st, self.t_values())
            ks, E = tb.bands(self.H, int(self.nk.get()))
        except Exception as e:
            messagebox.showerror("Bands", str(e))
            return
        EF, nocc = tb.fermi(E, self.st.n_electrons)
        self.bandres = (ks, E, EF, nocc)
        gap = E[:, nocc].min() - E[:, nocc - 1].max() if 0 < nocc < self.st.N else np.nan
        gap_txt = ("0 (odd electron count: half-filled band)" if self.st.n_electrons % 2
                   else f"{max(gap, 0):.4f} eV")
        lines = [f"E_F (half filling) = {EF:.4f} eV", f"gap = {gap_txt}",
                 f"cells coupled: R = 0..{max(self.H)}", "",
                 "band  (0 = HOMO)   Emin    Emax"]
        for n in range(max(0, nocc - 6), min(self.st.N, nocc + 6)):
            lines.append(f"{n - nocc + 1:4d}           {E[:, n].min():7.3f} {E[:, n].max():7.3f}")
        self.iso = tb.isolated_windows(E, nocc)
        lines += ["", "isolated band groups around E_F", "(a model needs one site per band):"]
        lines += [f"  bands {lo}..{hi}: {hi - lo + 1} bands, gap {g:.2f} eV" for lo, hi, g in self.iso[:6]]
        self.win_cb["values"] = [WIN_CENTRED, WIN_AUTO] + [
            f"{lo} {hi}   ({hi - lo + 1} bands, isolated by {g:.2f} eV)" for lo, hi, g in self.iso[:8]]
        self.out1.delete("1.0", "end")
        self.out1.insert("1.0", "\n".join(lines))
        self.status.set(f"{self.st.name}: bands computed. Click an atom to see its index.")
        self.dosres = None
        if self.do_dos.get():
            self.calc_dos()
        else:
            self.draw1()

    def calc_dos(self):
        """Eigenstates of the bands for the DOS, LDOS and maps (once per Hamiltonian); the
        spectra themselves are recomputed for the energy window shown at every redraw."""
        if self.st is None or self.bandres is None:
            return
        if self.dosres is None or self.dosres.get("src") is not self.H:
            try:
                self.config(cursor="watch")
                self.update()
                c = tb.bloch_cache(self.H)
                c["src"] = self.H
            except Exception as e:
                self.config(cursor="")
                messagebox.showerror("DOS", str(e))
                return
            self.config(cursor="")
            self.dosres, self._spec1, self._band1 = c, None, None
        self.draw1()

    def _zoom1(self):
        if self.st is None or self.bandres is None:
            return
        self._win1 = self._new_window(self._win1, self.ax_b1.get_ylim(), self.ax_p1.get_xlim())
        self.draw1(keep_view=True)

    def _band1_choice(self, Em):
        """(band index, k or None) shown by the band map modes: the clicked band or, before any
        click, the band closest to the map energy."""
        c = self.dosres
        EF = self.bandres[2]
        if self._band1 is not None and self._band1[0] < c["N"]:
            return self._band1
        n = tb.cache_nearest_band(c, Em, shift=EF)
        k = c["ks"][int(np.argmin(np.abs(c["E"][:, n] - EF - Em)))] if c["periodic"] else None
        return n, k

    def draw_structure(self, ax, highlight=None):
        st = self.st
        ax.clear()
        nn = tb.hoppings(st, t=(1.0,))
        n = max(1, int(self.ncells.get() or 1))
        cells = range(-(n // 2), n - n // 2)
        for c in cells:
            a = 1.0 if c == 0 else 0.25
            off = c * st.T
            for R, M in nn.items():
                for i, j in zip(*np.nonzero(np.triu(M) if R == 0 else M)):
                    p, q = st.xy[i] + off, st.xy[j] + off + R * st.T
                    col = "k" if (c == 0 and R == 0) else ("r" if c == 0 else "0.5")
                    ax.plot([p[0], q[0]], [p[1], q[1]], "-", color=col, lw=1.2, alpha=a, zorder=1)
            cols = ["k" if e == "C" else "tab:blue" for e in st.el]
            ax.scatter(*(st.xy + off).T, s=10, c=cols, alpha=a, zorder=2)
        for c in cells:
            for p in getattr(self, "removed", []):
                ax.scatter(*(p + c * st.T), s=70, facecolor="none", edgecolor="tab:blue",
                           linewidth=1.8, alpha=1.0 if c == 0 else 0.35, zorder=5)
        if self.show_idx.get():
            for i, (x, y) in enumerate(st.xy):
                ax.text(x, y, str(i), fontsize=6, color="tab:purple", zorder=5)
        ax.set_aspect("equal")
        ax.set_xlabel("x (A)")
        ax.set_ylabel("y (A)")

    def draw1(self, keep_view=False):
        if self.st is None:
            return
        st = self.st
        keep = bool(keep_view) and self._win1 is not None
        saved = self._get_lims([a for a in (self.ax_s1, self.ax_b1, self.ax_m1) if a.has_data()]) if keep else {}
        if not keep:
            self._win1 = (float(self.emin.get()), float(self.emax.get()))
            self.can1.toolbar.update()          # 'home' of the toolbar = this view
        W = tuple(self._win1)
        self.draw_structure(self.ax_s1)
        self.ax_s1.set_title(f"{st.name}: unit cell (black), bonds to next cell (red)", fontsize=9)
        for n, i in enumerate(self.sel_atoms):
            self.ax_s1.scatter(*st.xy[i], s=90, color=COLORS[n % len(COLORS)], edgecolor="k", zorder=5)
        for ax in (self.ax_b1, self.ax_d1, self.ax_m1, self.ax_p1):
            ax.clear()
        try:
            Em = float(self.emap.get())
        except ValueError:
            Em = 0.0
        c = self.dosres
        mode = self.map1_mode.get()
        band = None
        if c is not None and self.bandres and mode in (MAP_BAND, MAP_BANDK):
            band = self._band1_choice(Em)
        ax = self.ax_b1
        if self.bandres:
            ks, E, EF, nocc = self.bandres
            ax.plot(ks / np.pi, E - EF, "k", lw=0.8)
            ax.axhline(0, color="r", lw=0.5, ls="--")
            if band is not None:
                ax.plot(ks / np.pi, E[:, band[0]] - EF, color="tab:orange", lw=2.4, zorder=3)
                if mode == MAP_BANDK and band[1] is not None:
                    kb = band[1]
                    eb = np.interp(abs(kb), np.abs(ks[ks >= 0]), E[ks >= 0, band[0]]) - EF
                    ax.plot([kb / np.pi], [eb], "o", ms=7, mfc="none", mec="tab:orange", mew=2, zorder=4)
            elif c is not None:
                ax.axhline(Em, color="tab:blue", lw=0.8)
        ax.set_ylim(*W)
        ax.set_xlabel("k (pi / |T|)")
        ax.set_ylabel("E - E_F (eV)")
        if c is not None and self.bandres:
            EF, nocc = self.bandres[2], self.bandres[3]
            sig, auto = self._sigma(self.broad, W)
            key = (id(c), W, sig, tuple(self.sel_atoms))
            if self._spec1 is None or self._spec1[0] != key:
                x = self._egrid(W, sig)
                dos, L = tb.cache_spectra(c, x, sig, self.sel_atoms, shift=EF)
                self._spec1 = (key, x, dos, L)
            _, x, dos, L = self._spec1
            stxt = self._sigma_text(sig, auto)
            ax = self.ax_d1
            ax.fill_betweenx(x, 0, dos, color="0.8")
            ax.plot(dos, x, "k", lw=0.8)
            ax.axhline(0, color="r", lw=0.5, ls="--")
            ax.axhline(Em, color="tab:blue", lw=0.8)
            ax.set_xlim(0, max(dos.max(), 1e-9) * 1.05)
            ax.set_xticks([])
            ax.set_xlabel("DOS")
            ax.tick_params(labelleft=False)
            ax.set_ylim(*W)

            # tunnelling map
            ax = self.ax_m1
            n_ = max(1, int(self.ncells.get() or 1))
            cells = list(range(-(n_ // 2), n_ - n_ // 2))
            w = float(self.tipw.get() or 0)
            if band is not None:
                nb, kb = band
                v = tb.cache_band_map(c, nb, kb if mode == MAP_BANDK else None)
                what = (f"band {nb - nocc + 1:+d} (0 = HOMO band) " +
                        (f"at k = {abs(kb) / np.pi:.3f} pi/|T|" if mode == MAP_BANDK and kb is not None
                         else "averaged over k") + f": |psi|^2, tip {w:g} A")
            else:
                v = tb.cache_map(c, Em, sig, shift=EF)
                what = f"simulated dI/dV map at E - E_F = {Em:+.4f} eV (broadening {stxt}, tip {w:g} A)"
            self._draw_map(ax, st, v, cells, w, unit="" if band is not None else "states/eV")
            for k_, i in enumerate(self.sel_atoms):
                ax.scatter(*st.xy[i], s=60, facecolor="none", edgecolor=COLORS[k_ % len(COLORS)],
                           linewidth=2, zorder=5)
            for cc in cells:
                for p in self.removed:
                    ax.scatter(*(p + cc * st.T), s=40, marker="x", color="tab:cyan", linewidth=1.5, zorder=5)
            ax.set_xlabel("x (A)")
            ax.set_ylabel("y (A)")
            ax.set_title(what, fontsize=9)

            # point spectra
            ax = self.ax_p1
            ax.plot(x, dos / st.N, color="0.6", lw=1, ls="--", label="average per atom")
            for k_, i in enumerate(self.sel_atoms):
                ax.plot(x, L[k_], color=COLORS[k_ % len(COLORS)], lw=1.2, label=f"atom {i}")
            ax.axvline(0, color="r", lw=0.5, ls="--")
            ax.axvline(Em, color="tab:blue", lw=0.8)
            ax.set_xlim(*W)
            ax.set_xlabel("E - E_F (eV)  (sample bias)")
            ax.set_ylabel("LDOS (states / eV)")
            ax.set_title(f"dI/dV spectra, broadening {stxt} (click atoms on the structure or map)", fontsize=9)
            ax.legend(fontsize=7, loc="upper right")
            for ax in (self.ax_m1, self.ax_p1, self.ax_d1):
                ax.set_axis_on()
        else:
            self.ax_m1.set_axis_off()
            self.ax_p1.set_axis_off()
            self.ax_d1.set_axis_off()
            self.ax_p1.set_xlim(*W)
        self.ax_d1.set_xticks([])
        self._set_lims(saved, only=(self.ax_s1, self.ax_m1))
        self._set_lims(saved, y=False, only=(self.ax_b1,))
        self.ax_b1.set_ylim(*W)
        self.fig1.tight_layout()
        self.can1.draw_idle()
        self._views["tab1"] = (*self.ax_b1.get_ylim(), *self.ax_p1.get_xlim())

    def nearest_atom(self, x, y):
        d = np.linalg.norm(self.st.xy - [x, y], axis=1)
        # allow clicking on the periodic copies
        for c in (-2, -1, 1, 2):
            d = np.minimum(d, np.linalg.norm(self.st.xy + c * self.st.T - [x, y], axis=1))
        i = int(np.argmin(d))
        return i if d[i] < 0.7 else None

    def on_click1(self, ev):
        if self.st is None or ev.xdata is None:
            return
        if ev.inaxes in (self.ax_s1, self.ax_m1):
            i = self.nearest_atom(ev.xdata, ev.ydata)
            if i is None:
                return
            if self.click_mode.get() == "remove":
                if self.st.N <= 1:
                    return
                self.removed.append(self.st.xy[i].copy())
                self.status.set(f"removed atom at ({self.st.xy[i][0]:.3f}, {self.st.xy[i][1]:.3f}) "
                                f"in every cell; recomputing...")
                self.apply_edits()
                return
            x, y = self.st.xy[i]
            self.status.set(f"atom {i}   {self.st.el[i]}   x = {x:.3f}  y = {y:.3f} A")
            if i in self.sel_atoms:
                self.sel_atoms.remove(i)
            else:
                self.sel_atoms = (self.sel_atoms + [i])[-8:]
            self.draw1(keep_view=True)
        elif ev.inaxes in (self.ax_b1, self.ax_d1, self.ax_p1) and self.dosres and self.bandres:
            ks, E, EF, nocc = self.bandres
            Ec = ev.xdata if ev.inaxes is self.ax_p1 else ev.ydata
            if ev.inaxes is self.ax_b1:
                # the band closest (on screen) to the click at the clicked k
                ik = int(np.argmin(np.abs(ks / np.pi - ev.xdata)))
                n = int(np.argmin(np.abs(E[ik] - EF - ev.ydata)))
                k = float(np.clip(ev.xdata, -1, 1) * np.pi)
                self._band1 = (n, k)
                Eb = E[ik, n] - EF
                if self.map1_mode.get() != MAP_LDOS:
                    Ec = Eb
                self.status.set(f"band {n - nocc + 1:+d} (0 = HOMO band) at k = {k / np.pi:+.3f} pi/|T|: "
                                f"E - E_F = {Eb:+.4f} eV")
            else:
                n = tb.cache_nearest_band(self.dosres, Ec, shift=EF)
                c = self.dosres
                k = c["ks"][int(np.argmin(np.abs(c["E"][:, n] - EF - Ec)))] if c["periodic"] else None
                self._band1 = (n, k)
            self.emap.set(f"{Ec:.4f}")
            self.draw1(keep_view=True)

    def export_dos(self):
        if not self.dosres or not self.bandres:
            return
        p = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV", "*.csv")])
        if p:
            W = self._win1 or (float(self.emin.get()), float(self.emax.get()))
            sig, auto = self._sigma(self.broad, W)
            x = self._egrid(W, sig)
            dos, L = tb.cache_spectra(self.dosres, x, sig, range(self.st.N), shift=self.bandres[2])
            np.savetxt(p, np.c_[x, dos, L.T], delimiter=",",
                       header=f"E - E_F (eV), DOS per cell (states/eV), then LDOS of atom 0, 1, 2, ...; "
                              f"Gaussian broadening {1e3 * sig:.4g} meV, window shown in the plot")
            self.status.set(f"saved {p}")

    def export_bands(self):
        if not self.bandres:
            return
        p = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV", "*.csv")])
        if p:
            ks, E, EF, nocc = self.bandres
            np.savetxt(p, np.c_[ks, E], delimiter=",",
                       header=f"k (1/period), bands (eV); E_F = {EF:.6f}; HOMO = column {nocc}")
            self.status.set(f"saved {p}")

    def export_cdxml(self):
        if self.st is None:
            return
        p = filedialog.asksaveasfilename(defaultextension=".cdxml", filetypes=[("ChemDraw", "*.cdxml")])
        if p:
            tb.write_cdxml(self.st, p)
            self.status.set(f"saved {p}")

    # ------------------------------------------------------------------ effective model
    def load_eff(self):
        p = filedialog.askopenfilename(filetypes=[("Effective model", "*.eff *.txt"), ("All", "*.*")],
                                       initialdir=EXAMPLES)
        if p:
            self.editor.delete("1.0", "end")
            self.editor.insert("1.0", open(p).read())
            self.show_sites()

    def save_eff(self):
        p = filedialog.asksaveasfilename(defaultextension=".eff", filetypes=[("Effective model", "*.eff")])
        if p:
            open(p, "w").write(self.editor.get("1.0", "end"))
            self.status.set(f"saved {p}")

    def eff_help(self):
        doc = tb.__doc__
        messagebox.showinfo("Effective model syntax", doc[doc.index("Effective model file"):])

    def parse_model(self):
        if self.st is None:
            messagebox.showerror("Model", "load a structure first")
            return None
        try:
            return tb.parse_eff(self.editor.get("1.0", "end"), self.st)
        except Exception as e:
            messagebox.showerror("Model", str(e))
            return None

    def site_xy(self, m, n):
        atoms = m.site_atom[n]
        return self.st.xy[atoms].mean(axis=0) if atoms else None

    def draw_model(self, m):
        ax = self.ax_s2
        self.draw_structure(ax)
        self.zoom_cell(ax)
        T = self.st.T
        # effective hoppings: one colour per parameter
        pcol = {p: COLORS[k % len(COLORS)] for k, p in enumerate(m.params)}
        seen = set()
        for i, j, R, p, s in m.terms:
            a, b = self.site_xy(m, i), self.site_xy(m, j)
            if a is None or b is None or (i == j and R == 0):
                continue
            b = b + R * T
            ax.plot([a[0], b[0]], [a[1], b[1]], "-", color=pcol[p], lw=3, alpha=0.7, zorder=3,
                    label=p if p not in seen else None)
            seen.add(p)
        for n, name in enumerate(m.sites):
            atoms = m.site_atom[n]
            if not atoms:
                continue
            wt = m.site_w[n] or [1] * len(atoms)
            for a, w in zip(atoms, wt):
                ax.scatter(*self.st.xy[a], s=160, color="tab:green" if w > 0 else "tab:orange",
                           edgecolor="k", zorder=4)
            c = self.site_xy(m, n)
            ax.annotate(name, c, xytext=(4, 6), textcoords="offset points", fontsize=9,
                        weight="bold", color="darkgreen", zorder=6)
        if seen:
            ax.legend(loc="upper right", fontsize=7, framealpha=0.8)
        ax.set_title("effective sites (green: +, orange: - weight) and effective hoppings")

    def show_sites(self):
        m = self.parse_model()
        if m is None:
            return
        self.model = m
        self.sites_from_editor(quiet=True)
        self.draw_model(m)
        self.ax_b2.clear()
        self.fig2.tight_layout()
        self.can2.draw_idle()
        self.status.set(f"{len(m.sites)} sites, {len(m.params)} parameters: {', '.join(m.params)}")

    def run_fit(self):
        m = self.parse_model()
        if m is None:
            return
        if self.H is None:
            self.calc_bands()
        self.config(cursor="watch")
        self.update()
        try:
            r = tb.fit(self.H, self.st, m, use_projection=self.use_proj.get())
        except Exception as e:
            self.config(cursor="")
            messagebox.showerror("Fit", str(e))
            return
        self.config(cursor="")
        self.model, self.fitres = m, r
        EF = r["EF"]
        L = [f"sites: {len(m.sites)}   bands fitted: " +
             ("auto (weight on sites)" if m.window == "auto" else f"{m.window[0]}..{m.window[1]} (0 = HOMO)"),
             f"rms error  = {1e3 * r['rms']:.2f} meV   (start: {1e3 * r['rms0']:.2f})",
             f"max error  = {1e3 * r['maxerr']:.2f} meV",
             f"E_F        = {EF:.4f} eV"]
        if m.window != "auto":
            L.append(f"isolation  : gap below {r['gaps'][0]:.3f} eV, above {r['gaps'][1]:.3f} eV")
        if r["projected"] is not None:
            L.append(f"projection : min overlap {r['overlap']:.3f}" +
                     ("   (LOW: sites miss some band states)" if r["overlap"] < 0.02 else ""))
        L += ["", f"{'param':10s} {'start':>10s} {'from':>11s} {'fitted':>10s}"]
        for p in m.params:
            L.append(f"{p:10s} {r['init'][p]:10.4f} {r['source'][p]:>11s} {r['params'][p]:10.4f}"
                     + ("  fixed" if p in m.fixed else ""))
        if r["missing"]:
            L += ["", "projected couplings NOT in the model (> 10 meV):"]
            for v, a, b, R, h in r["missing"][:12]:
                L.append(f"  hop {a} {b} {R}   {h:+.4f} eV")
        try:
            diag, marks = tb.band_diagnostics(self.H, self.st, m, r)
        except Exception as e:                      # diagnostics must never block a fit
            diag, marks = [f"(band diagnostics failed: {e})"], []
        L += ["", "band crossings and band selection (the fit compares the sorted energies at every k,",
              "so the order of the bands does not matter; see the notes):"] + ["  " + x for x in diag]
        self.out2.delete("1.0", "end")
        self.out2.insert("1.0", "\n".join(L))

        self.draw_model(m)
        ax = self.ax_b2
        ax.clear()
        ks = r["ks"] / np.pi
        ax.plot(ks, r["E_tb"] - EF, color="0.8", lw=0.8)
        ax.plot(ks, r["E_ref"] - EF, "k.", ms=2)
        ax.plot(ks, r["E_eff"] - EF, "r-", lw=1.2)
        style = {"cross": ("o", "tab:green", "protected crossing"), "avoided": ("o", "tab:orange", "avoided crossing"),
                 "close": ("o", "0.5", "close approach"), "zone": ("s", "0.5", "degenerate / closest at 0, pi"),
                 "unknown": ("o", "0.5", "near-degeneracy")}
        used = set()
        for mk in marks:
            if mk[0] == "window":
                for sgn in (1, -1):
                    ax.axvspan(sgn * mk[1] / np.pi, sgn * mk[2] / np.pi, color="tab:red", alpha=0.08, lw=0)
                continue
            sym_, col, lab = style[mk[0]]
            ax.scatter([mk[1] / np.pi, -mk[1] / np.pi], [mk[2], mk[2]], marker=sym_, s=60, facecolors="none",
                       edgecolors=col, lw=1.6, zorder=5, label=None if lab in used else lab)
            used.add(lab)
        if used:
            ax.legend(fontsize=7, loc="upper right")
        lo, hi = (r["E_ref"] - EF).min(), (r["E_ref"] - EF).max()
        pad = 0.25 * (hi - lo) + 0.1
        ax.set_ylim(lo - pad, hi + pad)
        ax.set_xlabel("k (pi / |T|)")
        ax.set_ylabel("E - E_F (eV)")
        ax.set_title("TB (grey), fitted (black), model (red)", fontsize=8)
        self.fig2.tight_layout()
        self.can2.draw_idle()
        self.status.set(f"fit done: rms {1e3 * r['rms']:.2f} meV")

    # ------------------------------------------------------------------ site picker
    def on_click2(self, ev):
        if self.st is None or ev.inaxes is not self.ax_s2 or ev.xdata is None or ev.button not in (1, 3):
            return
        i = self.nearest_atom(ev.xdata, ev.ydata)
        if i is None:
            return
        owner = next((n for n, p in enumerate(self.picks) if i in p["atoms"]), None)
        if owner is not None and owner != self.cur:
            self.cur = owner                      # select that site
        elif owner == self.cur:
            p = self.picks[self.cur]
            k = p["atoms"].index(i)
            del p["atoms"][k], p["w"][k]          # remove the atom
        else:
            if self.cur < 0:
                self.new_site(redraw=False)
            p = self.picks[self.cur]
            p["atoms"].append(i)
            p["w"].append(1.0 if ev.button == 1 else -1.0)
        self.update_picks_label()
        self.draw_picks()

    def new_site(self, redraw=True):
        names = {p["name"] for p in self.picks}
        k = len(self.picks) + 1
        while f"S{k}" in names:
            k += 1
        self.picks.append(dict(name=f"S{k}", atoms=[], w=[]))
        self.cur = len(self.picks) - 1
        self.update_picks_label()
        if redraw:
            self.draw_picks()

    def delete_site(self):
        if 0 <= self.cur < len(self.picks):
            del self.picks[self.cur]
            self.cur = len(self.picks) - 1
            self.update_picks_label()
            self.draw_picks()

    def clear_sites(self):
        self.picks, self.cur = [], -1
        self.update_picks_label()
        self.draw_picks()

    def rename_site(self):
        new = self.name_var.get().strip().replace(" ", "_")
        if 0 <= self.cur < len(self.picks) and new:
            if any(p["name"] == new for n, p in enumerate(self.picks) if n != self.cur):
                messagebox.showerror("Rename", f"there is already a site called {new}")
                return
            self.picks[self.cur]["name"] = new
            self.update_picks_label()
            self.draw_picks()

    def sites_from_editor(self, quiet=False):
        """Take the sites of the model in the editor as the picked sites."""
        try:
            m = tb.parse_eff(self.editor.get("1.0", "end"), self.st)
        except Exception as e:
            if not quiet:
                messagebox.showerror("Model", str(e))
            return
        self.picks = [dict(name=n, atoms=list(a or []), w=list(w or [1.0] * len(a or [])))
                      for n, a, w in zip(m.sites, m.site_atom, m.site_w)]
        self.cur = len(self.picks) - 1
        self.update_picks_label()
        if not quiet:
            self.draw_picks()

    def update_picks_label(self):
        if not self.picks:
            self.picks_var.set("no sites yet: click atoms in the plot")
            self.name_var.set("")
            return
        L = []
        for n, p in enumerate(self.picks):
            at = " ".join(f"{a}{'(-)' if w < 0 else ''}" for a, w in zip(p["atoms"], p["w"]))
            L.append(f"{'>' if n == self.cur else ' '} {p['name']:6s} atoms {at or '-'}")
        self.picks_var.set("\n".join(L))
        self.name_var.set(self.picks[self.cur]["name"] if self.cur >= 0 else "")

    def window_choice(self):
        v = self.win_var.get()
        if v.startswith("centred"):
            return None
        if v.startswith("auto"):
            return "auto"
        w = v.split()
        return (int(w[0]), int(w[1]))

    def zoom_cell(self, ax):
        """Show one unit cell plus a margin (atoms big enough to click)."""
        x, y = self.st.xy[:, 0], self.st.xy[:, 1]
        mx = 0.15 * max(np.ptp(x), 3.0) + 1.5
        ax.set_xlim(x.min() - mx, x.max() + mx)
        ax.set_ylim(y.min() - 2.0, y.max() + 2.0)

    def draw_picks(self):
        if self.st is None:
            return
        ax = self.ax_s2
        self.draw_structure(ax)
        self.zoom_cell(ax)
        for n, p in enumerate(self.picks):
            col = COLORS[n % len(COLORS)]
            for a, w in zip(p["atoms"], p["w"]):
                ax.scatter(*self.st.xy[a], s=170, marker="o" if w > 0 else "s", color=col,
                           edgecolor="k", linewidth=2.5 if n == self.cur else 0.8, zorder=4)
            if p["atoms"]:
                c = self.st.xy[p["atoms"]].mean(axis=0)
                ax.annotate(p["name"], c, xytext=(5, 7), textcoords="offset points", fontsize=9,
                            weight="bold", color=col, zorder=6)
        ax.set_title("picked sites (circle: +, square: -; thick edge: current site)", fontsize=9)
        # bands, with the ones the model will fit in black
        ax = self.ax_b2
        ax.clear()
        if self.bandres:
            ks, E, EF, nocc = self.bandres
            ax.plot(ks / np.pi, E - EF, color="0.75", lw=0.8)
            n = len(self.picks)
            win = self.window_choice()
            if win is None and n:
                win = tb.default_window(n)
            if win not in (None, "auto"):
                sel = slice(nocc - 1 + win[0], nocc + win[1])
                ax.plot(ks / np.pi, E[:, sel] - EF, "k", lw=1)
                lo, hi = (E[:, sel] - EF).min(), (E[:, sel] - EF).max()
                pad = 0.4 * (hi - lo) + 0.2
                ax.set_ylim(lo - pad, hi + pad)
                ax.set_title(f"bands to fit: {win[0]}..{win[1]} ({win[1] - win[0] + 1})", fontsize=8)
            else:
                ax.set_ylim(-1.5, 1.5)
            ax.set_xlabel("k (pi / |T|)")
            ax.set_ylabel("E - E_F (eV)")
        self.fig2.tight_layout()
        self.can2.draw_idle()

    def build_from_sites(self):
        if self.st is None:
            return
        picks = [p for p in self.picks if p["atoms"]]
        if not picks:
            messagebox.showerror("Build", "pick at least one site: click atoms in the plot")
            return
        if self.H is None:
            self.calc_bands()
        try:
            sites = [(p["name"], p["atoms"], p["w"] if any(w < 0 for w in p["w"]) else None)
                     for p in picks]
            self.config(cursor="watch")
            self.update()
            text, info = tb.build_model(self.H, self.st, sites, window=self.window_choice(),
                                        thr=float(self.thr.get()) / 1e3, tol=float(self.tol.get()) / 1e3,
                                        Rmax=int(self.rmax.get()), optimise=self.opt_w.get())
        except Exception as e:
            self.config(cursor="")
            messagebox.showerror("Build", str(e))
            return
        self.config(cursor="")
        for p in self.picks:                      # show the optimised weights
            if p["name"] in info["weights"]:
                p["w"] = list(info["weights"][p["name"]])
        self.update_picks_label()
        self.editor.delete("1.0", "end")
        self.editor.insert("1.0", text)
        self.run_fit()
        self.status.set(f"model built: {info['n_terms']} couplings, {info['n_params']} parameters, "
                        f"overlap {info['overlap']:.3f}; fitted (rms in the panel)")


    # ================================================================== tab 3: finite chain
    def _build_tab3(self):
        self.fbase = self.fst = self.fres = None
        self.f_removed, self.f_sel = [], []
        self.f_terms, self.f_term_i, self.f_mf = [], -1, None
        left = self._scrollable(self.tab3, width=360)

        box = ttk.LabelFrame(left, text="Finite chain from the periodic structure (tab 1)", padding=6)
        box.pack(fill="x")
        self.f_n, self.f_off = self._entry2(box, "units", "4", "cut shift (A)", "0.0")
        self.f_extra = self._entry(box, "extra length (A)", "0.0")
        self.f_prune = tk.BooleanVar(value=True)
        ttk.Checkbutton(box, text="remove atoms left with one neighbour at the ends",
                        variable=self.f_prune).pack(anchor="w")
        ttk.Label(box, text="the chain uses the structure of tab 1 (with its sp3 edits)\n"
                            "and its hoppings t1, t2, t3; the cut shift moves where the\n"
                            "ends are cut (different terminations)", foreground="gray").pack(anchor="w")
        ttk.Button(box, text="Build finite chain and calculate", command=self.f_build).pack(fill="x", pady=(4, 0))
        ttk.Button(box, text="Symmetric ends: next option", command=self.f_next_symmetric).pack(fill="x", pady=(2, 0))
        ttk.Label(box, text="(cuts where the two ends are equivalent, e.g. a 7-AGNR ending\n"
                            "on three-ring columns at both sides; each click the next one.\n"
                            "Units linked by single bonds (olympicene): whole units only,\n"
                            "so a dimer cell also gives odd chains, e.g. 3 or 5 units)",
                  foreground="gray").pack(anchor="w")
        ttk.Button(box, text="...or load a finite molecule file (no period)",
                   command=self.f_load).pack(fill="x", pady=(2, 0))
        self.f_info = tk.StringVar(value="-")
        ttk.Label(box, textvariable=self.f_info, justify="left", wraplength=300).pack(anchor="w", pady=4)

        box = ttk.LabelFrame(left, text="Modify (sp3 / vacancies): only the clicked atom", padding=6)
        box.pack(fill="x", pady=(6, 0))
        row = ttk.Frame(box)
        row.pack(fill="x")
        ttk.Label(row, text="click on an atom:").pack(side="left")
        self.f_mode = tk.StringVar(value="spectrum")
        ttk.Radiobutton(row, text="dI/dV", value="spectrum", variable=self.f_mode).pack(side="left")
        ttk.Radiobutton(row, text="remove (sp3)", value="remove", variable=self.f_mode).pack(side="left")
        row = ttk.Frame(box)
        row.pack(fill="x", pady=(2, 0))
        ttk.Button(row, text="Undo", width=6, command=self.f_undo).pack(side="left")
        ttk.Button(row, text="Restore all", command=self.f_restore).pack(side="left")
        ttk.Button(row, text="Save structure as...", command=self.f_save).pack(side="left")

        box = ttk.LabelFrame(left, text="Levels, DOS and tunnelling LDOS (dI/dV)", padding=6)
        box.pack(fill="x", pady=(6, 0))
        self.f_emin, self.f_emax = self._entry2(box, "E-E_F  min", "-2", "max (eV)", "2")
        self.f_broad = self._entry(box, "broadening (meV)", "auto")
        self.f_emap = self._entry(box, "map energy E - E_F", "0.0")
        self.f_tip = self._entry(box, "tip resolution (A)", "0.9")
        row = ttk.Frame(box)
        row.pack(fill="x")
        ttk.Label(row, text="map:").pack(side="left")
        self.f_map = tk.StringVar(value=MAP_LDOS)
        cb = ttk.Combobox(row, textvariable=self.f_map, state="readonly", width=34,
                          values=[MAP_LDOS, MAP_LEVEL, MAP_SPIN, MAP_UP, MAP_DN, MAP_UPDN])
        cb.pack(side="left", padx=2)
        cb.bind("<<ComboboxSelected>>", lambda e: self.f_draw(keep_view=True))
        ttk.Label(box, text="click the levels/DOS: map energy (and the level)\nclick an atom: its dI/dV spectrum\n"
                            "zoom/pan with the toolbar: DOS, map and spectra are\nrecomputed for the window shown "
                            "(broadening\n'auto' = window / 250, 0.1 to 20 meV). Spin maps:\nmean-field states "
                            "(choose one in 'show' below)",
                  foreground="gray").pack(anchor="w")
        row = ttk.Frame(box)
        row.pack(fill="x", pady=(4, 0))
        ttk.Button(row, text="Update", command=self.f_calc).pack(side="left", fill="x", expand=True)
        ttk.Button(row, text="Export (CSV)...", command=self.f_export).pack(side="left")
        box = ttk.LabelFrame(left, text="Mean-field Hubbard of the finite system", padding=6)
        box.pack(fill="x", pady=(6, 0))
        ttk.Label(box, text="H_s = H_TB + U sum_x <n_x,-s> n_x,s  (all atoms)", foreground="#1f4e79",
                  font=("Courier", 9)).pack(anchor="w")
        self.f_mfU, self.f_mfT = self._entry2(box, "U (eV)", "3.0", "kT (eV)", "0.005")
        self.f_mfsz = self._entry(box, "S_z to try", "auto")
        self.f_btn_mf = ttk.Button(box, text="Run mean field (no sites to choose)", command=self.f_run_mf)
        self.f_btn_mf.pack(fill="x", pady=(2, 2))
        row = ttk.Frame(box)
        row.pack(fill="x")
        ttk.Label(row, text="show").pack(side="left")
        self.f_show = tk.StringVar(value="tight binding (U = 0)")
        self.f_showcb = ttk.Combobox(row, textvariable=self.f_show, width=34, state="readonly",
                                     values=["tight binding (U = 0)"])
        self.f_showcb.pack(side="left", padx=4)
        self.f_showcb.bind("<<ComboboxSelected>>", lambda e: self.f_draw(keep_view=True))
        ttk.Label(box, text="a mean-field state: levels up/down, spin DOS, spin density map,\n"
                            "spin-resolved dI/dV spectra (up solid, down dashed).\n"
                            "The output also gives the singlet-triplet gap from the mean\n"
                            "field (spin projection; Heisenberg mapping for >2 radicals).\n"
                            "The same calculation is in tab 5 ('finite chain' mode).",
                  foreground="gray").pack(anchor="w")
        self.f_out = tk.Text(left, width=40, height=14, font=("Courier", 9))
        self.f_out.pack(fill="both", expand=True, pady=(6, 0))

        right = ttk.Frame(self.tab3)
        right.pack(side="left", fill="both", expand=True)
        self.fig3 = Figure(figsize=(10, 8))
        gs = self.fig3.add_gridspec(2, 3, width_ratios=[2.6, 0.7, 0.5], height_ratios=[1, 1])
        self.ax_s3 = self.fig3.add_subplot(gs[0, 0])
        self.ax_l3 = self.fig3.add_subplot(gs[0, 1])
        self.ax_d3 = self.fig3.add_subplot(gs[0, 2], sharey=self.ax_l3)
        self.ax_m3 = self.fig3.add_subplot(gs[1, 0])
        self.ax_p3 = self.fig3.add_subplot(gs[1, 1:])
        self.can3 = FigureCanvasTkAgg(self.fig3, right)
        NavigationToolbar2Tk(self.can3, right).update()
        self.can3.get_tk_widget().pack(fill="both", expand=True)
        self._win3, self._lev3, self._spec3, self._f_cache = None, None, None, None
        self._on_clicks(self.can3, self.f_click)
        self._watch_view(self.can3, "tab3", lambda: (*self.ax_l3.get_ylim(), *self.ax_p3.get_xlim()),
                         self._zoom3)

    def f_build(self):
        if self.st is None or not getattr(self.st, "periodic", True):
            messagebox.showerror("Finite chain", "load a periodic structure in tab 1 first")
            return
        try:
            n = int(self.f_n.get())
            if n < 1:
                raise ValueError("the number of units must be at least 1")
            self.fbase = tb.finite_chain(self.st, n, float(self.f_off.get() or 0), self.f_prune.get(),
                                         float(self.f_extra.get() or 0))
        except Exception as e:
            messagebox.showerror("Finite chain", str(e))
            return
        self.f_removed, self.f_sel = [], []
        self.f_apply()

    def f_load(self):
        p = filedialog.askopenfilename(filetypes=STRUCT_TYPES, initialdir=EXAMPLES)
        if not p:
            return
        try:
            self.fbase = tb.load_structure(p, finite=True)
        except Exception as e:
            messagebox.showerror("Finite molecule", str(e))
            return
        self.fbase.n_pruned = 0
        if getattr(self.fbase, "n_sp3", 0):
            self.status.set(f"{self.fbase.n_sp3} sp3 carbons in the drawing left out of the pi system")
        self.f_removed, self.f_sel = [], []
        self.f_apply()

    def f_apply(self):
        self.fst = tb.remove_atoms(self.fbase, self.f_removed)
        self.fst.name = self.fbase.name
        self.f_calc()

    def f_calc(self):
        if self.fst is None:
            return
        fs = self.fst
        try:
            self.config(cursor="watch")
            self.update()
            H = tb.hoppings(fs, self.t_values())
            c = tb.bloch_cache({0: H[0]})
            E = c["E"][0]
            EF, nocc = tb.fermi(E[None, :], fs.n_electrons)
        except Exception as e:
            self.config(cursor="")
            messagebox.showerror("Finite chain", str(e))
            return
        self.config(cursor="")
        self.fres = dict(c=c, E=E - EF, EF=EF, nocc=nocc)
        self._spec3, self._lev3 = None, None
        homo, lumo = E[nocc - 1] - EF, (E[nocc] - EF if nocc < fs.N else np.nan)
        bal = tb.sublattice_balance(fs)
        near = int(np.sum(np.abs(E - EF) < 0.01))
        info = [f"{fs.N} pz sites, {fs.n_electrons} pi electrons",
                f"atoms removed at the ends (one neighbour): {getattr(self.fbase, 'n_pruned', 0)}"]
        if self.f_removed:
            info.append(f"sp3 / removed by hand: {len(self.f_removed)}")
        if bal:
            d = abs(bal[0] - bal[1])
            info.append(f"sublattices A/B = {bal[0]}/{bal[1]}" + (f": {d} zero mode(s) with NN hopping" if d else ""))
        info.append(f"HOMO-LUMO gap = {lumo - homo:.4f} eV" if fs.n_electrons % 2 == 0
                    else "odd electron count: singly occupied level at E_F")
        info.append(f"levels within 10 meV of E_F: {near}")
        self.f_info.set("\n".join(info))
        lines = [f"E_F = {EF:.4f} eV (midgap)", "", " level   E - E_F (eV)   occupation"]
        for k in range(max(0, nocc - 8), min(fs.N, nocc + 8)):
            occ = 2 if k < nocc - 1 or (k == nocc - 1 and fs.n_electrons % 2 == 0) else (1 if k == nocc - 1 else 0)
            lines.append(f"{k - nocc + 1:5d}   {E[k] - EF:12.4f}   {occ}")
        self.f_out.delete("1.0", "end")
        self.f_out.insert("1.0", "\n".join(lines))
        self.f_draw()

    def _chain_bonds(self, st, cells, lw=0.8, color="k", alpha=1.0, shown=None):
        """Bonds of the periodic structure drawn on the given cells, including the bonds that
        cross from one drawn cell to the next (a plain _bonds of shifted copies misses them).
        shown: optional bool array (cells 0 .. n-1, atoms) of the atoms to draw (partial cell)."""
        nn = tb.hoppings(st, t=(1.0,))
        cells = list(cells)
        seg = []
        for c in cells:
            for R, M in nn.items():
                if c + R not in cells:
                    continue
                for i, j in zip(*np.nonzero(np.triu(M) if R == 0 else M)):
                    if shown is not None and not (shown[c][i] and shown[c + R][j]):
                        continue
                    seg.append([st.xy[i] + c * st.T, st.xy[j] + (c + R) * st.T])
        return LineCollection(np.array(seg).reshape(-1, 2, 2), colors=color, linewidths=lw, alpha=alpha)

    def _bonds(self, st, lw=0.8, color="k", alpha=1.0):
        nn = tb.hoppings(st, t=(1.0,))
        i, j = np.nonzero(np.triu(nn[0]))
        return LineCollection(np.stack([st.xy[i], st.xy[j]], axis=1), colors=color, linewidths=lw, alpha=alpha)

    def f_next_symmetric(self):
        if self.st is None or not getattr(self.st, "periodic", True):
            messagebox.showerror("Finite chain", "load a periodic structure in tab 1 first")
            return
        try:
            n = int(self.f_n.get())
            key = (id(self.st), n, self.f_prune.get())
            if getattr(self, "_f_terms_key", None) != key:
                self.config(cursor="watch")
                self.update()
                self.f_terms = tb.symmetric_terminations(self.st, n, self.f_prune.get())
                self._f_terms_key, self.f_term_i = key, -1
        except Exception as e:
            messagebox.showerror("Finite chain", str(e))
            return
        finally:
            self.config(cursor="")
        if not self.f_terms:
            messagebox.showinfo("Finite chain", "no cut gives two equivalent ends for this structure")
            return
        self.f_term_i = (self.f_term_i + 1) % len(self.f_terms)
        d = self.f_terms[self.f_term_i]
        self.f_off.set(f"{d['offset']:.3f}")
        self.f_extra.set(f"{d['extra']:.3f}")
        self.f_build()
        units = f", {d['units']} units ({'odd' if d['units'] % 2 else 'even'})" if d.get("units") else ""
        self.status.set(f"symmetric ends {self.f_term_i + 1} of {len(self.f_terms)} ({d['symmetry']}): "
                        f"{d['N']} atoms{units}, cut shift {d['offset']:.3f} A, extra {d['extra']:.3f} A")

    def f_run_mf(self):
        fs = self.fst
        if fs is None:
            messagebox.showerror("Mean field", "build or load a finite system first")
            return
        try:
            U, kT = float(self.f_mfU.get()), float(self.f_mfT.get())
            txt = self.f_mfsz.get().strip().lower()
            sz = None if txt in ("", "auto") else [int(round(2 * float(x))) for x in txt.replace(",", " ").split()]
        except Exception as e:
            messagebox.showerror("Mean field", str(e))
            return
        H = tb.hoppings(fs, self.t_values())
        fs_run = fs

        def work():
            res = tm.mfh_scan({0: H[0]}, fs_run, U, 1, kT, sz)
            gap = tm.mfh_spin_gap(H[0], fs_run, res, U, kT)
            return tm.mfh_states(res, gap), gap

        def done(out):
            res, gap = out
            self.f_mf = dict(results=res, fs=fs_run, U=U, H0=H[0], gap=gap)
            labels = ["tight binding (U = 0)"] + [o["label"] for o in res]
            self.f_showcb.configure(values=labels)
            self.f_show.set(res[0]["label"])
            self.f_map.set(MAP_SPIN)
            self._f_cache = self._spec3 = None
            self.f_out.delete("1.0", "end")
            self.f_out.insert("1.0", "\n".join(self._mf_table(res, U, kT) + [""] + tm.spin_gap_report(gap)))
            self.f_draw(keep_view=True)

        self._sm_background(work, done, self.f_btn_mf)

    @staticmethod
    def _mf_table(res, U, kT, per_cell=False):
        """Table of mean-field states (energies from the lowest)."""
        E0 = res[0]["sol"]["E"]
        L = [f"mean field U = {U} eV, kT = {kT} eV:",
             f" {'state':34s}{'E-Emin (meV' + (' / cell)' if per_cell else ')'):>17s}{'|m|max':>8s}"]
        L += [f" {o['label'][:34]:34s}{1e3 * (o['sol']['E'] - E0):17.3f}{o['mmax']:8.3f}"
              + ("" if o["sol"]["converged"] else " NOT conv.") for o in res]
        return L

    def _f_mf_state(self):
        """The mean-field state chosen in 'show' for the current finite system, or None."""
        mf = self.f_mf
        if not mf or mf["fs"] is not self.fst:
            return None
        for o in mf["results"]:
            if o["label"] == self.f_show.get():
                return o
        return None

    def _f_channels(self, o):
        """Spin channels to draw: [(cache, reference energy, electrons, colour, label)].
        Tight binding: one spin-degenerate channel; a mean-field state: up and down, each with
        its own H_s = H_TB + U diag(<n_-s>) (eigenstates cached for the state shown)."""
        r = self.fres
        if o is None:
            return [(r["c"], r["EF"], r["nocc"], "k", "")]
        key = id(o)
        if self._f_cache is None or self._f_cache[0] != key:
            n, mf = o["sol"]["n"], self.f_mf
            self._f_cache = (key, [tb.bloch_cache({0: mf["H0"] + np.diag(mf["U"] * n[1 - s])}) for s in (0, 1)])
        sol, ref = o["sol"], tm.midgap(o["sol"])
        cu, cd = self._f_cache[1]
        return [(cu, ref, sol["n_up"], "tab:red", "up"), (cd, ref, sol["n_dn"], "tab:blue", "down")]

    def _zoom3(self):
        if self.fst is None or self.fres is None:
            return
        self._win3 = self._new_window(self._win3, self.ax_l3.get_ylim(), self.ax_p3.get_xlim())
        self.f_draw(keep_view=True)

    def f_draw(self, keep_view=False):
        """Finite system: levels, DOS, dI/dV (or spin density / level) map and spectra, for the
        tight binding or for the mean-field state chosen in 'show'. The spectra and the map are
        computed for the energy window shown (zoom with the toolbar)."""
        fs, r = self.fst, self.fres
        if fs is None or r is None:
            return
        o = self._f_mf_state()
        keep = bool(keep_view) and self._win3 is not None
        saved = self._get_lims([a for a in (self.ax_s3, self.ax_m3) if a.has_data()]) if keep else {}
        if not keep:
            self._win3 = (float(self.f_emin.get()), float(self.f_emax.get()))
            self.can3.toolbar.update()
        W = tuple(self._win3)
        for ax in (self.ax_s3, self.ax_l3, self.ax_d3, self.ax_m3, self.ax_p3):
            ax.clear()
        try:
            Em = float(self.f_emap.get())
        except ValueError:
            Em = 0.0
        sig, auto = self._sigma(self.f_broad, W)
        stxt = self._sigma_text(sig, auto)
        ch = self._f_channels(o)
        mf = o is not None
        sel = list(self.f_sel)
        key = (id(o), id(r["c"]), W, sig, tuple(sel))
        if self._spec3 is None or self._spec3[0] != key:
            x = self._egrid(W, sig)
            self._spec3 = (key, x, [tb.cache_spectra(c, x, sig, sel, shift=ref) for c, ref, *_ in ch])
        _, x, spec = self._spec3
        levels = [c["E"][0] - ref for c, ref, *_ in ch]
        mode = self.f_map.get()
        if not mf and mode in (MAP_SPIN, MAP_UP, MAP_DN, MAP_UPDN):
            mode = MAP_LDOS
        lev = self._lev3
        if mode == MAP_LEVEL:
            if lev is None or lev[0] >= len(ch) or lev[1] >= len(levels[lev[0]]):
                s_ = 0
                lev = (s_, int(np.argmin(np.abs(levels[s_] - Em))))
        # structure
        ax = self.ax_s3
        ax.add_collection(self._bonds(fs))
        ax.scatter(*fs.xy.T, s=6, c="k", zorder=2)
        if self.f_removed:
            ax.scatter(*np.array(self.f_removed).T, s=50, facecolor="none", edgecolor="tab:blue",
                       linewidth=1.8, zorder=3)
        for n, i in enumerate(sel):
            ax.scatter(*fs.xy[i], s=80, color=COLORS[n % len(COLORS)], edgecolor="k", zorder=4)
        ax.set_aspect("equal")
        ax.autoscale_view()
        ax.set_title(f"{fs.name}: {fs.N} atoms" + (f"; mean field: {o['label']}" if mf else
                                                     " (blue circles: removed)"), fontsize=9)
        # levels
        ax = self.ax_l3
        if mf:
            for s_, (c, ref, nel, col, lab) in enumerate(ch):
                x0 = 0.0 if s_ == 0 else 0.55
                e = levels[s_]
                occ = np.arange(len(e)) < nel
                ax.hlines(e[occ], x0, x0 + 0.45, color=col, lw=1.4)
                ax.hlines(e[~occ], x0, x0 + 0.45, color=col, lw=0.8, alpha=0.4)
            ax.set_xticks([0.22, 0.77])
            ax.set_xticklabels(["up", "down"])
            ax.set_xlabel("levels (faint: empty)")
        else:
            e, nocc = levels[0], r["nocc"]
            occ = np.arange(len(e)) < nocc
            ax.hlines(e[occ], 0, 1, color="k", lw=1)
            ax.hlines(e[~occ], 0, 1, color="0.6", lw=1)
            ax.set_xticks([])
            ax.set_xlabel("levels\n(black: occupied)")
        if mode == MAP_LEVEL:
            x0, x1 = ((0.0, 0.45) if lev[0] == 0 else (0.55, 1.0)) if mf else (0.0, 1.0)
            ax.hlines(levels[lev[0]][lev[1]], x0, x1, color="tab:orange", lw=3, zorder=3)
        else:
            ax.axhline(Em, color="tab:blue", lw=0.8)
        ax.axhline(0, color="r" if not mf else "k", lw=0.5, ls="--")
        ax.set_xlim(0, 1)
        ax.set_ylabel("E - E_F (eV)")
        ax.set_ylim(*W)
        # DOS
        ax = self.ax_d3
        if mf:
            for (dos, L), (c, ref, nel, col, lab), sign in zip(spec, ch, (1, -1)):
                ax.fill_betweenx(x, 0, sign * dos, color=col, alpha=0.5)
            m = max(max(d.max() for d, L in spec), 1e-9) * 1.05
            ax.set_xlim(-m, m)
            ax.axvline(0, color="k", lw=0.5)
            ax.set_xlabel("DOS\ndown | up")
        else:
            dos = spec[0][0]
            ax.fill_betweenx(x, 0, dos, color="0.8")
            ax.plot(dos, x, "k", lw=0.8)
            ax.set_xlim(0, max(dos.max(), 1e-9) * 1.05)
            ax.set_xlabel("DOS")
        ax.axhline(0, color="r", lw=0.5, ls="--")
        ax.axhline(Em, color="tab:blue", lw=0.8)
        ax.set_xticks([])
        ax.tick_params(labelleft=False)
        ax.set_ylim(*W)
        # map
        ax = self.ax_m3
        w = float(self.f_tip.get() or 0)
        signed = False
        if mode == MAP_SPIN:
            mz = o["sol"]["n"][0] - o["sol"]["n"][1]
            ax.add_collection(self._bonds(fs, 0.5, "0.6"))
            ax.scatter(*fs.xy.T, s=4000 * np.abs(mz) / 4 + 1, c=np.where(mz > 0, "tab:red", "tab:blue"),
                       alpha=0.8, zorder=3)
            ax.set_aspect("equal")
            ax.autoscale_view()
            title = f"spin density n_up - n_dn (red up, blue down), max |m| = {o['mmax']:.3f}"
        else:
            if mode == MAP_LEVEL:
                s_, n_ = lev
                v = ch[s_][0]["P"][0][:, n_].astype(float)
                nocc = ch[s_][2] if mf else r["nocc"]
                title = (f"level {n_ - nocc + 1:+d} (0 = HOMO)" + (f" spin {ch[s_][4]}" if mf else "") +
                         f", E - E_F = {levels[s_][n_]:+.4f} eV: |psi|^2, tip {w:g} A")
            else:
                maps = [tb.cache_map(c, Em, sig, shift=ref) for c, ref, *_ in ch]
                if not mf:
                    v, what = maps[0], "dI/dV"
                elif mode == MAP_UP:
                    v, what = maps[0], "dI/dV spin up"
                elif mode == MAP_DN:
                    v, what = maps[1], "dI/dV spin down"
                elif mode == MAP_UPDN:
                    v, what, signed = maps[0] - maps[1], "dI/dV up - down (red up)", True
                else:
                    v, what = maps[0] + maps[1], "dI/dV up + down"
                title = f"{what} at E - E_F = {Em:+.4f} eV (broadening {stxt}, tip {w:g} A)"
            self._draw_map(ax, fs, v, [0], w, signed=signed, unit="" if mode == MAP_LEVEL else "states/eV")
        for n, i in enumerate(sel):
            ax.scatter(*fs.xy[i], s=50, facecolor="none", edgecolor=COLORS[n % len(COLORS)], linewidth=2, zorder=5)
        ax.set_title(title, fontsize=9)
        # spectra
        ax = self.ax_p3
        if mf:
            for (dos, L), (c, ref, nel, col, lab), ls in zip(spec, ch, ("-", "--")):
                ax.plot(x, dos / fs.N, color=col, lw=0.8, alpha=0.5, ls=ls, label=f"average {lab}")
            for k, i in enumerate(sel):
                col = COLORS[k % len(COLORS)]
                ax.plot(x, spec[0][1][k], color=col, lw=1.4, label=f"atom {i} up")
                ax.plot(x, spec[1][1][k], color=col, lw=1.4, ls="--", label=f"atom {i} down")
            ax.set_ylabel("LDOS per spin (states / eV)")
            ax.set_title(f"spin-resolved dI/dV spectra (mean field), broadening {stxt}", fontsize=9)
        else:
            ax.plot(x, spec[0][0] / fs.N, color="0.6", lw=1, ls="--", label="average per atom")
            for k, i in enumerate(sel):
                ax.plot(x, spec[0][1][k], color=COLORS[k % len(COLORS)], lw=1.2, label=f"atom {i}")
            ax.set_ylabel("LDOS (states / eV)")
            ax.set_title(f"dI/dV spectra, broadening {stxt} (click atoms)", fontsize=9)
        ax.axvline(0, color="r", lw=0.5, ls="--")
        ax.axvline(Em, color="tab:blue", lw=0.8)
        ax.set_xlim(*W)
        ax.set_xlabel("E - E_F (eV)  (sample bias)")
        ax.legend(fontsize=7, loc="upper right")
        self._set_lims(saved)
        self.fig3.tight_layout()
        self.can3.draw_idle()
        self._views["tab3"] = (*self.ax_l3.get_ylim(), *self.ax_p3.get_xlim())

    def f_click(self, ev):
        if self.fst is None or ev.xdata is None:
            return
        fs = self.fst
        if ev.inaxes in (self.ax_s3, self.ax_m3):
            d = np.linalg.norm(fs.xy - [ev.xdata, ev.ydata], axis=1)
            i = int(np.argmin(d))
            if d[i] > 0.7:
                return
            if self.f_mode.get() == "remove":
                self.f_removed.append(fs.xy[i].copy())
                old = [fs.xy[k] for k in self.f_sel if k != i]
                self.fst = tb.remove_atoms(self.fbase, self.f_removed)
                self.fst.name = self.fbase.name
                self.f_sel = [int(np.argmin(np.linalg.norm(self.fst.xy - p, axis=1))) for p in old]
                self.status.set(f"removed atom at ({fs.xy[i][0]:.2f}, {fs.xy[i][1]:.2f}); recomputed")
                self.f_calc()
                return
            self.f_sel.remove(i) if i in self.f_sel else self.f_sel.append(i)
            self.f_sel = self.f_sel[-8:]
            self.status.set(f"finite chain: atom {i}  x = {fs.xy[i][0]:.3f}  y = {fs.xy[i][1]:.3f} A")
            self.f_draw(keep_view=True)
        elif ev.inaxes in (self.ax_l3, self.ax_d3, self.ax_p3) and self.fres is not None:
            o = self._f_mf_state()
            ch = self._f_channels(o)
            levels = [c["E"][0] - ref for c, ref, *_ in ch]
            E = ev.xdata if ev.inaxes is self.ax_p3 else ev.ydata
            if ev.inaxes is self.ax_l3 and o is not None:
                s_ = 0 if ev.xdata < 0.5 else 1
            else:
                s_ = int(np.argmin([np.abs(e - E).min() for e in levels]))
            n_ = int(np.argmin(np.abs(levels[s_] - E)))
            self._lev3 = (s_, n_)
            if self.f_map.get() == MAP_LEVEL:
                E = levels[s_][n_]
            if self.f_map.get() == MAP_SPIN:
                self.f_map.set(MAP_LDOS)
            self.f_emap.set(f"{E:.4f}")
            lab = f" spin {ch[s_][4]}" if o is not None else ""
            self.status.set(f"level {n_ - (ch[s_][2] if o is not None else self.fres['nocc']) + 1:+d} "
                            f"(0 = HOMO){lab}: E - E_F = {levels[s_][n_]:+.4f} eV")
            self.f_draw(keep_view=True)

    def _f_rebuild_keep_sel(self):
        old = [self.fst.xy[k] for k in self.f_sel]
        self.fst = tb.remove_atoms(self.fbase, self.f_removed)
        self.fst.name = self.fbase.name
        new = []
        for p in old:
            d = np.linalg.norm(self.fst.xy - p, axis=1)
            if d.min() < 0.3:
                new.append(int(np.argmin(d)))
        self.f_sel = new
        self.f_calc()

    def f_undo(self):
        if self.f_removed:
            self.f_removed.pop()
            self._f_rebuild_keep_sel()

    def f_restore(self):
        if self.f_removed:
            self.f_removed = []
            self._f_rebuild_keep_sel()

    def f_save(self):
        if self.fst is None:
            return
        p = filedialog.asksaveasfilename(filetypes=[("xyz", "*.xyz"), ("ChemDraw", "*.cdxml")],
                                         defaultextension=".xyz")
        if not p:
            return
        try:
            if p.lower().endswith(".cdxml"):
                tb.write_cdxml(self.fst, p)
            else:
                with open(p, "w") as f:
                    f.write(f"{self.fst.N}\nfinite molecule: {self.fst.name}\n")
                    for e, (x, y) in zip(self.fst.el, self.fst.xy):
                        f.write(f"{e} {x:.6f} {y:.6f} 0.0\n")
        except Exception as e:
            messagebox.showerror("Save", str(e))
            return
        self.status.set(f"saved {p}")

    def f_export(self):
        if self.fres is None:
            return
        p = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV", "*.csv")])
        if not p:
            return
        r, fs = self.fres, self.fst
        W = self._win3 or (float(self.f_emin.get()), float(self.f_emax.get()))
        sig, auto = self._sigma(self.f_broad, W)
        x = self._egrid(W, sig)
        o = self._f_mf_state()
        ch = self._f_channels(o)
        cols, head = [x], ["E - E_F (eV)"]
        for c, ref, nel, col, lab in ch:
            dos, L = tb.cache_spectra(c, x, sig, range(fs.N), shift=ref)
            tag = f" {lab}" if lab else ""
            cols += [dos, L.T]
            head += [f"DOS{tag}", f"LDOS{tag} of atom 0, 1, 2, ..."]
        np.savetxt(p, np.column_stack(cols), delimiter=",",
                   header=", ".join(head) + f"; Gaussian broadening {1e3 * sig:.4g} meV" +
                   (f"; mean-field state {o['label']}" if o is not None else ""))
        if o is None:
            np.savetxt(os.path.splitext(p)[0] + "_levels.csv", r["E"], delimiter=",",
                       header=f"levels E - E_F (eV); E_F = {r['EF']:.6f} eV; {r['nocc']} occupied")
        else:
            lv = [c["E"][0] - ref for c, ref, *_ in ch]
            np.savetxt(os.path.splitext(p)[0] + "_levels.csv", np.c_[lv[0], lv[1]], delimiter=",",
                       header=f"levels up, down: E - E_F (eV) of the mean-field state {o['label']}; "
                              f"{ch[0][2]} up and {ch[1][2]} down electrons")
        self.status.set(f"saved {p} and the levels")


    # ================================================================== tab 4: spin model
    def _scrollable(self, parent, width=430):
        """A vertically scrollable frame (for long control panels)."""
        outer = ttk.Frame(parent)
        outer.pack(side="left", fill="y")
        canvas = tk.Canvas(outer, width=width, highlightthickness=0)
        sb = ttk.Scrollbar(outer, orient="vertical", command=canvas.yview)
        inner = ttk.Frame(canvas, padding=6)
        inner.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=inner, anchor="nw")
        canvas.configure(yscrollcommand=sb.set)
        canvas.pack(side="left", fill="y", expand=True)
        sb.pack(side="left", fill="y")
        return inner

    def _build_tab4(self):
        self.sm = dict(cluster=None, ed=None, fit=None, mfh=None, rows=[], sel=None)
        left = self._scrollable(self.tab4, width=540)

        box = ttk.LabelFrame(left, text="1. Effective model of tab 2 + Hubbard U", padding=6)
        box.pack(fill="x")
        ttk.Label(box, text="The sites, on-site energies eps_i and hoppings t_ij listed below are the fitted "
                            "parameters of tab 2: edit and refit them there (tab 4 always uses the current tab 2 "
                            "model, refitting it if the editor changed). The sites are Wannier-like orbitals w_i, "
                            "U_i = U_atom sum_x |w_i(x)|^4.",
                  foreground="gray", wraplength=500, justify="left").pack(anchor="w")
        self.sm_modeltxt = tk.Text(box, width=68, height=8, font=("Courier", 9), background="#f4f4f4")
        self.sm_modeltxt.pack(fill="x", pady=(2, 4))
        self.sm_modeltxt.insert("1.0", "(press 'Load sites from the model' to read the tab 2 model)")
        self.sm_modeltxt.configure(state="disabled")
        self.sm_U, self.sm_thr = self._entry2(box, "U_atom (eV)", "3.0", "keep > (meV)", "1")
        row = ttk.Frame(box)
        row.pack(fill="x")
        self.sm_mode = tk.StringVar(value="onsite")
        ttk.Radiobutton(row, text="on-site U only", value="onsite", variable=self.sm_mode).pack(side="left")
        ttk.Radiobutton(row, text="full projected interaction", value="full", variable=self.sm_mode).pack(side="left")
        self.sm_ph = tk.BooleanVar(value=True)
        ttk.Checkbutton(box, text="particle-hole symmetric form U(n_up-1/2)(n_dn-1/2) (no double counting)",
                        variable=self.sm_ph).pack(anchor="w")
        ttk.Button(box, text="Load sites from the model (tab 2) and compute U_i", command=self.sm_sites).pack(fill="x", pady=(4, 2))
        self.sm_table = ttk.Frame(box)
        self.sm_table.pack(fill="x")

        box = ttk.LabelFrame(left, text="2. Many-body solution: ED or DMRG (cluster of unit cells)", padding=6)
        box.pack(fill="x", pady=(6, 0))
        ttk.Label(box, text="H = sum eps_i n_is + sum t_ij c+_is c_js\n"
                            "  + sum U_i (n_i,up - 1/2)(n_i,dn - 1/2)  [+ inter-site]\n"
                            "eps_i, t_ij: tab 2 fit; U_i: box 1; 'cells' copies",
                  foreground="#1f4e79", font=("Courier", 9)).pack(anchor="w")
        self._solver_row(box, "sm")
        self.sm_cells, self.sm_nst = self._entry2(box, "cells", "2", "states", "auto")
        row = ttk.Frame(box)
        row.pack(fill="x")
        self.sm_bc = tk.StringVar(value="open")
        ttk.Radiobutton(row, text="open chain", value="open", variable=self.sm_bc).pack(side="left")
        ttk.Radiobutton(row, text="ring (periodic)", value="ring", variable=self.sm_bc).pack(side="left")
        self.sm_keep = self._entry(box, "last cell keeps sites (blank: all)", "")
        ttk.Label(box, text="odd chains (open chain only): the tab 2 sites kept in the last cell,\n"
                            "the first ones along the chain. Olympicene (sites A, B): 3 cells,\n"
                            "keep 'A' -> A B A B A. A cell of two units with 3 sites each: 2 cells,\n"
                            "keep the 3 sites of its first unit -> 3 units (Run ED lists the order)",
                  foreground="gray").pack(anchor="w")
        self.sm_nel = self._entry(box, "electrons (blank: neutral)", "")
        self.sm_dim = tk.StringVar(value="")
        ttk.Label(box, textvariable=self.sm_dim, foreground="gray").pack(anchor="w")
        self.sm_btn_ed = ttk.Button(box, text="Run ED", command=self.sm_run_ed)
        self.sm_btn_ed.pack(fill="x", pady=(4, 0))
        self._solver_changed("sm")
        ttk.Label(box, text="then click a level in the spectrum: transition from the ground state\n"
                            "(spin-flip map, Delta S rule, change of <S_i.S_j>); click the ground\n"
                            "state to go back to its correlations",
                  foreground="gray").pack(anchor="w")

        box = ttk.LabelFrame(left, text="3. Spin model (spin sites ticked in box 1)", padding=6)
        box.pack(fill="x", pady=(6, 0))
        ttk.Label(box, text="H_spin = sum_(ij) J_ij S_i . S_j  (spins 1/2)\n"
                            "J_ij fitted to the lowest ED states of box 2",
                  foreground="#1f4e79", font=("Courier", 9)).pack(anchor="w")
        self.sm_maxdR = self._entry(box, "max distance (cells)", "1")
        ttk.Label(box, text="couplings (same name = same J; edit to impose symmetries):",
                  foreground="gray").pack(anchor="w")
        self.sm_jtext = tk.Text(box, width=64, height=6, font=("Courier", 9))
        self.sm_jtext.pack(fill="x")
        row = ttk.Frame(box)
        row.pack(fill="x", pady=(2, 0))
        ttk.Button(row, text="Generate couplings", command=self.sm_gen_pairs).pack(side="left")
        ttk.Button(row, text="Fit J to the ED spectrum", command=self.sm_fit).pack(side="left", fill="x", expand=True)

        box = ttk.LabelFrame(left, text="4. Mean-field Hubbard (atomistic, benchmark)", padding=6)
        box.pack(fill="x", pady=(6, 0))
        ttk.Label(box, text="H_s = H_TB + U sum_x <n_x,-s> n_x,s\n"
                            "E = sum_occ e - U sum_x <n_x,up><n_x,dn>\n"
                            "H_TB: all carbons, t1 t2 t3 of tab 1 (NOT the tab 2 model)",
                  foreground="#1f4e79", font=("Courier", 9)).pack(anchor="w")
        self.sm_mfU, self.sm_mfk = self._entry2(box, "U (eV)", "3.0", "k points", "24")
        self.sm_mfT = self._entry(box, "kT (eV)", "0.005")
        ttk.Label(box, text="FM and AFM starting spins on the atoms of the spin sites\n"
                            "(AFM: by sublattice); same conventions as the 'hubbard' package",
                  foreground="gray").pack(anchor="w")
        self.sm_btn_mf = ttk.Button(box, text="Run mean-field Hubbard (FM and AFM)", command=self.sm_run_mfh)
        self.sm_btn_mf.pack(fill="x", pady=(4, 0))

        self.sm_out = tk.Text(left, width=68, height=22, font=("Courier", 9))
        self.sm_out.pack(fill="both", expand=True, pady=(6, 0))

        right = ttk.Frame(self.tab4)
        right.pack(side="left", fill="both", expand=True)
        self.fig4 = Figure(figsize=(10, 8))
        gs = self.fig4.add_gridspec(2, 2, width_ratios=[2.2, 1])
        self.ax_c4 = self.fig4.add_subplot(gs[0, 0])
        self.ax_e4 = self.fig4.add_subplot(gs[0, 1])
        self.ax_m4 = self.fig4.add_subplot(gs[1, 0])
        self.ax_b4 = self.fig4.add_subplot(gs[1, 1])
        self.can4 = FigureCanvasTkAgg(self.fig4, right)
        self.tb4 = NavigationToolbar2Tk(self.can4, right)
        self.tb4.update()
        self.can4.get_tk_widget().pack(fill="both", expand=True)
        self._on_clicks(self.can4, self.sm_click)

    # ------------------------------------------------------------------ helpers
    def _sm_log(self, text, clear=False):
        if clear:
            self.sm_out.delete("1.0", "end")
        self.sm_out.insert("end", text + "\n")
        self.sm_out.see("end")

    def _sm_background(self, work, done, button):
        """Run work() in a thread, then done(result) in the GUI thread."""
        box = {}

        def target():
            try:
                box["r"] = work()
            except Exception as e:           # reported in the GUI thread
                box["e"] = e

        th = threading.Thread(target=target, daemon=True)
        button.state(["disabled"])
        self.config(cursor="watch")
        th.start()

        def poll():
            if th.is_alive():
                self.after(200, poll)
                return
            button.state(["!disabled"])
            self.config(cursor="")
            if "e" in box:
                messagebox.showerror("Spin model", str(box["e"]))
            else:
                done(box["r"])
        self.after(200, poll)

    def _sm_model(self):
        """The fitted effective model of tab 2 (fits it if needed)."""
        if self.st is None:
            raise ValueError("load a structure in tab 1 first")
        m = tb.parse_eff(self.editor.get("1.0", "end"), self.st)
        if self.H is None:
            self.calc_bands()
        if self.fitres is None or self.model is None or self.model.text() != m.text():
            self.fitres = tb.fit(self.H, self.st, m, use_projection=self.use_proj.get())
            self.model = m
            if hasattr(self, "sm_out"):
                self._sm_log("(the tab 2 model changed: refitted, the parameters in box 1 are updated)")
        if hasattr(self, "sm_modeltxt"):
            self._sm_show_model(m, self.fitres)
        return m, self.fitres

    # ------------------------------------------------------------------ Hamiltonians as text
    @staticmethod
    def _term_name(m, i, j, R):
        if i == j and R == 0:
            return f"eps {m.sites[i]}"
        return f"t {m.sites[i]}-{m.sites[j]}" + ("" if R == 0 else f" ({'next' if R == 1 else f'+{R}'} cell)")

    def _sm_show_model(self, m, r, widget=None):
        """Box 1: the tab 2 parameters that the ED uses."""
        pv = r["params"]
        L = [f"tab 2 model: {len(m.sites)} sites ({', '.join(m.sites)}); band fit rms {1e3 * r['rms']:.2f} meV",
             f"E_F = {r['EF']:+.4f} eV (the ED measures eps from E_F)",
             f"{'param':8s}{'value (eV)':>11s}   used in the ED as"]
        for p in m.params:
            uses = [("-" if sg < 0 else "") + self._term_name(m, i, j, R) for i, j, R, q, sg in m.terms if q == p]
            L.append(f"{p:8s}{pv[p]:+11.4f}   " + ", ".join(uses))
        w = self.sm_modeltxt if widget is None else widget
        w.configure(state="normal")
        w.delete("1.0", "end")
        w.insert("1.0", "\n".join(L))
        w.configure(height=min(12, len(L) + 1), state="disabled")

    def _sm_ed_hamiltonian(self, m, r, cl, mode, u_source="box 1"):
        """The Hamiltonian of the ED with its numbers."""
        pv, ns = r["params"], len(m.sites)
        L = [f"Hamiltonian of the cluster ({self._cells_text(cl)}, {'ring' if cl['periodic'] else 'open chain'}, "
             f"sites {cl['labels'][0]} .. {cl['labels'][-1]}; the digit is the cell):",
             "  H = sum_i,s eps_i n_is + sum_ij,s t_ij c+_is c_js + sum_i U_i (n_i,up - 1/2)(n_i,dn - 1/2)"
             + ("  + inter-site terms" if mode == "full" else ""),
             "  eps_i - E_F (eV, tab 2): " + ", ".join(
                 f"{m.sites[i]} {sum(sg * pv[q] for i2, j2, R2, q, sg in m.terms if i2 == i and j2 == i and R2 == 0) - r['EF']:+.4f}"
                 for i in range(ns)),
             "  t_ij (eV, tab 2): " + ", ".join(
                 f"{m.sites[i]}-{m.sites[j]}{'' if R == 0 else chr(39) * R} {sg * pv[q]:+.4f} ({q})"
                 for i, j, R, q, sg in m.terms if not (i == j and R == 0))
             + "   (' = site in the next cell)",
             f"  U_i (eV, {u_source}): " + ", ".join(f"{m.sites[i]} {cl['U_site'][i]:.4f}" for i in range(ns))]
        if mode == "full":
            other = sorted((t for t in cl["terms"] if not (t[0] == t[1] == t[2] == t[3])), key=lambda t: -abs(t[4]))
            lab = cl["labels"]

            def kind(a, b, c, d):
                if a == b and c == d:
                    return f"V {lab[a]}-{lab[c]} n_up n_dn"
                if a == d and b == c:
                    return f"exchange {lab[a]}-{lab[b]}"
                if a == c and b == d:
                    return f"pair hopping {lab[a]}<-{lab[b]}"
                return f"assisted hopping c+{lab[a]} c{lab[b]} c+{lab[c]} c{lab[d]}"
            L.append(f"  inter-site terms v c+_a,up c_b,up c+_c,dn c_d,dn kept (|v| > threshold): {len(other)}; largest:")
            L += [f"    {kind(*t[:4])}  {1e3 * t[4]:+.2f} meV" for t in other[:4]]
        return L

    # ------------------------------------------------------------------ 1. sites and U_i
    def sm_sites(self):
        try:
            m, r = self._sm_model()
            w, hR, smin = tm.wannier_orbitals(self.H, self.st, m)
        except Exception as e:
            messagebox.showerror("Spin model", str(e))
            return
        ipr, n90 = tm.orbital_spread(w)
        U = float(self.sm_U.get())
        self._sm_show_model(m, r)
        for wdg in self.sm_table.winfo_children():
            wdg.destroy()
        hdr = ["site", "atoms 90%", "U_i (eV)", "spin"]
        for c, h in enumerate(hdr):
            ttk.Label(self.sm_table, text=h, foreground="gray").grid(row=0, column=c, padx=3, sticky="w")
        self.sm["rows"] = []
        # default: the most localised sites carry the spins
        loc = ipr >= 0.8 * ipr.max()
        for k, name in enumerate(m.sites):
            uvar = tk.StringVar(value=f"{U * ipr[k]:.4f}")
            svar = tk.BooleanVar(value=bool(loc[k]))
            ttk.Label(self.sm_table, text=name).grid(row=k + 1, column=0, sticky="w")
            ttk.Label(self.sm_table, text=str(n90[k])).grid(row=k + 1, column=1)
            ttk.Entry(self.sm_table, textvariable=uvar, width=8).grid(row=k + 1, column=2)
            ttk.Checkbutton(self.sm_table, variable=svar).grid(row=k + 1, column=3)
            self.sm["rows"].append((name, uvar, svar, float(U * ipr[k])))
        self._sm_log(f"model: {len(m.sites)} sites, fit rms {1e3 * r['rms']:.2f} meV; projection overlap {smin:.3f}\n"
                     f"U_i = {U} eV x sum|w|^4: " +
                     ", ".join(f"{n} {U * v:.3f}" for n, v in zip(m.sites, ipr)) +
                     "\n(the most localised sites are proposed as spin sites; edit freely)", clear=True)
        self.sm_gen_pairs()

    def _sm_settings(self):
        rows = self.sm["rows"]
        if not rows:
            raise ValueError("press 'Load sites from the model and compute U_i' first")
        Uov = {}
        for name, uvar, svar, uauto in rows:
            u = float(uvar.get())
            if abs(u - uauto) > 1e-4:
                Uov[name] = u
        spins = [n for n, u, s_, a in rows if s_.get()]
        if Uov and self.sm_mode.get() == "full":
            Uov = {n: float(u.get()) for n, u, s_, a in rows}
        return Uov, spins

    # ------------------------------------------------------------------ 2. ED
    def sm_run_ed(self):
        try:
            m, r = self._sm_model()
            Uov, spins = self._sm_settings()
            ncells = int(self.sm_cells.get())
            periodic = self.sm_bc.get() == "ring"
            ns = len(m.sites)
            keep = self._keep_list(self.sm_keep)
            if not self._keep_check(m, keep):
                return
            kept = tm.cluster_keep(m, ncells, keep, periodic)
            Ns = len(kept)
            nel_neutral, per_cell = tm.neutral_filling(m, r, ncells, Ns)
            nel = int(self.sm_nel.get()) if self.sm_nel.get().strip() else nel_neutral
            nup = (nel + nel % 2) // 2
            dim = comb(Ns, nup) * comb(Ns, nel - nup)
            nsp = sum(1 for i in kept if m.sites[i % ns] in spins)
            M = comb(nsp, nsp // 2) if nsp else 1
            solver = self._solver_opts("sm")
            nst = int(self.sm_nst.get()) if self.sm_nst.get().strip().isdigit() else min(dim, max(12, M + 4))
            if solver[0] == "dmrg" and not self.sm_nst.get().strip().isdigit():
                nst = min(nst, DMRG_MAX_AUTO_STATES)
            self.sm_dim.set(f"{Ns} sites, {nel} electrons: dimension {dim:,}; {nst} states"
                            + ("  (large: may take minutes)" if dim > 3e5 or solver[0] == "dmrg" else ""))
            if dim > 3e6 and solver[0] == "ed":
                raise ValueError(f"dimension {dim:,} is too large for ED; use fewer cells or the DMRG solver")
        except Exception as e:
            messagebox.showerror("Spin model", str(e))
            return
        U, thr, mode, ph = float(self.sm_U.get()), float(self.sm_thr.get()) / 1e3, self.sm_mode.get(), self.sm_ph.get()
        H, st = self.H, self.st

        def work():
            cl = tm.build_cluster(H, st, m, r["params"], ncells=ncells, periodic=periodic, U=U, mode=mode,
                                  thr=thr, EF=r["EF"], U_override=Uov or None, ph=ph, keep_last=keep)
            res = self._solve(cl, nel, nst, solver)
            res["levels"], res["lw"] = tm.level_weights(res)
            return cl, res

        def done(out):
            cl, res = out
            self.sm["cluster"], self.sm["ed"], self.sm["fit"], self.sm["sel"] = cl, res, None, None
            res["M"] = M if nsp else 0
            E = res["E"] - res["E"][0]
            L = [f"ED: {self._cells_text(cl)} ({'ring' if periodic else 'open chain'}), {nel} electrons "
                 f"(neutral: {per_cell:.2f} per cell in the model bands"
                 + (", NOT one per site" if abs(per_cell - ns) > 1e-6 else "") + "), "
                 f"{mode} interaction ({len(cl['terms'])} terms), dimension {res['dim']:,}"]
            L += self._solver_lines(res)
            L += self._sm_ed_hamiltonian(m, r, cl, mode)
            L += [
                 "lowest states (E - E0 in meV, total S):"]
            L += [f"  {k:3d} {1e3 * e:10.3f}   S = {s:.2f}" for k, (e, s) in enumerate(zip(E, res["S"]))]
            C0 = res["C"][0]
            L.append("ground state: charge / local moment <S_i^2> per site (0.75 = free spin):")
            L += [f"  {lab:6s} n = {n:.3f}  <S^2> = {C0[i, i]:.3f}" for i, (lab, n) in
                  enumerate(zip(cl["labels"], res["n"][0]))][:len(cl["labels"]) if len(cl["labels"]) <= 12 else 2 * ns]
            L += self._sm_transition_table(res, cl)
            self._sm_log("\n".join(L))
            self.sm_draw()

        self._sm_log(f"running {'DMRG' if solver[0] == 'dmrg' else 'ED'} (dimension {dim:,}, {nst} states) ...")
        self._sm_background(work, done, self.sm_btn_ed)

    # ------------------------------------------------------------------ 3. spin model
    def sm_gen_pairs(self):
        try:
            m, _ = self._sm_model()
            _, spins = self._sm_settings()
            ncells, periodic = int(self.sm_cells.get()), self.sm_bc.get() == "ring"
            kept = tm.cluster_keep(m, ncells, self._keep_list(self.sm_keep), periodic)
            sites, pairs, groups, labels = tm.spin_pairs(m, ncells, periodic, spins, int(self.sm_maxdR.get()),
                                                         *self._sm_pos(m), kept=kept)
        except Exception as e:
            messagebox.showerror("Spin model", str(e))
            return
        self.sm_jtext.delete("1.0", "end")
        self.sm_jtext.insert("1.0", "\n".join(f"{lab} = J{g + 1}" for g, lab in enumerate(labels)))

    def sm_fit(self):
        res, cl = self.sm["ed"], self.sm["cluster"]
        if res is None:
            messagebox.showerror("Spin model", "run the ED first")
            return
        try:
            m, r = self._sm_model()
            _, spins = self._sm_settings()
            sites, pairs, groups, labels = tm.spin_pairs(m, cl["ncells"], cl["periodic"], spins,
                                                         int(self.sm_maxdR.get()), *self._sm_pos(m),
                                                         kept=cl["kept"])
            names = {}
            for line in self.sm_jtext.get("1.0", "end").splitlines():
                if "=" in line:
                    lab, nm = line.rsplit("=", 1)
                    names[lab.strip()] = nm.strip()
            gname = [names.get(labels[g], f"J{g + 1}") for g in groups]
            uniq = list(dict.fromkeys(gname))
            groups2 = [uniq.index(n) for n in gname]
            nsp = len(sites)
            if nsp < 2:
                raise ValueError("tick at least one spin site")
            nel_ed = res["n_up"] + res["n_dn"]
            if (nel_ed - nsp) % 2:
                raise ValueError(f"the ED has {nel_ed} electrons but the spin model has {nsp} spins: with "
                                 "different parity the lowest S_z sectors differ (an extra unpaired electron is "
                                 "not on a spin site). Tick the sites that carry the moments or change the "
                                 "electron number")
            M = comb(nsp, nsp // 2)
            if len(res["E"]) < M:
                raise ValueError(f"the spin model has {M} states in this sector but the ED computed only "
                                 f"{len(res['E'])}: set 'states' to at least {M + 1} and run the ED again")
            Cspin = res["C"][:M][:, sites][:, :, sites]
            J, Em, Sm, rms = tm.fit_spin_model(res["E"][:M], res["S"][:M], nsp, pairs, groups2, C_ref=Cspin)
        except Exception as e:
            messagebox.showerror("Spin model", str(e))
            return
        rank = getattr(tm.fit_spin_model, "rank", len(J))
        gap = (res["E"][M] - res["E"][M - 1]) if len(res["E"]) > M else float("nan")
        spread = res["E"][M - 1] - res["E"][0]
        self.sm["fit"] = dict(J=J, names=uniq, Em=Em, Sm=Sm, M=M, rms=rms)
        L = ["", f"spin model on {nsp} spins ({', '.join(spins)}), fitted to the lowest {M} ED states:"]
        for n, j in zip(uniq, J):
            bonds = [labels[g] for g in sorted(set(groups)) if gname[groups.index(g)] == n]
            L.append(f"  {n:6s} = {1e3 * j:+10.4f} meV ({'AFM' if j > 0 else 'FM '})  {'; '.join(bonds)}")
        lab = cl["labels"]
        L.append("  Hamiltonian:  H_spin = " + "\n                + ".join(
            f"{n} ({' + '.join(f'S_{lab[sites[a]]}.S_{lab[sites[b]]}' for (a, b), g in zip(pairs, groups2) if g == k)})"
            for k, n in enumerate(uniq)))
        L.append("  with " + ", ".join(f"{n} = {1e3 * j:+.4f} meV" for n, j in zip(uniq, J)) +
                 "  (J > 0 antiferromagnetic, J < 0 ferromagnetic)")
        L.append(f"  rms error {1e3 * rms:.4f} meV; spin manifold spread {1e3 * spread:.3f} meV, "
                 f"gap to the next state {1e3 * gap:.3f} meV")
        if not gap > 3 * spread:
            L.append("  WARNING: the lowest states are not an isolated spin manifold (other excitations are as low):"
                     " the Heisenberg mapping on these sites is not reliable (add sites / change sites)")
        if rank < len(J):
            L.append(f"  WARNING: only {rank} of {len(J)} couplings are determined by the spectrum;"
                     " give equivalent couplings the same name")
        cerr = getattr(tm.fit_spin_model, "corr_err", float("nan"))
        L.append(f"  ground-state <S_i.S_j>: largest difference Heisenberg - ED = {cerr:.3f} "
                 "(the fit uses the correlations too: the spectrum alone cannot tell which bond is which)")
        nun = getattr(tm.fit_spin_model, "unmatched", 0)
        if nun:
            L.append(f"  WARNING: {nun} of the lowest {M} ED states have a total spin that {nsp} spins 1/2 in "
                     "this sector cannot have (too many states of that S): they are not spin-manifold states")
        if rms > 0.1 * max(spread, 1e-9):
            L.append("  WARNING: large fit error compared with the spin spectrum")
        if cerr > 0.1:
            L.append("  WARNING: with these equal couplings the Heisenberg model does not reproduce the ED "
                     "correlations: the couplings depend on the position in the cluster (edge effects: try a "
                     "ring or more cells) or longer-range / multi-spin terms matter. Giving bonds separate "
                     "names shows how different they are in this cluster")
        # analytic estimates
        est = self._sm_estimates(m, r["params"], cl)
        if est:
            L += ["analytic estimates (perturbation theory):"] + est
        self._sm_log("\n".join(L))
        self.sm_draw()

    # ------------------------------------------------------------------ ED / DMRG selector
    def _solver_row(self, box, pre):
        """Solver selector (ED or DMRG) with the DMRG settings, for tabs 4 and 7."""
        row = ttk.Frame(box)
        row.pack(fill="x", pady=(2, 0))
        ttk.Label(row, text="solver").pack(side="left")
        var = tk.StringVar(value=SOLVER_ED)
        cb = ttk.Combobox(row, textvariable=var, values=[SOLVER_ED, SOLVER_DMRG], state="readonly", width=28)
        cb.pack(side="left", padx=4)
        cb.bind("<<ComboboxSelected>>", lambda e: self._solver_changed(pre))
        M, sw = self._entry2(box, "bond dim", "500", "sweeps", "24")
        info = tk.StringVar(value="")
        ttk.Label(box, textvariable=info, foreground="gray", justify="left").pack(anchor="w")
        setattr(self, f"{pre}_solver", var)
        setattr(self, f"{pre}_M", M)
        setattr(self, f"{pre}_sweeps", sw)
        setattr(self, f"{pre}_solver_info", info)

    def _solver_changed(self, pre):
        dmrg = getattr(self, f"{pre}_solver").get() == SOLVER_DMRG
        btn = getattr(self, f"{pre}_btn_ed", None)
        if btn is not None:
            btn.configure(text="Run DMRG" if dmrg else "Run ED")
        info = getattr(self, f"{pre}_solver_info")
        if not dmrg:
            info.set("ED: exact, up to about 3 million states (12-13 sites at half filling)")
            return
        ok, msg = tdm.available()
        info.set(("DMRG (block2 " + msg + "): lowest states with energies, spins, correlations,\n"
                  "spin-flip and charge spectra as with ED; for larger clusters. Check the\n"
                  "convergence: increase the bond dimension until the energies do not change")
                 if ok else "DMRG needs the block2 package:  python -m pip install block2")

    def _solver_opts(self, pre):
        """('ed', None) or ('dmrg', options); raises ValueError if DMRG is not available."""
        if getattr(self, f"{pre}_solver").get() != SOLVER_DMRG:
            return "ed", None
        ok, msg = tdm.available()
        if not ok:
            raise ValueError(msg)
        M, sw = int(getattr(self, f"{pre}_M").get()), int(getattr(self, f"{pre}_sweeps").get())
        if M < 10 or sw < 4:
            raise ValueError("DMRG: bond dimension >= 10 and at least 4 sweeps")
        return "dmrg", dict(bond_dim=M, sweeps=sw)

    @staticmethod
    def _solve(cl, nel, nst, solver):
        """Many-body states of the cluster with the selected solver."""
        kind, opts = solver
        if kind == "dmrg":
            return tdm.solve_cluster(cl, n_elec=nel, nstates=nst, opts=opts)
        return tm.solve_cluster(cl, n_elec=nel, nstates=nst)

    @staticmethod
    def _solver_lines(res):
        if res.get("solver") == "dmrg":
            return tdm.report(res)
        return [f"exact diagonalisation (sparse Lanczos / dense), dimension {res['dim']:,}"]

    @staticmethod
    def _keep_list(var):
        """Site names typed in a 'last cell keeps' entry (None when blank)."""
        txt = var.get().replace(",", " ").split()
        return txt or None

    def _sites_along_chain(self, m):
        """The sites of the model in order along the chain: [(name, position along T in A)]."""
        pos, T, _ = self._sm_pos(m)
        x = pos @ (np.asarray(T, float) / np.linalg.norm(T))
        x = np.round(x - x.min(), 1)               # sites of one unit (same x) keep the model order
        return [(m.sites[i], float(x[i])) for i in np.lexsort((np.arange(len(x)), x))]

    def _keep_check(self, m, keep):
        """Odd chains: the names typed in 'last cell keeps sites' must be sites of the tab 2 model
        and the first ones along the chain; otherwise the dropped sites leave a gap between the
        previous cell and the kept ones. Raises ValueError for unknown names; for a gap asks
        whether to run anyway. Returns True to go on."""
        if not keep:
            return True
        try:
            order = self._sites_along_chain(m)
        except Exception:                        # sites without atoms: no positions
            order = [(n, 0.0) for n in m.sites]
        listing = ", ".join(f"{n} ({x:.1f} A)" for n, x in order)
        bad = [n for n in keep if n not in m.sites]
        if bad:
            half = " ".join(n for n, x in order[:max(1, len(order) // 2)])
            raise ValueError(f"unknown site(s) {' '.join(bad)} in 'last cell keeps sites'.\n\n"
                             "Type the names of the tab 2 sites that the last cell keeps; they must be the first "
                             f"ones along the chain. Sites of this model in order along the chain: {listing}.\n\n"
                             f"For example '{half}' (the first half of the cell).")
        kept = [x for n, x in order if n in keep]
        dropped = [x for n, x in order if n not in keep]
        if not dropped or max(kept) <= min(dropped) + 0.5:
            return True
        sugg = " ".join(n for n, x in order[:len(kept)])
        return messagebox.askyesno(
            "Partial last cell",
            f"The last cell keeps {' '.join(keep)}, but other sites of the cell come before them along "
            f"the chain ({listing}). Dropping those leaves a gap: the kept sites would hang on the "
            f"previous cell only through long-range hoppings.\n\nFor a continuous chain keep '{sugg}'.\n\n"
            "Run anyway?")

    @staticmethod
    def _cells_text(cl):
        """'3 cells' or '3 cells, last one partial (A): 5 sites'."""
        ns, nc, kept = cl["ns"], cl["ncells"], cl["kept"]
        if len(kept) == ns * nc:
            return f"{nc} cells, {len(kept)} sites"
        last = [cl["labels"][k] for k, (c, n) in enumerate(cl["site_of"]) if c == nc - 1]
        return f"{nc} cells, last one partial ({' '.join(last)}): {len(kept)} sites"

    def _cluster_pos(self, m, cl):
        """Position of every cluster site (centroid of its atoms, shifted by its cell)."""
        st = self.st
        return np.array([st.xy[m.site_atom[n]].mean(axis=0) + c * np.asarray(st.T) for c, n in cl["site_of"]])

    def _sm_pos(self, m):
        """Site positions (centroid of their atoms) and the period, for grouping bonds."""
        return np.array([self.st.xy[a].mean(axis=0) for a in m.site_atom]), self.st.T, tb.symmetry_ops(self.st)

    def _sm_estimates(self, m, pv, cl):
        """Superexchange 4t^2/U (spin-spin hopping) and Kondo 8tau^2/U (spin - other site)."""
        _, spins = self._sm_settings()
        ns = len(m.sites)
        Ui = cl["U_site"][:ns]
        out = []
        for i, j, R, p, s in m.terms:
            if i == j and R == 0:
                continue
            t = s * pv[p]
            a, b = m.sites[i], m.sites[j]
            if a in spins and b in spins:
                Uav = 0.5 * (Ui[i] + Ui[j])
                if Uav > 0:
                    out.append(f"  superexchange {a}-{b} (R={R}): 4t^2/U = {1e3 * tm.superexchange(t, Uav):.3f} meV")
            elif (a in spins) != (b in spins):
                f = i if a in spins else j
                if Ui[f] > 0:
                    jk = tm.kondo_coupling(t, -Ui[f] / 2, Ui[f])
                    out.append(f"  spin site - other site {m.sites[f]} - {b if f == i else a} (R={R}): "
                               f"8 tau^2/U = {1e3 * jk:.3f} meV (exchange of the moment with that orbital)")
        return out

    # ------------------------------------------------------------------ 4. mean-field Hubbard
    def sm_run_mfh(self):
        try:
            m, r = self._sm_model()
            _, spins = self._sm_settings()
            if not spins:
                raise ValueError("tick at least one spin site")
            U, nk, kT = float(self.sm_mfU.get()), int(self.sm_mfk.get()), float(self.sm_mfT.get())
        except Exception as e:
            messagebox.showerror("Spin model", str(e))
            return
        st, H = self.st, self.H
        col = tb.sublattice_colors(st)
        sites = [(name, m.site_atom[k]) for k, name in enumerate(m.sites) if name in spins]
        sites_ = sites
        N, ne = st.N, st.n_electrons
        nsp = len(sites)

        def allowed(two_sz):
            """2 S_z with the parity of the electron number."""
            return two_sz if (ne - two_sz) % 2 == 0 else two_sz - 1

        def work():
            up_fm = [a for _, atoms in sites for a in atoms]
            s_fm = allowed(nsp)
            fm = tm.mean_field_hubbard(H, (ne + s_fm) // 2, (ne - s_fm) // 2, U=U, nk=nk, kT=kT,
                                       n0=tm.polarized_start(N, (ne + s_fm) // 2, (ne - s_fm) // 2, up_fm, []))
            up, dn = [], []
            cols = [col[atoms[0]] if col is not None else k_ % 2 for k_, (_, atoms) in enumerate(sites)]
            if len(set(cols)) < 2:
                # all spin sites on one sublattice (Lieb: FM coupling expected): the AFM trial
                # state alternates the spin sites instead, so it differs from the FM one
                cols = [k_ % 2 for k_ in range(nsp)]
            for c, (_, atoms) in zip(cols, sites):
                (up if c == 0 else dn).extend(atoms)
            # AFM: spins by sublattice; S_z from the imbalance of the spin sites (Lieb)
            dS = allowed(abs(sum(1 if c == 0 else -1 for c in cols)))
            af = tm.mean_field_hubbard(H, (ne + dS) // 2, (ne - dS) // 2, U=U, nk=nk, kT=kT,
                                       n0=tm.polarized_start(N, (ne + dS) // 2, (ne - dS) // 2, up, dn))
            pm = tm.mean_field_hubbard(H, ne // 2, ne - ne // 2, U=U, nk=nk, kT=kT, n0=np.full((2, N), ne / (2 * N)))
            return fm, af, pm, dS, s_fm

        def done(out):
            fm, af, pm, dS, s_fm = out
            self.sm["mfh"] = dict(fm=fm, af=af, pm=pm)
            tv = self.t_values()
            L = ["", f"mean-field Hubbard (atomistic, U = {U} eV, {nk} k points, kT = {kT} eV):",
                 "  Hamiltonian:  H_s = H_TB + U sum_x <n_x,-s> n_x,s   (x = every carbon, s = up, down)",
                 f"  H_TB: {st.N} atoms per cell, " + ", ".join(f"t{k + 1} = {t:+.3f} eV" for k, t in enumerate(tv))
                 + " (tab 1); energy E = sum_occ e - U sum_x <n_x,up><n_x,dn>",
                 "  (independent of the tab 2 model: a check of the effective-model results)",
                 f"  FM  (S_z = {s_fm / 2:g} per cell): E = {fm['E']:.6f} eV, max |m| = {np.abs(fm['n'][0] - fm['n'][1]).max():.3f}"
                 f"{'' if fm['converged'] else '  NOT converged'}",
                 f"  AFM (S_z = {dS / 2:g} per cell): E = {af['E']:.6f} eV, max |m| = {np.abs(af['n'][0] - af['n'][1]).max():.3f}"
                 f"{'' if af['converged'] else '  NOT converged'}",
                 f"  non-magnetic:          E = {pm['E']:.6f} eV",
                 f"  E_FM - E_AFM = {1e3 * (fm['E'] - af['E']):+.3f} meV per cell; "
                 f"E_nonmag - E_lowest = {1e3 * (pm['E'] - min(fm['E'], af['E'])):+.3f} meV",
                 "  (mean field breaks the spin symmetry and usually overestimates the couplings)"]
            if len(sites_) == 1:
                L.append("  NOTE: one spin site per cell: FM and AFM are the same state here; an AFM order "
                         "between cells needs a doubled cell (repeat the unit twice in the .hex)")
            self._sm_log("\n".join(L))
            self.sm_draw()

        self._sm_log("running mean-field Hubbard ...")
        self._sm_background(work, done, self.sm_btn_mf)

    # ------------------------------------------------------------------ transitions
    def _sm_transition_table(self, res, cl, nmax=12, clickable=True):
        """Overview of the transitions from the ground level (site spin-flip weights)."""
        lev, lw, S, E = res["levels"], res["lw"], res["S"], res["E"]
        g0 = lev[0]
        S0 = S[g0].mean()
        n0, D0 = res["n"][g0].mean(axis=0), np.diag(res["C"][g0].mean(axis=0))
        L = ["transitions from the ground state" + (" (click a level in the spectrum for its maps):" if clickable
              else " (maps: 'Use the fitted U_i in tab 4', then click the levels there):"),
             " W = sum_i sum_a |<f|S_i^a|0>|^2 (spin-flip weight, unpolarised tip, B = 0)",
             " rule: Delta S = 0, +-1 (not 0 -> 0); type: spin / spin* (local moments",
             " change) / charge (electrons move)",
             "  st  E-E0 (meV)    S      W   rule  type    strongest"]
        for li in range(1, min(len(lev), nmax + 1)):
            g = lev[li]
            Sf = S[g].mean()
            rule = abs(Sf - S0) < 1.01 and not (S0 < 0.25 and Sf < 0.25)
            kind = tm.classify(res["n"][g].mean(axis=0) - n0, np.diag(res["C"][g].mean(axis=0)) - D0)
            short = "charge" if kind.startswith("charge") else ("spin*" if "moments" in kind else "spin")
            w = lw[li]
            top = ",".join(cl["labels"][i] for i in np.argsort(w)[::-1][:2] if w[i] > 0.05 * max(w.max(), 1e-12))
            L.append(f"{g[0]:4d}{'+' if len(g) > 1 else ' '}{1e3 * (E[g[0]] - E[0]):10.4f} {Sf:4.1f} {w.sum():6.3f}   "
                     f"{'yes' if rule else 'no '}  {short:6s}  " + (top if w.sum() > 1e-4 else "-"))
        return L

    def sm_click(self, ev):
        """Click on a level of the spectrum: analysis of the transition from the ground state."""
        res, cl = self.sm["ed"], self.sm["cluster"]
        if res is None or ev.inaxes is not self.ax_e4 or ev.button != 1:
            return
        E = 1e3 * (res["E"] - res["E"][0])
        pts = self.ax_e4.transData.transform(np.c_[res["S"], E])
        d = np.hypot(pts[:, 0] - ev.x, pts[:, 1] - ev.y)
        k = int(np.argmin(d))
        if d[k] > 25:
            return
        lev = next(L for L in res["levels"] if k in L)
        if lev == res["levels"][0]:
            self.sm["sel"] = None
            self.sm_draw()
            return
        self.config(cursor="watch")
        self.update_idletasks()
        try:
            a = tm.analyse_transition(res, lev, W=cl["W"])
        except Exception as e:
            messagebox.showerror("Spin model", str(e))
            return
        finally:
            self.config(cursor="")
        self.sm["sel"] = a
        labs = cl["labels"]
        C0 = res["C"][a["g_level"]].mean(axis=0)
        L = ["", f"transition 0 -> {lev[0]}{' (degenerate states ' + str(lev) + ')' if len(lev) > 1 else ''}: "
                 f"Delta E = {1e3 * a['dE']:.4f} meV, S {a['S0']:.1f} -> {a['Sf']:.1f} (Delta S = {a['dS']:+g}), "
                 f"{'allowed' if a['allowed'] else 'forbidden'} by the Delta S = 0, +-1 rule; {a['kind']}",
             f"  total spin-flip weight W = {a['total']:.4f}"
             + ("  (zero: dark in inelastic spectroscopy)" if a["total"] < 1e-4 else ""),
             "  per site: " + ", ".join(f"{labs[i]} {a['w_site'][i]:.3f}" for i in np.argsort(a["w_site"])[::-1]
                                       if a["w_site"][i] > 1e-3 * max(a["w_site"].max(), 1e-12))]
        iu = np.triu_indices(len(labs), 1)
        order = np.argsort(np.abs(a["dC"][iu]))[::-1][:4]
        L.append("  largest changes of <S_i.S_j>: " + "; ".join(
            f"{labs[iu[0][o]]}-{labs[iu[1][o]]} {C0[iu[0][o], iu[1][o]]:+.3f} -> "
            f"{C0[iu[0][o], iu[1][o]] + a['dC'][iu[0][o], iu[1][o]]:+.3f}" for o in order if abs(a["dC"][iu][o]) > 1e-3))
        if a["moved"] >= 0.05 or np.abs(a["dS2"]).max() >= 0.05:
            L.append("  charge / local moment changes: " + ", ".join(
                f"{labs[i]} dn {a['dn'][i]:+.3f} d<S^2> {a['dS2'][i]:+.3f}" for i in range(len(labs))
                if abs(a["dn"][i]) > 0.02 or abs(a["dS2"][i]) > 0.02))
        self._sm_log("\n".join(L))
        self.sm_draw()

    # ------------------------------------------------------------------ drawing
    def _sm_draw_transition(self, ax, sel, cl, pos):
        """Cluster panel for a selected transition: simulated inelastic (spin-flip) map on the
        atoms, spin-flip weight per site, change of the spin correlations."""
        wa, xy = sel["w_atom"], cl["atom_xy"]
        if wa is not None and wa.max() > 0:
            keep = wa > 1e-3 * wa.max()
            img, ext = tb.ldos_image(tb.Structure(xy[keep], ["C"] * int(keep.sum()), self.st.T), wa[keep],
                                     cells=(0,), sigma=0.9)
            ax.imshow(img, origin="lower", extent=ext, cmap="Oranges", vmin=0, aspect="equal", zorder=1)
        dC, ws, labs = sel["dC"], sel["w_site"], cl["labels"]
        for a in range(len(pos)):
            for b in range(a + 1, len(pos)):
                v = dC[a, b]
                if abs(v) > 0.02:
                    ax.plot(*pos[[a, b]].T, color="tab:red" if v > 0 else "tab:blue",
                            lw=0.5 + 5 * abs(v), alpha=0.75, zorder=3)
        wmax = max(ws.max(), 1e-12)
        ax.scatter(*pos.T, s=30 + 300 * ws / wmax, facecolors="none", edgecolors="k", lw=1.0, zorder=4)
        for a, lab in enumerate(labs):
            ax.annotate(f"{lab} {ws[a]:.2f}", pos[a], xytext=(4, 5), textcoords="offset points", fontsize=7, zorder=5)
        ax.set_title(f"0 -> {sel['f_level'][0]}: {1e3 * sel['dE']:.4f} meV, S {sel['S0']:.1f} -> {sel['Sf']:.1f}, "
                     f"{'allowed' if sel['allowed'] else 'forbidden'}, {sel['kind']}, W = {sel['total']:.3f}\n"
                     "orange: spin-flip map (inelastic dI/dV step); circles: weight per site;\n"
                     "lines: change of <S_i.S_j> (red more FM, blue more AFM)", fontsize=8)

    def sm_draw(self):
        for ax in (self.ax_c4, self.ax_e4, self.ax_m4, self.ax_b4):
            ax.clear()
        st = self.st
        cl, res, fit, mf = self.sm["cluster"], self.sm["ed"], self.sm["fit"], self.sm["mfh"]
        m = self.model
        # cluster with spin correlations
        ax = self.ax_c4
        if cl is not None and res is not None and m is not None:
            ns, nc = cl["ns"], cl["ncells"]
            ax.add_collection(self._chain_bonds(st, range(nc), 0.5, "0.75", shown=tm.cluster_atoms_shown(cl)))
            pos = self._cluster_pos(m, cl)
            sel = self.sm.get("sel")
            if sel is not None:
                self._sm_draw_transition(ax, sel, cl, pos)
            C = res["C"][0] if sel is None else np.zeros((0, 0))
            for a in range(len(C)):
                for b in range(a + 1, len(pos)):
                    v = C[a, b]
                    if abs(v) > 0.02:
                        ax.plot(*pos[[a, b]].T, color="tab:blue" if v < 0 else "tab:red",
                                lw=0.5 + 6 * abs(v), alpha=0.7, zorder=3)
            if sel is None:
                ax.scatter(*pos.T, s=400 * np.diag(C), color="k", zorder=4)
                for a, lab in enumerate(cl["labels"]):
                    ax.annotate(lab, pos[a], xytext=(4, 5), textcoords="offset points", fontsize=7, zorder=5)
                ax.set_title("ground state <S_i.S_j>: blue AFM, red FM (width ~ |value|); dot size ~ <S_i^2>",
                             fontsize=8)
            ax.set_aspect("equal")
            ax.autoscale_view()
        # spectrum vs S
        ax = self.ax_e4
        if res is not None:
            E = 1e3 * (res["E"] - res["E"][0])
            ax.scatter(res["S"], E, marker="_", s=400, color="k", label="Hubbard ED")
            if "lw" in res:
                # spin-flip weight of the transition from the ground state (inelastic intensity)
                tot = res["lw"].sum(axis=1)[1:]                 # without the ground level (elastic)
                idx = [L[0] for L in res["levels"]][1:]
                if tot.max(initial=0) > 1e-4:
                    ax.scatter(res["S"][idx] - 0.14, E[idx], s=120 * tot / tot.max(), color="tab:orange",
                               alpha=0.8, zorder=3, label="spin-flip weight")
            sel = self.sm.get("sel")
            if sel is not None:
                k = sel["f_level"][0]
                ax.scatter([res["S"][k]], [E[k]], marker="s", s=90, facecolors="none", edgecolors="tab:orange",
                           lw=1.8, zorder=4)
                ax.annotate("", xy=(res["S"][k], E[k]), xytext=(res["S"][0], 0.0),
                            arrowprops=dict(arrowstyle="->", color="tab:orange", lw=1.2), zorder=4)
            if fit is not None:
                ax.scatter(fit["Sm"] + 0.12, 1e3 * fit["Em"][:fit["M"]] if len(fit["Em"]) >= fit["M"] else 1e3 * fit["Em"],
                           marker="x", color="tab:red", label="spin model")
                emax = 1e3 * (res["E"][fit["M"] - 1] - res["E"][0])      # top of the spin manifold
                ax.set_ylim(-0.15 * emax - 1e-3, 1.3 * emax + 1e-3)
                if len(res["E"]) > fit["M"]:
                    ax.text(0.02, 0.97, f"next state at +{1e3 * (res['E'][fit['M']] - res['E'][0]):.2f} meV (off scale)",
                            transform=ax.transAxes, fontsize=7, va="top")
            Mm = res.get("M", 0)
            if fit is None and Mm and len(res["E"]) > Mm:
                spread = E[Mm - 1]
                if E[Mm] > 3 * spread and spread > 0:          # isolated spin manifold: zoom on it
                    ax.set_ylim(-0.15 * spread - 1e-3, 1.3 * spread + 1e-3)
                    ax.text(0.02, 0.97, f"next state at +{E[Mm]:.2f} meV (off scale)",
                            transform=ax.transAxes, fontsize=7, va="top")
            ax.set_xlabel("total spin S")
            ax.set_ylabel("E - E0 (meV)")
            ax.set_title("many-body spectrum (click a level)", fontsize=9)
            ax.legend(fontsize=7)
        # mean-field spin density and bands
        if mf is not None:
            sol = mf["af"] if mf["af"]["E"] <= mf["fm"]["E"] else mf["fm"]
            which = "AFM" if sol is mf["af"] else "FM"
            mz = sol["n"][0] - sol["n"][1]
            ax = self.ax_m4
            ax.add_collection(self._chain_bonds(st, (-1, 0, 1), 0.4, "0.6"))
            for c in (-1, 0, 1):
                P = st.xy + c * st.T
                ax.scatter(*P.T, s=4000 * np.abs(mz) / 4 + 1, c=np.where(mz > 0, "tab:red", "tab:blue"),
                           alpha=0.8 if c == 0 else 0.35, zorder=3)
            ax.set_aspect("equal")
            ax.autoscale_view()
            ax.set_title(f"mean-field spin density of the lower solution ({which}): red up, blue down", fontsize=8)
            ax = self.ax_b4
            ks = sol["ks"] / np.pi
            for s_, ls, col, lw in ((0, "-", "tab:red", 2.2), (1, (0, (4, 2.5)), "tab:blue", 1.8)):
                B = sol["bands"][s_] - sol["Ef"].mean()
                if len(ks) > 1:
                    order = np.argsort(np.where(ks > 1, ks - 2, ks))
                    ax.plot(np.where(ks > 1, ks - 2, ks)[order], B[order], linestyle=ls, color=col, lw=lw)
                else:
                    ax.hlines(B[0], -0.3 + 0.3 * s_, 0.3 * s_, color=col, lw=lw)
            ax.set_ylim(-1.5, 1.5)
            ax.axhline(0, color="k", lw=0.5, ls=":")
            ax.set_xlabel("k (pi / |T|)")
            ax.set_ylabel("E - E_F (eV)")
            ax.set_title("mean-field bands: up (red), down (blue dashed)", fontsize=8)
        self.fig4.tight_layout()
        self.can4.draw_idle()



    # ================================================================== tab 5: mean-field Hubbard
    def _build_tabMF(self):
        self.mf = dict(results=None, par=None, sel=0)
        left = self._scrollable(self.tabMF, width=540)
        box = ttk.LabelFrame(left, text="Atomistic mean-field Hubbard (all carbons, no sites to choose)", padding=6)
        box.pack(fill="x")
        ttk.Label(box, text="H_s = H_TB + U sum_x <n_x,-s> n_x,s\n"
                            "E = sum_occ e - U sum_x <n_x,up><n_x,dn>\n"
                            "H_TB: t1 t2 t3 of tab 1; collinear, self-consistent, Pulay mixing",
                  foreground="#1f4e79", font=("Courier", 9)).pack(anchor="w")
        row = ttk.Frame(box)
        row.pack(fill="x", pady=(2, 2))
        self.mf_sys = tk.StringVar(value="periodic")
        ttk.Radiobutton(row, text="periodic chain (tab 1, k points)", value="periodic",
                        variable=self.mf_sys).pack(side="left")
        ttk.Radiobutton(row, text="finite chain / molecule (tab 3)", value="finite",
                        variable=self.mf_sys).pack(side="left")
        self.mf_U, self.mf_nk = self._entry2(box, "U (eV)", "3.0", "k points", "24")
        self.mf_kT = self._entry(box, "kT (eV)", "0.005")
        self.mf_sz = self._entry(box, "S_z to try (per cell)", "auto")
        ttk.Label(box, text="auto: the smallest allowed S_z, the sublattice imbalance |N_A - N_B|/2\n"
                            "(Lieb) and one more. For every S_z two starts are run, spins by\n"
                            "sublattice and all spins up, and the lower solution is kept; a\n"
                            "non-magnetic solution is always added. Click a state in the energy\n"
                            "plot to show its spin density and bands.\n"
                            "Finite chain: the structure built in tab 3 (no k points, S_z of the\n"
                            "whole piece), its levels instead of bands, and the singlet-triplet\n"
                            "gap: raw E(S_z+1) - E(S_z), spin-projected (Yamaguchi) and, with\n"
                            "more than two radicals, from the Heisenberg mapping of the collinear\n"
                            "states (extra runs with the moments of some radicals flipped).",
                  foreground="gray").pack(anchor="w")
        self.mf_btn = ttk.Button(box, text="Run mean-field Hubbard", command=self.mf_run)
        self.mf_btn.pack(fill="x", pady=(4, 0))

        box = ttk.LabelFrame(left, text="Spatial projection: dI/dV maps and spectra of the state shown",
                             padding=6)
        box.pack(fill="x", pady=(6, 0))
        row = ttk.Frame(box)
        row.pack(fill="x")
        ttk.Label(row, text="map:").pack(side="left")
        self.mf_map = tk.StringVar(value=MAP_SPIN)
        cb = ttk.Combobox(row, textvariable=self.mf_map, state="readonly", width=40,
                          values=[MAP_SPIN, "dI/dV up + down at the map energy", MAP_UP, MAP_DN, MAP_UPDN,
                                  MAP_BAND, MAP_BANDK])
        cb.pack(side="left", padx=2)
        cb.bind("<<ComboboxSelected>>", lambda e: self.mf_draw(keep_view=True))
        self.mf_emin, self.mf_emax = self._entry2(box, "E-E_F  min", "-1.5", "max (eV)", "1.5")
        self.mf_broad, self.mf_emap = self._entry2(box, "broad. (meV)", "auto", "map E - E_F", "0.0")
        self.mf_tip = self._entry(box, "tip resolution (A)", "0.9")
        ttk.Label(box, text="H_s = H_TB + U diag(<n_-s>) of the state shown, on a dense k grid\n"
                            "(finite: its levels). Click the bands / levels / DOS: map energy and\n"
                            "band (spin of the band clicked; finite: left column up, right down);\n"
                            "click atoms on the map: spin-resolved spectra (up solid, down dashed).\n"
                            "Zoom/pan with the toolbar: DOS, map and spectra are recomputed for\n"
                            "the window shown; broadening 'auto' = window / 250 (0.1 to 20 meV).",
                  foreground="gray").pack(anchor="w")
        ttk.Button(box, text="Export spin-resolved LDOS (CSV)...", command=self.mf_export).pack(fill="x", pady=(2, 0))
        self.mf_out = tk.Text(left, width=68, height=26, font=("Courier", 9))
        self.mf_out.pack(fill="both", expand=True, pady=(6, 0))

        right = ttk.Frame(self.tabMF)
        right.pack(side="left", fill="both", expand=True)
        self.figMF = Figure(figsize=(10, 8))
        gs = self.figMF.add_gridspec(2, 4, width_ratios=[1, 1, 0.45, 1.15])
        self.ax_mfE = self.figMF.add_subplot(gs[0, 0])
        self.ax_mfm = self.figMF.add_subplot(gs[0, 1:])
        self.ax_mfb = self.figMF.add_subplot(gs[1, :2])
        self.ax_mfd = self.figMF.add_subplot(gs[1, 2], sharey=self.ax_mfb)
        self.ax_mfp = self.figMF.add_subplot(gs[1, 3])
        self.canMF = FigureCanvasTkAgg(self.figMF, right)
        self.tbMF = NavigationToolbar2Tk(self.canMF, right)
        self.tbMF.update()
        self.canMF.get_tk_widget().pack(fill="both", expand=True)
        self._winMF = self._bandMF = self._specMF = self._mf_cache = None
        self.mf_atoms = []
        self._on_clicks(self.canMF, self.mf_click)
        self._watch_view(self.canMF, "tabMF", lambda: (*self.ax_mfb.get_ylim(), *self.ax_mfp.get_xlim()),
                         self._zoomMF)

    def _mf_log(self, text, clear=False):
        if clear:
            self.mf_out.delete("1.0", "end")
        self.mf_out.insert("end", text + "\n")
        self.mf_out.see("end")

    def mf_run(self):
        finite = self.mf_sys.get() == "finite"
        try:
            U, nk, kT = float(self.mf_U.get()), int(self.mf_nk.get()), float(self.mf_kT.get())
            txt = self.mf_sz.get().strip().lower()
            sz = None if txt in ("", "auto") else [int(round(2 * float(x))) for x in txt.replace(",", " ").split()]
            if finite:
                if self.fst is None:
                    raise ValueError("build a finite chain (or load a finite molecule) in tab 3 first")
                st = self.fst
                H, nk = {0: tb.hoppings(st, self.t_values())[0]}, 1
            else:
                if self.st is None:
                    raise ValueError("load a structure in tab 1 first")
                if self.H is None:
                    self.calc_bands()
                H, st = self.H, self.st
        except Exception as e:
            messagebox.showerror("Mean field", str(e))
            return

        def work():
            res = tm.mfh_scan(H, st, U, nk, kT, sz)
            if not finite:
                return res, None
            gap = tm.mfh_spin_gap(H[0], st, res, U, kT)
            return tm.mfh_states(res, gap), gap

        def done(out):
            res, gap = out
            if finite:
                self.mf.update(results=res, par=(U, nk, kT), sel=0, finite=True, st=st, H0=H[0], gap=gap, H=H)
                self._mf_cache = self._specMF = self._bandMF = None
                if self.mf_atoms and max(self.mf_atoms) >= st.N:
                    self.mf_atoms = []
                L = [f"mean-field Hubbard of the FINITE system of tab 3 ({st.name}): {st.N} atoms, "
                     f"{st.n_electrons} electrons, U = {U} eV on every carbon, kT = {kT} eV"]
                bal = tb.sublattice_balance(st)
                L.append(f"sublattices A/B = {bal[0]}/{bal[1]} (Lieb: S = {abs(bal[0] - bal[1]) / 2:g} for a "
                         "bipartite lattice)" if bal else "not bipartite")
                L += [""] + self._mf_table(res, U, kT) + [""] + tm.spin_gap_report(gap)
                L += ["", "(the U fit of tab 6 needs the periodic mean field: run this tab in 'periodic chain' mode "
                          "for it)"]
                self._mf_log("\n".join(L), clear=True)
                if hasattr(self, "u5_states"):
                    self.u5_list_states()
                self.mf_draw()
                return
            self.mf.update(results=res, par=(U, nk, kT), sel=0, finite=False, st=st, H0=None, gap=None, H=H)
            self._mf_cache = self._specMF = self._bandMF = None
            if self.mf_atoms and max(self.mf_atoms) >= st.N:
                self.mf_atoms = []
            col = tb.sublattice_colors(st)
            E0 = res[0]["sol"]["E"]
            tv = self.t_values()
            L = [f"mean-field Hubbard, U = {U} eV on every carbon, {nk} k points, kT = {kT} eV",
                 f"H_TB: {st.N} atoms per cell, " + ", ".join(f"t{k + 1} = {t:+.3f} eV" for k, t in enumerate(tv)),
                 (f"sublattices A/B = {(col == 0).sum()}/{(col == 1).sum()} per cell (Lieb: S = "
                  f"{abs((col == 0).sum() - (col == 1).sum()) / 2:g} for a bipartite lattice)" if col is not None
                  else "not bipartite"),
                 "", f"  {'state':34s}{'E - E_min (meV)':>17s}{'max |m|':>9s}",
                 ]
            for o in res:
                L.append(f"  {o['label']:34s}{1e3 * (o['sol']['E'] - E0):17.3f}{o['mmax']:9.3f}"
                         + ("" if o["sol"]["converged"] else "  NOT converged"))
            L += ["", f"lowest: {res[0]['label']}. Mean field breaks the spin symmetry and usually overestimates "
                      "the magnetic energies; use tab 6 to fit the U of the reduced model to these states and "
                      "tab 7 for the exact diagonalisation."]
            self._mf_log("\n".join(L), clear=True)
            if hasattr(self, "u5_states"):
                self.u5_list_states()
            self.mf_draw()

        self._mf_log("running the mean field ...", clear=True)
        self._sm_background(work, done, self.mf_btn)

    def _mf_channels(self, o):
        """Spin channels of the mean-field state o: [(cache, reference, electrons, colour, label)],
        each spin with H_s = H_TB + U diag(<n_-s>) on a dense k grid (finite: its levels)."""
        key = id(o)
        if self._mf_cache is None or self._mf_cache[0] != key:
            self.config(cursor="watch")
            self.update_idletasks()
            try:
                sol, U = o["sol"], self.mf["par"][0]
                n = sol["n"]
                cs = []
                for s_ in (0, 1):
                    Hs = dict(self.mf["H"]) if not self.mf.get("finite") else {0: self.mf["H0"]}
                    Hs[0] = Hs[0] + np.diag(U * n[1 - s_])
                    cs.append(tb.bloch_cache(Hs))
            finally:
                self.config(cursor="")
            self._mf_cache = (key, cs)
            self._specMF = None
        sol, ref = o["sol"], tm.midgap(o["sol"])
        cu, cd = self._mf_cache[1]
        return [(cu, ref, sol["n_up"], "tab:red", "up"), (cd, ref, sol["n_dn"], "tab:blue", "down")]

    def _zoomMF(self):
        if not self.mf.get("results"):
            return
        self._winMF = self._new_window(self._winMF, self.ax_mfb.get_ylim(), self.ax_mfp.get_xlim())
        self.mf_draw(keep_view=True)

    def _mf_mode(self, finite):
        mode = self.mf_map.get()
        if finite and mode in (MAP_BAND, MAP_BANDK):
            return MAP_LEVEL
        if mode.startswith("dI/dV up + down"):
            return MAP_LDOS
        return mode

    def mf_click(self, ev):
        res = self.mf["results"]
        if res is None or ev.inaxes is None or ev.button != 1 or ev.xdata is None:
            return
        if ev.inaxes is self.ax_mfE:
            pts = self.ax_mfE.transData.transform(np.c_[[o["two_sz"] / 2 for o in res],
                                                        [1e3 * (o["sol"]["E"] - res[0]["sol"]["E"]) for o in res]])
            d = np.hypot(pts[:, 0] - ev.x, pts[:, 1] - ev.y)
            if d.min() < 25:
                self.mf["sel"] = int(np.argmin(d))
                self._bandMF = None
                self.mf_draw(keep_view=True)
            return
        st = self.mf.get("st") or self.st
        finite = self.mf.get("finite", False)
        o = res[self.mf["sel"]]
        if ev.inaxes is self.ax_mfm:
            cells = (0,) if finite else (-1, 0, 1)
            d = np.min([np.linalg.norm(st.xy + c * st.T - [ev.xdata, ev.ydata], axis=1) for c in cells], axis=0)
            i = int(np.argmin(d))
            if d[i] > 0.7:
                return
            self.mf_atoms.remove(i) if i in self.mf_atoms else self.mf_atoms.append(i)
            self.mf_atoms = self.mf_atoms[-8:]
            m = o["sol"]["n"][0][i] - o["sol"]["n"][1][i]
            self.status.set(f"mean field: atom {i}  x = {st.xy[i][0]:.3f}  y = {st.xy[i][1]:.3f} A,  "
                            f"n_up = {o['sol']['n'][0][i]:.4f}  n_dn = {o['sol']['n'][1][i]:.4f}  m = {m:+.4f}")
            self.mf_draw(keep_view=True)
            return
        if ev.inaxes not in (self.ax_mfb, self.ax_mfd, self.ax_mfp):
            return
        ch = self._mf_channels(o)
        E = ev.xdata if ev.inaxes is self.ax_mfp else ev.ydata
        k = None
        if ev.inaxes is self.ax_mfb and not finite:
            k = float(np.clip(ev.xdata, -1, 1) * np.pi)
            ik = [int(np.argmin(np.abs(c["ks"] - abs(k)))) for c, *_ in ch]
            d = [np.abs(c["E"][j] - ref - E) for (c, ref, *_), j in zip(ch, ik)]
            s_ = int(np.argmin([x.min() for x in d]))
            n_ = int(np.argmin(d[s_]))
            Eb = ch[s_][0]["E"][ik[s_], n_] - ch[s_][1]
        else:
            if ev.inaxes is self.ax_mfb and finite:
                s_ = 0 if ev.xdata < 0.5 else 1
            else:
                s_ = int(np.argmin([np.abs(c["E"] - ref - E).min() for c, ref, *_ in ch]))
            c, ref = ch[s_][0], ch[s_][1]
            n_ = tb.cache_nearest_band(c, E, shift=ref)
            j = int(np.argmin(np.abs(c["E"][:, n_] - ref - E)))
            k = None if finite else float(c["ks"][j])
            Eb = c["E"][j, n_] - ref
        self._bandMF = (s_, n_, k)
        mode = self._mf_mode(finite)
        if mode in (MAP_BAND, MAP_BANDK, MAP_LEVEL):
            E = Eb
        elif mode == MAP_SPIN:
            self.mf_map.set("dI/dV up + down at the map energy")
        self.mf_emap.set(f"{E:.4f}")
        what = "level" if finite else "band"
        self.status.set(f"{what} {n_ - ch[s_][2] + 1:+d} (0 = highest occupied) spin {ch[s_][4]}"
                        + ("" if finite else f" at k = {abs(k) / np.pi:.3f} pi/|T|") + f": E - E_F = {Eb:+.4f} eV")
        self.mf_draw(keep_view=True)

    def mf_draw(self, keep_view=False):
        res = self.mf["results"]
        keep = bool(keep_view) and self._winMF is not None
        axes = (self.ax_mfE, self.ax_mfm, self.ax_mfb, self.ax_mfd, self.ax_mfp)
        saved = self._get_lims([a for a in (self.ax_mfE, self.ax_mfm, self.ax_mfb) if a.has_data()]) if keep else {}
        if not keep:
            try:
                self._winMF = (float(self.mf_emin.get()), float(self.mf_emax.get()))
            except ValueError:
                self._winMF = (-1.5, 1.5)
            self.tbMF.update()
        W = tuple(self._winMF)
        for ax in axes:
            ax.clear()
        if not res:
            self.canMF.draw_idle()
            return
        st = self.mf.get("st") or self.st
        finite = self.mf.get("finite", False)
        gap = self.mf.get("gap")
        E0 = res[0]["sol"]["E"]
        ax = self.ax_mfE
        if finite and gap is not None:
            for key, txt in (("LS", "low spin"), ("HS", "high spin")):
                o = gap[key]
                ax.annotate(txt, (o["two_sz"] / 2, 1e3 * (o["sol"]["E"] - E0)), xytext=(6, -3),
                            textcoords="offset points", fontsize=7, color="tab:purple")
        for k, o in enumerate(res):
            mag = o["mmax"] > 0.05
            ax.scatter(o["two_sz"] / 2, 1e3 * (o["sol"]["E"] - E0), s=90 if k == self.mf["sel"] else 45,
                       marker="o" if mag else "s", color="tab:red" if mag else "0.5",
                       edgecolor="k" if k == self.mf["sel"] else "none", zorder=3)
        # zoom on the magnetic states when a non-magnetic one is far above them
        Em_ = np.array([1e3 * (o["sol"]["E"] - E0) for o in res])
        mag_ = np.array([o["mmax"] > 0.05 for o in res])
        if mag_.any() and (~mag_).any():
            top = Em_[mag_].max()
            far = Em_[~mag_] > 3 * top + 1
            if top > 0 and far.any():
                ax.set_ylim(-0.12 * top - 0.5, 1.3 * top + 0.5)
                ax.text(0.02, 0.98, "off scale: " + ", ".join(f"non-magnetic at +{e:.0f} meV" for e in Em_[~mag_][far]),
                        transform=ax.transAxes, fontsize=7, va="top")
        ax.set_xlabel("S_z" if finite else "S_z per cell")
        ax.set_ylabel("E - E_min (meV)" if finite else "E - E_min (meV / cell)")
        title = "states (circle: magnetic, square: non-magnetic);\nclick one to show it"
        if finite and gap is not None and not gap["closed_shell"]:
            best = gap["heis"]["gap_up"] if (gap["heis"] is not None and gap["n_centres"] > 2) else gap["yamaguchi"]
            title += f"\nspin gap (projected): {1e3 * best:.2f} meV"
        ax.set_title(title, fontsize=8)

        o = res[self.mf["sel"]]
        sol = o["sol"]
        ch = self._mf_channels(o)
        try:
            Em = float(self.mf_emap.get())
        except ValueError:
            Em = 0.0
        sig, auto = self._sigma(self.mf_broad, W)
        stxt = self._sigma_text(sig, auto)
        sel = [i for i in self.mf_atoms if i < st.N]
        key = (id(o), W, sig, tuple(sel))
        if self._specMF is None or self._specMF[0] != key:
            x = self._egrid(W, sig)
            self._specMF = (key, x, [tb.cache_spectra(c, x, sig, sel, shift=ref) for c, ref, *_ in ch])
        _, x, spec = self._specMF
        mode = self._mf_mode(finite)
        band = self._bandMF
        if mode in (MAP_BAND, MAP_BANDK, MAP_LEVEL) and band is None:
            s_ = 0
            c, ref = ch[s_][0], ch[s_][1]
            n_ = tb.cache_nearest_band(c, Em, shift=ref)
            j = int(np.argmin(np.abs(c["E"][:, n_] - ref - Em)))
            band = (s_, n_, None if finite else float(c["ks"][j]))

        # map
        ax = self.ax_mfm
        cells = [0] if finite else [-1, 0, 1]
        w = float(self.mf_tip.get() or 0)
        if mode == MAP_SPIN:
            mz = sol["n"][0] - sol["n"][1]
            if finite:
                ax.add_collection(self._bonds(st, 0.5, "0.6"))
            else:
                ax.add_collection(self._chain_bonds(st, (-1, 0, 1), 0.4, "0.6"))
            for c_ in cells:
                P = st.xy + c_ * st.T
                ax.scatter(*P.T, s=4000 * np.abs(mz) / 4 + 1, c=np.where(mz > 0, "tab:red", "tab:blue"),
                           alpha=0.8 if c_ == 0 else 0.35, zorder=3)
            ax.set_aspect("equal")
            ax.autoscale_view()
            title = f"{o['label']}: spin density n_up - n_dn (red up, blue down); max |m| = {o['mmax']:.3f}"
        else:
            signed = False
            if mode in (MAP_BAND, MAP_BANDK, MAP_LEVEL):
                s_, n_, kb = band
                c = ch[s_][0]
                v = tb.cache_band_map(c, n_, kb if mode == MAP_BANDK else None)
                lab = "level" if finite else "band"
                where = "" if finite else (f" at k = {abs(kb) / np.pi:.3f} pi/|T|" if mode == MAP_BANDK
                                           else " averaged over k")
                title = (f"{o['label']}: {lab} {n_ - ch[s_][2] + 1:+d} (0 = highest occupied) spin {ch[s_][4]}"
                         f"{where}: |psi|^2, tip {w:g} A")
            else:
                maps = [tb.cache_map(c, Em, sig, shift=ref) for c, ref, *_ in ch]
                if mode == MAP_UP:
                    v, what = maps[0], "dI/dV spin up"
                elif mode == MAP_DN:
                    v, what = maps[1], "dI/dV spin down"
                elif mode == MAP_UPDN:
                    v, what, signed = maps[0] - maps[1], "dI/dV up - down (red up)", True
                else:
                    v, what = maps[0] + maps[1], "dI/dV up + down"
                title = f"{o['label']}: {what} at E - E_F = {Em:+.4f} eV ({stxt}, tip {w:g} A)"
            self._draw_map(ax, st, v, cells, w, signed=signed,
                           unit="" if mode in (MAP_BAND, MAP_BANDK, MAP_LEVEL) else "states/eV")
        for k_, i in enumerate(sel):
            ax.scatter(*st.xy[i], s=60, facecolor="none", edgecolor=COLORS[k_ % len(COLORS)], linewidth=2, zorder=5)
        ax.set_title(title, fontsize=8)

        # bands / levels
        ax = self.ax_mfb
        if finite:
            for s_, (c, ref, nel, col, lab) in enumerate(ch):
                x0 = 0.0 if s_ == 0 else 0.55
                e = c["E"][0] - ref
                occ = np.arange(len(e)) < nel
                ax.hlines(e[occ], x0, x0 + 0.45, color=col, lw=1.6)
                ax.hlines(e[~occ], x0, x0 + 0.45, color=col, lw=0.8, alpha=0.4)
            if mode == MAP_LEVEL:
                s_, n_, _ = band
                x0 = 0.0 if s_ == 0 else 0.55
                ax.hlines(ch[s_][0]["E"][0, n_] - ch[s_][1], x0, x0 + 0.45, color="tab:orange", lw=3, zorder=3)
            ax.set_xlim(0, 1)
            ax.set_xticks([0.22, 0.77])
            ax.set_xticklabels(["up", "down"])
            ax.set_xlabel("levels (faint: empty)")
            ax.set_title("spin-resolved levels of the finite system", fontsize=8)
        else:
            for (c, ref, nel, col, lab), ls, lw in zip(ch, ("-", (0, (4, 2.5))), (2.2, 1.8)):
                kx = np.concatenate([-c["ks"][::-1], c["ks"][1:]]) / np.pi
                B = np.concatenate([c["E"][::-1], c["E"][1:]]) - ref
                ax.plot(kx, B, linestyle=ls, color=col, lw=lw)
            if mode in (MAP_BAND, MAP_BANDK):
                s_, n_, kb = band
                c, ref = ch[s_][0], ch[s_][1]
                kx = np.concatenate([-c["ks"][::-1], c["ks"][1:]]) / np.pi
                B = np.concatenate([c["E"][::-1, n_], c["E"][1:, n_]]) - ref
                ax.plot(kx, B, color="tab:orange", lw=3.2, alpha=0.8, zorder=3)
                if mode == MAP_BANDK and kb is not None:
                    j = int(np.argmin(np.abs(c["ks"] - abs(kb))))
                    ax.plot([kb / np.pi], [c["E"][j, n_] - ref], "o", ms=7, mfc="none", mec="tab:orange", mew=2,
                            zorder=4)
            ax.set_xlim(-1, 1)
            ax.set_xlabel("k (pi / |T|)")
            ax.set_title("spin-resolved bands: up red, down blue dashed", fontsize=8)
        if mode not in (MAP_SPIN, MAP_BAND, MAP_BANDK, MAP_LEVEL):
            ax.axhline(Em, color="tab:green", lw=0.9)
        ax.axhline(0, color="k", lw=0.5, ls=":")
        ax.set_ylabel("E - E_F (eV)")
        ax.set_ylim(*W)

        # spin DOS
        ax = self.ax_mfd
        for (dos, L), (c, ref, nel, col, lab), sign in zip(spec, ch, (1, -1)):
            ax.fill_betweenx(x, 0, sign * dos, color=col, alpha=0.5)
        m = max(max(d.max() for d, L in spec), 1e-9) * 1.05
        ax.set_xlim(-m, m)
        ax.axvline(0, color="k", lw=0.5)
        ax.axhline(0, color="k", lw=0.5, ls=":")
        if mode not in (MAP_SPIN, MAP_BAND, MAP_BANDK, MAP_LEVEL):
            ax.axhline(Em, color="tab:green", lw=0.9)
        ax.set_xlabel("DOS down | up")
        ax.set_xticks([])
        ax.tick_params(labelleft=False)
        ax.set_title(f"spin DOS ({stxt})", fontsize=8)
        ax.set_ylim(*W)

        # spin-resolved spectra
        ax = self.ax_mfp
        for (dos, L), (c, ref, nel, col, lab), ls in zip(spec, ch, ("-", "--")):
            ax.plot(x, dos / st.N, color=col, lw=0.8, alpha=0.5, ls=ls, label=f"average {lab}")
        for k_, i in enumerate(sel):
            col = COLORS[k_ % len(COLORS)]
            ax.plot(x, spec[0][1][k_], color=col, lw=1.4, label=f"atom {i} up")
            ax.plot(x, spec[1][1][k_], color=col, lw=1.4, ls="--", label=f"atom {i} down")
        ax.axvline(0, color="k", lw=0.5, ls=":")
        if mode not in (MAP_SPIN, MAP_BAND, MAP_BANDK, MAP_LEVEL):
            ax.axvline(Em, color="tab:green", lw=0.9)
        ax.set_xlim(*W)
        ax.set_xlabel("E - E_F (eV)")
        ax.set_ylabel("LDOS per spin (states / eV)")
        ax.set_title("spectra: click atoms on the map", fontsize=8)
        ax.legend(fontsize=6, loc="upper right")
        self._set_lims(saved, only=(self.ax_mfE, self.ax_mfm))
        self._set_lims(saved, y=False, only=(self.ax_mfb,))
        self.ax_mfb.set_ylim(*W)
        self.figMF.tight_layout()
        self.canMF.draw_idle()
        self._views["tabMF"] = (*self.ax_mfb.get_ylim(), *self.ax_mfp.get_xlim())

    def mf_export(self):
        res = self.mf.get("results")
        if not res:
            return
        p = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV", "*.csv")])
        if not p:
            return
        o = res[self.mf["sel"]]
        st = self.mf.get("st") or self.st
        W = self._winMF or (float(self.mf_emin.get()), float(self.mf_emax.get()))
        sig, auto = self._sigma(self.mf_broad, W)
        x = self._egrid(W, sig)
        cols = [x]
        for c, ref, *_ in self._mf_channels(o):
            dos, L = tb.cache_spectra(c, x, sig, range(st.N), shift=ref)
            cols += [dos, L.T]
        np.savetxt(p, np.column_stack(cols), delimiter=",",
                   header=f"mean-field state {o['label']}: E - E_F (eV), DOS up, LDOS up of atom 0, 1, ..., "
                          f"DOS down, LDOS down of atom 0, 1, ... (states / eV); Gaussian broadening "
                          f"{1e3 * sig:.4g} meV")
        self.status.set(f"saved {p}")


    # ================================================================== tab 6: U from the mean field
    def _build_tab5(self):
        self.u5 = dict(rows=[], refs=None, fit=None, groups=None, model=None, states=[], used=[])
        left = self._scrollable(self.tab5, width=540)
        blue = dict(foreground="#1f4e79", font=("Courier", 9))

        box = ttk.LabelFrame(left, text="1. Effective model of tab 2 (t fixed)", padding=6)
        box.pack(fill="x")
        ttk.Label(box, text="Steps: tight binding (tab 1) -> t of the reduced model (tab 2 fit) -> "
                            "atomistic mean-field Hubbard (tab 5) -> U_i of the reduced model fitted to "
                            "those mean-field states with t fixed (here). The exact diagonalisation with t "
                            "and U_i is in tab 7.",
                  foreground="gray", wraplength=500, justify="left").pack(anchor="w")
        self.u5_modeltxt = tk.Text(box, width=68, height=6, font=("Courier", 9), background="#f4f4f4")
        self.u5_modeltxt.pack(fill="x", pady=(2, 4))
        self.u5_modeltxt.insert("1.0", "(press 'Load model from tab 2')")
        self.u5_modeltxt.configure(state="disabled")
        ttk.Button(box, text="Load model from tab 2", command=self.u5_load).pack(fill="x")
        ttk.Label(box, text="U group: sites with the same name share one U (default: sites with the same\n"
                            "on-site parameter)", foreground="gray").pack(anchor="w")
        self.u5_table = ttk.Frame(box)
        self.u5_table.pack(fill="x")

        box = ttk.LabelFrame(left, text="2. Reference: mean-field states of tab 5", padding=6)
        box.pack(fill="x", pady=(6, 0))
        ttk.Label(box, text="the U_i are fitted to the ticked states (magnetic ones are ticked by default)",
                  foreground="gray").pack(anchor="w")
        self.u5_states = ttk.Frame(box)
        self.u5_states.pack(fill="x")
        ttk.Button(box, text="Refresh the list from tab 5", command=self.u5_list_states).pack(fill="x", pady=(2, 0))

        box = ttk.LabelFrame(left, text="3. Fit U_i of the reduced model (t fixed)", padding=6)
        box.pack(fill="x", pady=(6, 0))
        ttk.Label(box, text="H_s(k) = H_t(k) + sum_i U_i (<n_i,-s> - 1/2) n_i,s   self-consistent,\n"
                            "same S_z per cell as the atomistic state; fitted: its spin-up and\n"
                            "spin-down bands (+ one rigid offset per state) and the site moments",
                  **blue).pack(anchor="w")
        self.u5_fitm = tk.BooleanVar(value=True)
        ttk.Checkbutton(box, text="fit the site moments too (0.1 eV per unit moment)",
                        variable=self.u5_fitm).pack(anchor="w")
        self.u5_btn_fit = ttk.Button(box, text="Fit U to the ticked mean-field states", command=self.u5_fit)
        self.u5_btn_fit.pack(fill="x", pady=(4, 0))

        row = ttk.Frame(box)
        row.pack(fill="x", pady=(2, 0))
        ttk.Button(row, text="Use these U_i in tab 7 (ED)", command=self.u5_to_tab6).pack(side="left", fill="x", expand=True)
        ttk.Button(row, text="Use them in tab 4 (spin model)", command=self.u5_to_tab4).pack(side="left", fill="x", expand=True)

        self.u5_out = tk.Text(left, width=68, height=22, font=("Courier", 9))
        self.u5_out.pack(fill="both", expand=True, pady=(6, 0))

        right = ttk.Frame(self.tab5)
        right.pack(side="left", fill="both", expand=True)
        self.fig5 = Figure(figsize=(10, 8))
        gs = self.fig5.add_gridspec(2, 2)
        self.ax_f5 = self.fig5.add_subplot(gs[0, 0])
        self.ax_a5 = self.fig5.add_subplot(gs[0, 1])
        self.ax_c5 = self.fig5.add_subplot(gs[1, 0])
        self.ax_e5 = self.fig5.add_subplot(gs[1, 1])
        self.can5 = FigureCanvasTkAgg(self.fig5, right)
        NavigationToolbar2Tk(self.can5, right).update()
        self.can5.get_tk_widget().pack(fill="both", expand=True)

    def _u5_log(self, text, clear=False):
        if clear:
            self.u5_out.delete("1.0", "end")
        self.u5_out.insert("end", text + "\n")
        self.u5_out.see("end")

    # ------------------------------------------------------------------ 1. model
    def u5_load(self):
        try:
            m, r = self._sm_model()
            w, _, smin = tm.wannier_orbitals(self.H, self.st, m)
        except Exception as e:
            messagebox.showerror("U from mean field", str(e))
            return
        self._sm_show_model(m, r, widget=self.u5_modeltxt)
        ipr, n90 = tm.orbital_spread(w)
        U = float(self.mf_U.get())
        for wdg in self.u5_table.winfo_children():
            wdg.destroy()
        for c, h in enumerate(["site", "atoms 90%", "U group", "U projected (eV)"]):
            ttk.Label(self.u5_table, text=h, foreground="gray").grid(row=0, column=c, padx=3, sticky="w")
        # default groups: sites with the same on-site parameter
        onsite = {i: q for i, j, R, q, sg in m.terms if i == j and R == 0}
        first = {}
        for k, name in enumerate(m.sites):
            first.setdefault(onsite.get(k, name), name)
        self.u5["rows"] = []
        for k, name in enumerate(m.sites):
            gvar = tk.StringVar(value="U_" + first[onsite.get(k, name)])
            ttk.Label(self.u5_table, text=name).grid(row=k + 1, column=0, sticky="w")
            ttk.Label(self.u5_table, text=str(n90[k])).grid(row=k + 1, column=1)
            ttk.Entry(self.u5_table, textvariable=gvar, width=10).grid(row=k + 1, column=2)
            ttk.Label(self.u5_table, text=f"{U * ipr[k]:.4f}").grid(row=k + 1, column=3)
            self.u5["rows"].append((name, gvar, None, float(U * ipr[k])))
        self.u5.update(refs=None, fit=None, model=m.text())
        self.u5_list_states()
        self._u5_log(f"model loaded: {len(m.sites)} sites, band fit rms {1e3 * r['rms']:.2f} meV. "
                     f"Projected U_i = U_atom sum|w|^4 (U_atom = {U} eV, tab 5) are the start of the fit.", clear=True)
        self.u5_draw()

    def _u5_check(self):
        m, r = self._sm_model()
        if not self.u5["rows"] or self.u5["model"] != m.text():
            raise ValueError("press 'Load model from tab 2' first (the tab 2 model changed or was not loaded)")
        return m, r

    # ------------------------------------------------------------------ 2. states from tab 5
    def u5_list_states(self):
        for wdg in self.u5_states.winfo_children():
            wdg.destroy()
        res = self.mf.get("results") if hasattr(self, "mf") else None
        self.u5["states"] = []
        if res and self.mf.get("finite"):
            ttk.Label(self.u5_states, text="(tab 5 holds a finite-chain mean field: the U fit needs the periodic\n"
                                           " one, run tab 5 in 'periodic chain' mode)", foreground="gray").pack(anchor="w")
            return
        if not res:
            ttk.Label(self.u5_states, text="(no mean-field states yet: run tab 5)", foreground="gray").pack(anchor="w")
            return
        E0 = res[0]["sol"]["E"]
        for o in res:
            v = tk.BooleanVar(value=o["mmax"] > 0.05 and bool(o["sol"]["converged"]))
            ttk.Checkbutton(self.u5_states, variable=v,
                            text=f"{o['label']}: E - E_min = {1e3 * (o['sol']['E'] - E0):.2f} meV, "
                                 f"max |m| = {o['mmax']:.2f}").pack(anchor="w")
            self.u5["states"].append((o, v))
        U, nk, kT = self.mf["par"]
        ttk.Label(self.u5_states, text=f"(U_atom = {U} eV, {nk} k points, kT = {kT} eV)",
                  foreground="gray").pack(anchor="w")

    # ------------------------------------------------------------------ 3. fit U
    def u5_fit(self):
        try:
            m, r = self._u5_check()
            chosen = [o for o, v in self.u5["states"] if v.get()]
            if not chosen:
                raise ValueError("tick at least one mean-field state (run tab 5 first, then 'Refresh the list')")
            gnames = [g.get().strip() or n for n, g, mv, u in self.u5["rows"]]
            uniq = list(dict.fromkeys(gnames))
            groups = [uniq.index(g) for g in gnames]
            U0 = [np.mean([u for (n, g, mv, u), gn in zip(self.u5["rows"], gnames) if gn == name]) for name in uniq]
            Uat, nk_, kT = self.mf["par"]
            wm = 0.1 if self.u5_fitm.get() else 0.0
        except Exception as e:
            messagebox.showerror("U from mean field", str(e))
            return
        pv = r["params"]
        H, st = self.H, self.st

        def work():
            Nmod, _ = tm.neutral_filling(m, r)
            refs = []
            for o in chosen:
                ref = tm.mfh_reference(H, st, m, o["sol"], Uat, kT)
                two_sz = o["sol"]["n_up"] - o["sol"]["n_dn"]
                ref["n_up_mod"], ref["n_dn_mod"] = (Nmod + two_sz) / 2, (Nmod - two_sz) / 2
                refs.append(ref)
            return refs, tm.fit_U_to_mfh(m, pv, refs, groups, U0, kT, moment_weight=wm)

        def done(out):
            refs, f = out
            self.u5.update(fit=f, groups=(uniq, groups), refs=refs, used=chosen)
            names = [row[0] for row in self.u5["rows"]]
            L = ["", "U of the reduced model fitted to the atomistic mean field (t fixed):",
                 f"  {'group':10s}{'sites':14s}{'U fitted':>10s}{'U projected':>13s}{'sensitivity':>14s}"]
            for g, name in enumerate(uniq):
                ss = [n for n, gg in zip(names, groups) if gg == g]
                L.append(f"  {name:10s}{','.join(ss):14s}{f['U_group'][g]:10.4f}{U0[g]:13.4f}"
                         f"{1e3 * f['sensitivity'][g]:11.2f} meV")
            L.append("  (sensitivity: rms change of the fitted bands for a change of 0.1 eV of that U; "
                     "a small value means that U is not determined by the mean field)")
            for c, o in enumerate(chosen):
                ma, mm = f["moments"][c]
                L.append(f"  {o['label']}: band rms {1e3 * f['rms'][c]:.2f} meV, offset {f['offsets'][c]:+.4f} eV; "
                         "moments atomistic / model: " + ", ".join(f"{n} {a:+.3f}/{b:+.3f}" for n, a, b in zip(names, ma, mm)))
                if refs[c]["overlap"] < 0.5:
                    L.append(f"    WARNING: the model bands are poorly represented in this state (overlap "
                             f"{refs[c]['overlap']:.2f}): its magnetic bands mix with bands outside the window")
            if len(chosen) > 1:
                L.append("  energies relative to the first state (per cell): " + "; ".join(
                    f"{o['label']}: atomistic {1e3 * (o['sol']['E'] - chosen[0]['sol']['E']):+.3f} meV, model "
                    f"{1e3 * (f['sols'][c]['E'] - f['sols'][0]['E']):+.3f} meV" for c, o in enumerate(chosen) if c))
            weak = [uniq[g] for g in range(len(uniq)) if f["sensitivity"][g] < 0.002]
            if weak:
                L.append("  WARNING: " + ", ".join(weak) + " barely change the mean-field bands (the sites do not "
                         "polarise): their value is not determined; keep the projected value or group them")
            if f["rank"] < len(uniq):
                L.append(f"  WARNING: only {f['rank']} of {len(uniq)} U values are determined")
            self._u5_log("\n".join(L))
            self.u5_draw()

        self._u5_log(f"fitting {len(uniq)} U values ({', '.join(uniq)}) to {len(chosen)} states, several starts ...")
        self._sm_background(work, done, self.u5_btn_fit)

    def u5_to_tab4(self):
        f = self.u5["fit"]
        if f is None:
            messagebox.showerror("U from mean field", "fit U first (box 3)")
            return
        m, _ = self._sm_model()
        if not self.sm["rows"]:
            self.sm_sites()
        vals = dict(zip(m.sites, f["U_site"]))
        for name, uvar, svar, uauto in self.sm["rows"]:
            if name in vals:
                uvar.set(f"{vals[name]:.4f}")
        self.sm_mode.set("onsite")
        self._sm_log("U_i set from tab 6 (fitted to the atomistic mean field): " +
                     ", ".join(f"{n} {v:.4f}" for n, v in vals.items()))
        self._u5_log("the fitted U_i are now in the U_i column of tab 4 (on-site mode)")

    # ------------------------------------------------------------------ drawing
    def u5_draw(self):
        for ax in (self.ax_f5, self.ax_a5, self.ax_c5, self.ax_e5):
            ax.clear()
        refs, f, used = self.u5.get("refs"), self.u5.get("fit"), self.u5.get("used")
        for c, ax in enumerate((self.ax_f5, self.ax_a5)):
            if f is None or c >= len(used):
                ax.set_axis_off()
                continue
            ax.set_axis_on()
            sol, ref, lab = used[c]["sol"], refs[c], used[c]["label"]
            ks = sol["ks"] / np.pi
            kx = np.where(ks > 1, ks - 2, ks)
            o = np.argsort(kx)
            ref0 = 0.5 * (sol["Ef"][0] + sol["Ef"][1])
            for s_ in (0, 1):
                ax.plot(kx[o], sol["bands"][s_][o] - ref0, color="0.85", lw=0.6)
            for s_, col, mk in ((0, "tab:red", "o"), (1, "tab:blue", "s")):
                ax.plot(kx[o], ref["E"][s_][o] - ref0, mk, color=col, ms=2.5, alpha=0.8)
            if f is not None:
                ms = f["sols"][c]
                for s_, col, ls, lw in ((0, "tab:red", "-", 2.0), (1, "tab:blue", (0, (4, 2.5)), 1.7)):
                    ax.plot(kx[o], ms["bands"][s_][o] + f["offsets"][c] - ref0, linestyle=ls, color=col, lw=lw)
            lo, hi = min(e.min() for e in ref["E"]) - ref0, max(e.max() for e in ref["E"]) - ref0
            pad = 0.25 * (hi - lo) + 0.05
            ax.set_ylim(lo - pad, hi + pad)
            ax.set_xlabel("k (pi / |T|)")
            ax.set_ylabel("E - E_F (eV)")
            ax.set_title(f"{lab}: atomistic MF (dots: up red, down blue)\nreduced model MF with fitted U "
                         "(lines: up red, down blue dashed)", fontsize=8)
        names = [row[0] for row in self.u5["rows"]]
        ax = self.ax_c5
        if f is not None:
            ax.set_axis_on()
            x = np.arange(len(names))
            nst = len(used)
            wbar = 0.8 / (2 * nst)
            cols = ["tab:red", "tab:blue", "tab:green", "tab:purple", "tab:brown"]
            for c, o in enumerate(used):
                ma, mm = f["moments"][c]
                x0 = x - 0.4 + (2 * c + 0.5) * wbar
                ax.bar(x0, ma, wbar, color=cols[c % 5], alpha=0.45, label=f"{o['label']} atomistic")
                ax.bar(x0 + wbar, mm, wbar, color=cols[c % 5], label="model")
            ax.axhline(0, color="k", lw=0.5)
            ax.set_xticks(x)
            ax.set_xticklabels(names)
            ax.set_ylabel("moment n_up - n_dn")
            ax.legend(fontsize=7)
            ax.set_title("site moments: atomistic mean field (light) vs reduced model (dark)", fontsize=8)
            ax = self.ax_e5
            ax.set_axis_on()
            Up = [row[3] for row in self.u5["rows"]]
            ax.bar(x - 0.2, Up, 0.4, color="0.7", label="projected U sum|w|^4")
            ax.bar(x + 0.2, f["U_site"], 0.4, color="tab:green", label="fitted to the mean field")
            ax.set_xticks(x)
            ax.set_xticklabels(names)
            ax.set_ylabel("U_i (eV)")
            ax.legend(fontsize=7)
            ax.set_title("U of the reduced model", fontsize=9)
        else:
            self.ax_e5.set_axis_off()
            ax.set_axis_off()
        self.fig5.tight_layout()
        self.can5.draw_idle()


    def u5_to_tab6(self):
        if self.u5.get("fit") is None:
            messagebox.showerror("U from mean field", "fit U first (box 3)")
            return
        self.e6_src.set("fitted")
        self.e6_load()
        self.nb.select(self.tab6)

    # ================================================================== tab 7: ED, spectra and maps
    def _build_tab6(self):
        self.e6 = dict(rows=[], cluster=None, ed=None, ch=None, taus=None, sel=None, peak=None, tip=None,
                       model=None, usrc="")
        left = self._scrollable(self.tab6, width=540)
        blue = dict(foreground="#1f4e79", font=("Courier", 9))

        box = ttk.LabelFrame(left, text="1. Hubbard model of the reduced sites", padding=6)
        box.pack(fill="x")
        ttk.Label(box, text="H = sum eps_i n_is + sum t_ij c+_is c_js\n"
                            "  + sum U_i (n_i,up - 1/2)(n_i,dn - 1/2)\n"
                            "eps_i, t_ij: tab 2 fit;  U_i: chosen below", **blue).pack(anchor="w")
        self.e6_modeltxt = tk.Text(box, width=68, height=5, font=("Courier", 9), background="#f4f4f4")
        self.e6_modeltxt.pack(fill="x", pady=(2, 4))
        self.e6_modeltxt.configure(state="disabled")
        row = ttk.Frame(box)
        row.pack(fill="x")
        self.e6_src = tk.StringVar(value="fitted")
        for val, lab in (("fitted", "U fitted in tab 6"), ("projected", "U_atom sum|w|^4"), ("manual", "manual")):
            ttk.Radiobutton(row, text=lab, value=val, variable=self.e6_src).pack(side="left")
        self.e6_U = self._entry(box, "U_atom (eV)", "3.0")
        ttk.Button(box, text="Load model (tab 2) and U", command=self.e6_load).pack(fill="x", pady=(4, 2))
        self.e6_table = ttk.Frame(box)
        self.e6_table.pack(fill="x")

        box = ttk.LabelFrame(left, text="2. Many-body solution: ED or DMRG (cluster of unit cells)", padding=6)
        box.pack(fill="x", pady=(6, 0))
        self._solver_row(box, "e6")
        self.e6_cells, self.e6_nst = self._entry2(box, "cells", "2", "states", "16")
        row = ttk.Frame(box)
        row.pack(fill="x")
        self.e6_bc = tk.StringVar(value="open")
        ttk.Radiobutton(row, text="open chain", value="open", variable=self.e6_bc).pack(side="left")
        ttk.Radiobutton(row, text="ring (periodic)", value="ring", variable=self.e6_bc).pack(side="left")
        self.e6_keep = self._entry(box, "last cell keeps sites (blank: all)", "")
        ttk.Label(box, text="odd chains (open chain only): the tab 2 sites kept in the last cell,\n"
                            "the first ones along the chain. Olympicene (sites A, B): 3 cells,\n"
                            "keep 'A' -> A B A B A. A cell of two units with 3 sites each: 2 cells,\n"
                            "keep the 3 sites of its first unit -> 3 units (Run ED lists the order)",
                  foreground="gray").pack(anchor="w")
        self.e6_nel = self._entry(box, "electrons (blank: neutral)", "")
        self.e6_btn_ed = ttk.Button(box, text="Run ED", command=self.e6_run_ed)
        self.e6_btn_ed.pack(fill="x", pady=(4, 0))
        self._solver_changed("e6")

        box = ttk.LabelFrame(left, text="3. Excitations: maps and tip spectra", padding=6)
        box.pack(fill="x", pady=(6, 0))
        ttk.Label(box, text="Click a level of the many-body spectrum: map of its spin-flip weight.\n"
                            "Click an atom of the map: tip there (spectra below the map).\n"
                            "Click a peak of the tip spectrum: map of that charge excitation.",
                  foreground="gray").pack(anchor="w")
        self.e6_nch = self._entry(box, "N-1, N+1 states / sector", "12")
        self.e6_btn_ch = ttk.Button(box, text="Compute charge excitations (N-1 and N+1)", command=self.e6_run_charge)
        self.e6_btn_ch.pack(fill="x", pady=(2, 2))
        row = ttk.Frame(box)
        row.pack(fill="x")
        ttk.Label(row, text="tip on site").pack(side="left")
        self.e6_site = tk.StringVar(value="")
        self.e6_sitecb = ttk.Combobox(row, textvariable=self.e6_site, width=8, state="readonly")
        self.e6_sitecb.pack(side="left", padx=4)
        self.e6_sitecb.bind("<<ComboboxSelected>>", lambda e: self.e6_set_tip(("site", self.e6_site.get())))
        self.e6_wel, self.e6_win = self._entry2(box, "width (eV)", "0.02", "inel. (meV)", "auto")
        self.e6_tiptxt = tk.StringVar(value="tip: (none)")
        ttk.Label(box, textvariable=self.e6_tiptxt, foreground="gray").pack(anchor="w")

        self.e6_out = tk.Text(left, width=68, height=22, font=("Courier", 9))
        self.e6_out.pack(fill="both", expand=True, pady=(6, 0))

        right = ttk.Frame(self.tab6)
        right.pack(side="left", fill="both", expand=True)
        self.fig6 = Figure(figsize=(10, 8))
        gs = self.fig6.add_gridspec(2, 2, width_ratios=[1.8, 1])
        self.ax_m6 = self.fig6.add_subplot(gs[0, 0])
        self.ax_s6 = self.fig6.add_subplot(gs[0, 1])
        self.ax_a6 = self.fig6.add_subplot(gs[1, 0])
        self.ax_i6 = self.fig6.add_subplot(gs[1, 1])
        self.can6 = FigureCanvasTkAgg(self.fig6, right)
        self.tb6 = NavigationToolbar2Tk(self.can6, right)
        self.tb6.update()
        self.can6.get_tk_widget().pack(fill="both", expand=True)
        self._on_clicks(self.can6, self.e6_click)

    def _e6_log(self, text, clear=False):
        if clear:
            self.e6_out.delete("1.0", "end")
        self.e6_out.insert("end", text + "\n")
        self.e6_out.see("end")

    # ------------------------------------------------------------------ 1. model and U
    def e6_load(self):
        try:
            m, r = self._sm_model()
            w, _, _ = tm.wannier_orbitals(self.H, self.st, m)
            src = self.e6_src.get()
            ipr, n90 = tm.orbital_spread(w)
            Ua = float(self.e6_U.get())
            if src == "fitted":
                f = self.u5.get("fit") if hasattr(self, "u5") else None
                if f is None or self.u5.get("model") != m.text():
                    raise ValueError("no U fitted in tab 6 for the current tab 2 model: fit it there, or choose "
                                     "'U_atom sum|w|^4' or 'manual'")
                Uvals = list(f["U_site"])
                usrc = "fitted to the mean field in tab 6"
            elif src == "projected":
                Uvals = list(Ua * ipr)
                usrc = f"projected, U_atom = {Ua} eV x sum|w|^4"
            else:
                old = {n: float(u.get()) for n, u, sp in self.e6["rows"]} if self.e6["rows"] else {}
                Uvals = [old.get(n, Ua * ipr[k]) for k, n in enumerate(m.sites)]
                usrc = "manual"
        except Exception as e:
            messagebox.showerror("ED", str(e))
            return
        self._sm_show_model(m, r, widget=self.e6_modeltxt)
        for wdg in self.e6_table.winfo_children():
            wdg.destroy()
        for c, h in enumerate(["site", "atoms 90%", "U_i (eV)", "spin"]):
            ttk.Label(self.e6_table, text=h, foreground="gray").grid(row=0, column=c, padx=3, sticky="w")
        loc = ipr >= 0.8 * ipr.max()
        self.e6["rows"] = []
        for k, name in enumerate(m.sites):
            uvar = tk.StringVar(value=f"{Uvals[k]:.4f}")
            svar = tk.BooleanVar(value=bool(loc[k]))
            ttk.Label(self.e6_table, text=name).grid(row=k + 1, column=0, sticky="w")
            ttk.Label(self.e6_table, text=str(n90[k])).grid(row=k + 1, column=1)
            ttk.Entry(self.e6_table, textvariable=uvar, width=8).grid(row=k + 1, column=2)
            ttk.Checkbutton(self.e6_table, variable=svar).grid(row=k + 1, column=3)
            self.e6["rows"].append((name, uvar, svar))
        self.e6.update(model=m.text(), usrc=usrc, cluster=None, ed=None, ch=None, taus=None, sel=None, peak=None,
                       tip=None)
        self._e6_log(f"model of tab 2 loaded ({len(m.sites)} sites); U_i {usrc}: " +
                     ", ".join(f"{n} {u:.4f}" for n, u in zip(m.sites, Uvals)) +
                     "\n(the U_i column can be edited; 'spin' ticks only set the zoom on the spin manifold)", clear=True)
        self.e6_draw()

    # ------------------------------------------------------------------ 2. ED
    def e6_run_ed(self):
        try:
            m, r = self._sm_model()
            if not self.e6["rows"] or self.e6["model"] != m.text():
                raise ValueError("press 'Load model (tab 2) and U' first (the tab 2 model changed or was not loaded)")
            Uov = {n: float(u.get()) for n, u, sp in self.e6["rows"]}
            ncells, nst = int(self.e6_cells.get()), int(self.e6_nst.get())
            periodic = self.e6_bc.get() == "ring"
            ns = len(m.sites)
            keep = self._keep_list(self.e6_keep)
            if not self._keep_check(m, keep):
                return
            kept = tm.cluster_keep(m, ncells, keep, periodic)
            Ns = len(kept)
            nel_neutral, per_cell = tm.neutral_filling(m, r, ncells, Ns)
            nel = int(self.e6_nel.get()) if self.e6_nel.get().strip() else nel_neutral
            nup = (nel + nel % 2) // 2
            dim = comb(Ns, nup) * comb(Ns, nel - nup)
            solver = self._solver_opts("e6")
            if dim > 3e6 and solver[0] == "ed":
                raise ValueError(f"dimension {dim:,} is too large for ED; use fewer cells or the DMRG solver")
            spn = {row[0] for row in self.e6["rows"] if row[2].get()}
            nsp = sum(1 for i in kept if m.sites[i % ns] in spn)
        except Exception as e:
            messagebox.showerror("ED", str(e))
            return
        H, st, Ua = self.H, self.st, float(self.e6_U.get())

        def work():
            cl = tm.build_cluster(H, st, m, r["params"], ncells=ncells, periodic=periodic, U=Ua, mode="onsite",
                                  EF=r["EF"], U_override=Uov, ph=True, keep_last=keep)
            res = self._solve(cl, nel, min(nst, dim), solver)
            res["levels"], res["lw"] = tm.level_weights(res)
            lev, taus, ng = tm.inelastic_amplitudes(res)
            return cl, res, (taus, ng)

        def done(out):
            cl, res, taus = out
            res["M"] = comb(nsp, nsp // 2) if nsp else 0
            labels = cl["labels"]
            self.e6.update(cluster=cl, ed=res, taus=taus, ch=None, sel=None, peak=None, tip=("site", labels[0]))
            self.e6_sitecb.configure(values=labels)
            self.e6_site.set(labels[0])
            E = res["E"] - res["E"][0]
            L = ["", f"ED: {self._cells_text(cl)} ({'ring' if periodic else 'open chain'}), {nel} electrons, "
                     f"dimension {res['dim']:,}"]
            L += self._solver_lines(res)
            L += self._sm_ed_hamiltonian(m, r, cl, "onsite", u_source=self.e6["usrc"])
            L += ["lowest states (E - E0 in meV, total S):"]
            L += [f"  {k:3d} {1e3 * e:10.4f}   S = {s_:.2f}" for k, (e, s_) in enumerate(zip(E, res["S"]))]
            C0 = res["C"][0]
            L.append("ground state: charge / local moment <S_i^2> (0.75 = free spin):")
            L += [f"  {lab:6s} n = {n:.3f}  <S^2> = {C0[i, i]:.3f}" for i, (lab, n) in
                  enumerate(zip(labels, res["n"][0]))][:len(labels) if len(labels) <= 12 else 2 * ns]
            L += self._sm_transition_table(res, cl)
            self._e6_log("\n".join(L))
            self.e6_draw()

        self._e6_log(f"running {'DMRG' if solver[0] == 'dmrg' else 'ED'} (dimension {dim:,}) ...")
        self._sm_background(work, done, self.e6_btn_ed)

    def e6_run_charge(self):
        res, cl = self.e6["ed"], self.e6["cluster"]
        if res is None:
            messagebox.showerror("ED", "run the ED first")
            return
        try:
            nch = int(self.e6_nch.get())
        except ValueError:
            nch = 12

        def work():
            return tm.charge_excitations(res, cl, nstates=nch)

        def done(ch):
            self.e6["ch"] = ch
            add = np.concatenate([c["w"] for c in ch if c["kind"] == "add"])
            rem = np.concatenate([c["w"] for c in ch if c["kind"] == "remove"])
            L = ["", f"charge excitations ({nch} states per sector; energies from E_F):",
                 "  removal (N-1), highest first (eV): " + ", ".join(f"{x:+.4f}" for x in np.sort(rem)[::-1][:8]),
                 "  addition (N+1), lowest first (eV): " + ", ".join(f"{x:+.4f}" for x in np.sort(add)[:8]),
                 f"  charge gap E0(N+1) + E0(N-1) - 2 E0(N) = {add.min() - rem.max():.4f} eV",
                 "  (only the lowest states of each sector are computed: the tip spectrum is complete near",
                 f"   the gap, up to {add.max():+.3f} eV above and {rem.min():+.3f} eV below)"]
            self._e6_log("\n".join(L))
            self.e6_draw()

        self._e6_log("computing the N-1 and N+1 states" + (" (DMRG)" if res.get("solver") == "dmrg" else "") + " ...")
        self._sm_background(work, done, self.e6_btn_ch)

    # ------------------------------------------------------------------ tip and clicks
    def _e6_tip_vector(self):
        cl, tip = self.e6["cluster"], self.e6["tip"]
        if cl is None or tip is None:
            return None, ""
        if tip[0] == "atom":
            x = tip[1]
            v = cl["W"][x]
            return v, f"atom at ({cl['atom_xy'][x][0]:.2f}, {cl['atom_xy'][x][1]:.2f}) A"
        i = cl["labels"].index(tip[1])
        return np.eye(len(cl["labels"]))[i], f"effective site {tip[1]}"

    def e6_set_tip(self, tip):
        self.e6["tip"] = tip
        self.e6_draw()

    def e6_click(self, ev):
        res, cl = self.e6["ed"], self.e6["cluster"]
        if res is None or ev.button != 1 or ev.inaxes is None:
            return
        if ev.inaxes is self.ax_m6 and ev.xdata is not None:
            P = cl["atom_xy"]
            d = np.hypot(P[:, 0] - ev.xdata, P[:, 1] - ev.ydata)
            x = int(np.argmin(d))
            if d[x] < 1.0:
                self.e6_set_tip(("atom", x))
        elif ev.inaxes is self.ax_s6:
            E = 1e3 * (res["E"] - res["E"][0])
            pts = self.ax_s6.transData.transform(np.c_[res["S"], E])
            d = np.hypot(pts[:, 0] - ev.x, pts[:, 1] - ev.y)
            k = int(np.argmin(d))
            if d[k] > 25:
                return
            lev = next(L_ for L_ in res["levels"] if k in L_)
            if lev == res["levels"][0]:
                self.e6.update(sel=None, peak=None)
            else:
                self.config(cursor="watch")
                self.update_idletasks()
                try:
                    self.e6.update(sel=tm.analyse_transition(res, lev, W=cl["W"]), peak=None)
                finally:
                    self.config(cursor="")
                a = self.e6["sel"]
                self._e6_log(f"\nlevel {lev[0]}: Delta E = {1e3 * a['dE']:.4f} meV, S {a['S0']:.1f} -> {a['Sf']:.1f}, "
                             f"{'allowed' if a['allowed'] else 'forbidden'}, {a['kind']}, W = {a['total']:.4f}; per site: " +
                             ", ".join(f"{cl['labels'][i]} {a['w_site'][i]:.3f}" for i in np.argsort(a["w_site"])[::-1][:4]))
            self.e6_draw()
        elif ev.inaxes is self.ax_a6 and self.e6["ch"] is not None and ev.xdata is not None:
            best = None
            for c in self.e6["ch"]:
                norm = (c["D"] ** 2).sum(axis=2).mean(axis=0)            # total weight of each state
                for j, w in enumerate(c["w"]):
                    if norm[j] < 1e-4:                                    # dark (e.g. spin 3/2 from a singlet)
                        continue
                    if best is None or abs(w - ev.xdata) < abs(best[2] - ev.xdata):
                        best = (c["kind"], j, w, c["spin"])
            if best is not None:
                self.e6.update(peak=best, sel=None)
                spin = "" if len([c for c in self.e6["ch"] if c["kind"] == best[0]]) == 1 else \
                    (" (spin up)" if best[3] == 0 else " (spin down)")
                self._e6_log(f"\ncharge excitation: {'addition (N+1)' if best[0] == 'add' else 'removal (N-1)'}{spin} "
                             f"state {best[1]} at {best[2]:+.4f} eV: map = |Dyson orbital|^2 on the atoms")
                self.e6_draw()

    # ------------------------------------------------------------------ drawing
    def e6_draw(self):
        for ax in (self.ax_m6, self.ax_s6, self.ax_a6, self.ax_i6):
            ax.clear()
        cl, res, m = self.e6["cluster"], self.e6["ed"], self.model
        if cl is None or res is None or m is None:
            for ax in (self.ax_m6, self.ax_s6, self.ax_a6, self.ax_i6):
                ax.set_axis_off()
            self.fig6.tight_layout()
            self.can6.draw_idle()
            return
        for ax in (self.ax_m6, self.ax_s6, self.ax_a6, self.ax_i6):
            ax.set_axis_on()
        st = self.st
        ns, nc = cl["ns"], cl["ncells"]
        pos = self._cluster_pos(m, cl)
        # --- map
        ax = self.ax_m6
        ax.add_collection(self._chain_bonds(st, range(nc), 0.5, "0.7", shown=tm.cluster_atoms_shown(cl)))
        sel, peak = self.e6["sel"], self.e6["peak"]
        if sel is not None:
            self._sm_draw_transition(ax, sel, cl, pos)
        elif peak is not None and self.e6["ch"] is not None:
            vals = tm.dyson_map(self.e6["ch"], peak[0], peak[1], cl["W"], spin=peak[3])
            xy = cl["atom_xy"]
            keep = vals > 1e-3 * max(vals.max(), 1e-12)
            if keep.any():
                img, ext = tb.ldos_image(tb.Structure(xy[keep], ["C"] * int(keep.sum()), st.T), vals[keep],
                                         cells=(0,), sigma=0.9)
                ax.imshow(img, origin="lower", extent=ext, cmap="Purples", vmin=0, aspect="equal", zorder=1)
            ax.scatter(*pos.T, s=20, color="k", zorder=4)
            for a, lab in enumerate(cl["labels"]):
                ax.annotate(lab, pos[a], xytext=(4, 5), textcoords="offset points", fontsize=7, zorder=5)
            ax.set_title(f"{'addition (N+1)' if peak[0] == 'add' else 'removal (N-1)'} at {peak[2]:+.4f} eV: "
                         "|Dyson orbital|^2 on the atoms (STM map at that bias)", fontsize=8)
        else:
            C = res["C"][0]
            for a in range(len(pos)):
                for b in range(a + 1, len(pos)):
                    if abs(C[a, b]) > 0.02:
                        ax.plot(*pos[[a, b]].T, color="tab:blue" if C[a, b] < 0 else "tab:red",
                                lw=0.5 + 6 * abs(C[a, b]), alpha=0.7, zorder=3)
            ax.scatter(*pos.T, s=400 * np.diag(C), color="k", zorder=4)
            for a, lab in enumerate(cl["labels"]):
                ax.annotate(lab, pos[a], xytext=(4, 5), textcoords="offset points", fontsize=7, zorder=5)
            ax.set_title("ground state <S_i.S_j> (blue AFM, red FM); click an atom to place the tip", fontsize=8)
        v, tipname = self._e6_tip_vector()
        tip = self.e6["tip"]
        if tip is not None:
            tp = cl["atom_xy"][tip[1]] if tip[0] == "atom" else pos[cl["labels"].index(tip[1])]
            ax.scatter([tp[0]], [tp[1]], marker="*", s=260, color="tab:green", edgecolor="k", zorder=6)
        self.e6_tiptxt.set(f"tip: {tipname}")
        ax.set_aspect("equal")
        ax.autoscale_view()
        # --- many-body spectrum
        ax = self.ax_s6
        E = 1e3 * (res["E"] - res["E"][0])
        ax.scatter(res["S"], E, marker="_", s=400, color="k")
        tot = res["lw"].sum(axis=1)[1:]
        idx = [L_[0] for L_ in res["levels"]][1:]
        if tot.max(initial=0) > 1e-4:
            ax.scatter(res["S"][idx] - 0.14, E[idx], s=120 * tot / tot.max(), color="tab:orange", alpha=0.8,
                       label="spin-flip weight (all sites)")
            ax.legend(fontsize=7, loc="lower right")
        if sel is not None:
            k = sel["f_level"][0]
            ax.scatter([res["S"][k]], [E[k]], marker="s", s=90, facecolors="none", edgecolors="tab:orange", lw=1.8)
        Mm = res.get("M", 0)
        if Mm and len(E) > Mm and E[Mm - 1] > 0 and E[Mm] > 3 * E[Mm - 1]:
            ax.set_ylim(-0.15 * E[Mm - 1] - 1e-3, 1.3 * E[Mm - 1] + 1e-3)
            ax.text(0.02, 0.97, f"next state at +{E[Mm]:.2f} meV (off scale)", transform=ax.transAxes,
                    fontsize=7, va="top")
        ax.set_xlabel("total spin S")
        ax.set_ylabel("E - E0 (meV)")
        ax.set_title("many-body spectrum (click a level)", fontsize=9)
        # --- elastic tip spectrum (charge excitations)
        ax = self.ax_a6
        ch = self.e6["ch"]
        if ch is not None and v is not None:
            Ep, Wp, kind = tm.tip_peaks(ch, v)
            try:
                wd = float(self.e6_wel.get())
            except ValueError:
                wd = 0.02
            lo, hi = Ep.min() - 5 * wd, Ep.max() + 5 * wd
            grid = np.linspace(lo, hi, 1500)
            ax.plot(grid, tm.broadened(Ep, Wp, grid, wd), color="k", lw=1.3)
            for kd, col in (("remove", "tab:blue"), ("add", "tab:red")):
                sel_ = kind == kd
                ax.vlines(Ep[sel_], 0, Wp[sel_] / (np.sqrt(2 * np.pi) * wd), color=col, lw=1.0, alpha=0.8,
                          label="removal (N-1)" if kd == "remove" else "addition (N+1)")
            if peak is not None:
                ax.axvline(peak[2], color="tab:purple", ls=":", lw=1.2)
            ax.axvline(0, color="0.5", lw=0.6, ls="--")
            ax.set_xlabel("bias = E - E_F (eV)")
            ax.set_ylabel("A(E) at the tip")
            ax.legend(fontsize=7)
            ax.set_title(f"tip spectrum, charge excitations ({tipname}); click a peak for its map", fontsize=8)
        else:
            ax.text(0.5, 0.5, "press 'Compute charge excitations'\nfor the tip spectrum (N-1, N+1)",
                    ha="center", va="center", transform=ax.transAxes, color="0.4")
            ax.set_xticks([])
            ax.set_yticks([])
        # --- inelastic tip spectrum (spin flips)
        ax = self.ax_i6
        if self.e6["taus"] is not None and v is not None:
            taus, ng = self.e6["taus"]
            Wl = tm.inelastic_weights_at(taus, ng, v)[1:]
            dE = np.array([E[L_[0]] for L_ in res["levels"]][1:])
            Mm = res.get("M", 0)
            top = E[Mm - 1] if (Mm and len(E) > Mm and E[Mm - 1] > 0 and E[Mm] > 3 * E[Mm - 1]) else dE.max()
            try:
                wi = float(self.e6_win.get())
            except ValueError:
                pos_dE = dE[dE > 1e-9]
                wi = 0.03 * (pos_dE.min() if len(pos_dE) else 1.0)
            bias = np.linspace(-1.4 * top, 1.4 * top, 1201)
            ax.plot(bias, tm.iets_dIdV(dE, Wl, bias, wi), color="tab:orange", lw=1.6)
            for x_, w_ in zip(dE, Wl):
                if w_ > 1e-3 * max(Wl.max(), 1e-12) and x_ <= 1.4 * top:
                    ax.axvline(x_, color="0.7", lw=0.6, ls=":")
                    ax.axvline(-x_, color="0.7", lw=0.6, ls=":")
            ax.set_xlabel("bias (meV)")
            ax.set_ylabel("inelastic dI/dV (steps = W)")
            ax.set_title("tip spectrum, spin-flip excitations (IETS)", fontsize=8)
        self.fig6.tight_layout()
        self.can6.draw_idle()


if __name__ == "__main__":
    App().mainloop()
