# tbfit: tight-binding bands of 1D ribbons and effective-model fitting

    python tbgui.py            (needs numpy, scipy, matplotlib; tkinter comes with Python)

`tbcore.py` holds all the physics and can be used without the GUI.

## Tab 1: Structure and bands

Load a structure (or pick an example). The program builds the pz tight-binding
Hamiltonian with 1st, 2nd, 3rd-neighbour hoppings t1, t2, t3, found from the interatomic
distances (a, sqrt(3)a, 2a with a = 1.42 A), and computes the bands of the infinite
periodic system, H(k) = sum_R H_R e^{ikR}. The plot shows the unit cell (black), the bonds
to the next cell (red) and the neighbouring cells (grey). Click an atom to see its index
and coordinates.

Set t2 = t3 = 0 for nearest neighbours only.

### Modify the structure (sp3 carbons, vacancies)

Set "click on an atom" to "remove (sp3)" and click atoms in the structure or in the dI/dV
map: each click removes that pz site in every unit cell (an sp3 CH2 carbon or a vacancy)
and the bands, the DOS and the maps are recalculated at once. Removed atoms are marked
with blue circles. "Undo" puts back the last one, "Restore all" returns to the loaded
structure. The info box shows the sublattice balance: with nearest-neighbour hopping each
unit of imbalance gives one zero-energy band (Lieb). With an odd number of pi electrons
E_F is placed in the middle of the half-filled band.
"Save structure as..." writes the modified structure as a hexagon map with `remove` lines
(for .hex structures), as .xyz with the period, or as ChemDraw .cdxml.
Atoms picked for spectra or as effective-model sites follow the edit automatically.

### DOS and tunnelling spectroscopy

With the bands the program computes the local density of states of every atom,
LDOS_i(E) = sum over k, n of |psi_n(k)_i|^2 g(E - E_n(k)), with Gaussian broadening
(the energy resolution of the measurement), averaged over the Brillouin zone.

- The **DOS** per unit cell is shown beside the bands, on the same energy axis.
- The **simulated dI/dV map** below the structure shows the LDOS at one energy. Every
  atom is a Gaussian spot of width "tip resolution" (about 1 A mimics the pz orbital
  seen by the tip at constant height). Set it to 0 to see the LDOS on the atoms
  themselves. Click the bands or the DOS to choose the energy (the blue line).
- **dI/dV spectra**: click atoms in the structure or in the map to add their LDOS(E) (up
  to 8, click again to remove). This is a point spectrum with the tip on that atom; the
  grey dashed curve is the average per atom.
- "Export (CSV)" saves E - E_F, the DOS and the LDOS of every atom, on the energy window
  shown and with the broadening used in the plot.
- **map** chooses what the map shows: the dI/dV (LDOS) at the map energy, or the clicked
  band, |psi_n|^2 averaged over k or at the clicked k (the band is drawn in orange). The
  maximum on an atom is written in the corner (every map is scaled to its own maximum, so a
  map in a gap is weak, not empty).

**Zoom.** Zoom or pan the bands, the DOS or the spectra with the toolbar (rectangle, pan, home,
back): the DOS, the map and the spectra are recomputed for the energy window shown, and the
two energy axes (bands and spectra) follow each other. Clicks keep working while the zoom or
pan tool is on (a click is a press and release at the same place; a drag zooms or pans), and
they do not reset the zoom. With broadening "auto" (default) the Gaussian width is 1/250 of the
window, between 0.1 and 20 meV: 20 meV for the full -3..3 eV view, 0.6 meV for a window of
0.16 eV, so bands a few meV apart are resolved when zoomed in. A number (meV) fixes it.

How: the eigenstates of all the bands are computed once (401 k points between 0 and pi for small
cells, fewer for large ones); every band is interpolated linearly between the k points and the
pieces are broadened analytically (Gaussian convolved with a box: 1D linear tetrahedron). The
spectra therefore do not depend on an energy grid or on the k mesh even at 0.1 meV, and the map is
evaluated exactly at the clicked energy (the old version took the nearest point of a 10 meV grid
with a fixed 20 meV broadening, so close bands could not be told apart).

Untick "compute together with the bands" for very large cells, and use
"Update DOS / LDOS" after changing the broadening or the energy range.

### Structure files

| format | how the period is given |
|---|---|
| `.hex` / `.txt` hexagon map | the width of the map (`O` = hexagon, `O.O` = neighbours, each row shifted by one character; the ribbon axis is a zigzag direction). With a line `armchair`: flat-top hexagons for ribbons along an armchair direction, each character is a column and each row half a hexagon down, the `O`s form a checkerboard (example `examples/7AGNR.hex`). Add `remove x y` lines to delete atoms without a pz orbital (sp3 CH2 carbons, vacancies); x, y in A as shown when clicking the atom in the GUI |
| `.cdxml` (ChemDraw: File > Save As > CDXML) | star atoms: type `*` (or R, X) as an atom label at the two ends of each periodic bond |
| `.mol` / `.sdf` | star atoms `*` (or R, X) |
| `.xyz` | comment line `period=Tx,Ty` or extended-xyz `Lattice="..."`, or star atoms |

How the period of a drawing (.cdxml, .mol) is found, in this order:

1. **Repeat units**: if the drawing shows two or more repeat units, the period is the
   shortest shift that maps every atom exactly onto another atom (atoms whose image would
   fall beyond the end of the drawing are exempt). One period from the middle of the
   drawing is kept.
2. **Star atoms or dashed bonds**: a star atom (`*`, R, X) or the far atom of a dashed
   bond at the end of the drawing sits where the atom of the next (or previous) cell would
   be; the translation vector is deduced from them.
3. The period found is reduced to the shortest one (a drawing of 2 cells with stars gives
   the 1-cell period). The info box says which rule was used.

Both axes of the drawing are searched. A short, wide drawing can repeat along two
directions: two columns of three stacked hexagons (a 7-AGNR drawn with 2 cells) repeat
every 4.26 A horizontally and every 2.46 A vertically. The GUI then asks which one is the
ribbon axis (the horizontal one is proposed first). Drawing three or more repeat units along
the ribbon, or marking the bonds to the next cell with stars or dashed bonds, avoids the
question.

sp3 carbons are removed from the pi system automatically:
- a carbon with 4 neighbours counting its hydrogens: explicit H atoms (ChemDraw, .mol, or an
  .xyz from DFT, where H closer than 1.65 A count, also across the periodic boundary) or a
  ChemDraw label such as "CH2";
- in a Kekule drawing (with double bonds), a carbon with no double bond and no radical mark.
Radical carbons stay sp2. The info box reports how many sp3 carbons were found.
Example: `examples/sextet_ribbon_H.xyz` (hydrogenated ribbon with four CH2 per cell). Hydrogens and halogens are
ignored and drawings are rescaled to C-C = 1.42 A. Heteroatoms use the Hueckel on-site
energies in `ONSITE` in `tbcore.py`. If a file has no stars, type the period T in the GUI.
Binary `.cdx` is not read: save as `.cdxml`. Any structure can be exported to `.cdxml`.

## Tab 2: Effective model

### Build a model by picking sites (no typing needed)

1. In tab 1, read the list of isolated band groups around E_F. A model with n sites
   fits n bands, so this list tells you how many sites to pick.
2. In tab 2, click atoms on the structure. A left click adds an atom to the current site
   with weight +1, a right click adds it with -1, and clicking it again removes it.
   Clicking an atom that belongs to another site selects that site. "New site" starts the
   next site, and you can rename the current site. Atoms in the neighbouring cells can be
   clicked too. The plot on the right shows, in black, the bands the model will fit.
3. Choose the bands to fit: the ones centred on E_F, one of the isolated groups, or "auto".
4. Press "Build model from sites and fit". The program then:
   - optimises the weights of multi-atom sites (for example, it finds the +1 -1
     antisymmetric tip states of triangular edges on its own);
   - projects the TB bands on the sites (exact effective Hamiltonian) and keeps every
     coupling above the threshold ("keep > meV"), up to "max cells R" cells away;
   - gives couplings with equal values (within "group within meV", up to a sign) one
     shared parameter, which detects the symmetries of the structure;
   - writes the model into the editor (each line has the projected value as a comment)
     and fits it.
5. Edit the generated model if you want (rename or merge parameters, delete small
   terms) and press FIT again. "Sites from editor" loads the sites of a model file back
   into the picker.

If the panel reports a LOW overlap, the sites miss some of the band states: pick more
atoms or other sites.

### Write or edit a model by hand

Write (or load) an effective-model scheme, then press FIT. The model bands are fitted by
least squares to the tight-binding bands. If every site sits on atoms, the starting values
come from an exact projection (Loewdin-orthogonalised projection of the TB bands on the
sites). The same projection also lists the couplings that are missing from your scheme,
so you can see which terms to add.

    site  T  near 3.689 1.42                   # site on the atom nearest to (x, y)
    site  T  near 2.46 0.71  4.92 0.71  weights 1 -1   # combination of two atoms
    site  B  atom 65                           # or by atom index
    onsite T eT
    hop   T B 0 tau                            # <T, cell 0|H|B, cell 0> = tau
    hop   B T 1 -tau                           # <B, cell 0|H|T, cell 1> = -tau
    param tau 0.03                             # optional start value ('fix' to freeze)
    bands -1 2                                 # bands to fit, 0 = HOMO band (default: centred on E_F)
    bands auto                                 # or: at each k the states living on the sites

Shared parameter names impose symmetries; a leading `-` flips the sign.
The fit uses as many bands as there are sites. Clicking an atom in the plot inserts its
coordinates at the cursor.

### Band crossings

The fit compares, at every k, the sorted energies of the model with the sorted energies of
the selected TB bands. It never follows a band "by label", so a change of order where two
bands cross is not ambiguous for the fit. Two things can still go wrong, and the FIT report
checks both (section "band crossings and band selection"; marks on the band plot):

- **Band selection.** With an index window ("bands -1 2"), the fitted bands are the ones
  with those indices at every k. If a band of another character (for example a flat band
  of the other mirror symmetry) sits inside the window while a band with more weight on
  the sites is outside, the fit follows the wrong band on part of the zone. The report
  warns and the plot shades those k in red. Use `bands auto`, which picks at every k the
  bands with most weight on the sites. Example: polyparaphenylene, rms 207 meV with the
  index window and 0 with `bands auto`.
- **Crossing or avoided crossing.** In one dimension two bands can only cross when they
  belong to different eigenvalues of a symmetry that keeps k (mirror across the chain axis,
  or glide); otherwise they repel and a gap opens (avoided crossing). The report finds the
  near-degeneracies of the fitted bands and classifies them with the symmetries found in the
  structure: protected crossing (green circle), avoided crossing (orange), close approach of
  bands of different parity, or degeneracy at k = 0 or pi (a glide makes the bands stick in
  pairs at k = pi, as in the goblet chains). At every TB crossing it checks the model: if the
  model has a gap there, either it crosses somewhere else (it has the symmetry but lacks the
  terms that place the crossing, e.g. polyacene fitted with nearest-neighbour hopping only)
  or it does not cross at all (its sites or hoppings couple the two symmetry sectors).

## Examples

| structure | cell | model | rms error (1st+2nd+3rd NN) |
|---|---|---|---|
| polyparaphenylene (`.cdxml`, `.mol`) | 6 C | 4-site chain P-Q-Q-P (exact mapping of the mirror-symmetric bands) | 0 meV |
| goblet_chain | 190 C | 4 tips + 4 backbone sites | 5.2 meV |
| triangulene_goblet (picture 1) | 100 C | 2 tips + 2 backbone sites | 6.1 meV |
| triangle_ribbon (picture 2) | 72 C | 2 tips + 2 backbone sites | 5.5 meV |
| trianthracene_ribbon (picture a) | 84 C = 2 x 42 C precursor | 2 tips + 2 backbone sites | 5.0 meV |

In the triangular heads the tip state is antisymmetric between the two corner atoms next
to the apex (`weights 1 -1`). It therefore couples to the backbone sites of the neighbouring
protrusions with opposite signs (+tau, -tau), and not to its own backbone site. The
anthracene tip is symmetric and couples directly to its own backbone site.

`examples/chemdraw_exports/` has the four ribbons as ChemDraw files, with star atoms on the
periodic bonds.

## Tab 3: Finite chain

Finite piece of the ribbon with open ends, with the same tools as tab 1.

- **Build**: choose the number of units; the chain is made from the periodic structure of
  tab 1 (including its sp3 edits) with the hoppings t1, t2, t3 of tab 1. "cut shift (A)"
  moves the place where the ends are cut, to try different terminations; atoms left with
  a single neighbour at the ends are removed (optional).
- **Or load a finite molecule file** (.cdxml, .mol, .xyz, .hex) as it is: no period is
  searched, sp3 carbons of a Kekule drawing are still detected.
- **Levels and DOS**: the discrete energy levels (black: occupied) replace the bands, with
  the DOS beside them. The info box gives the HOMO-LUMO gap, the number of levels near
  E_F and the sublattice balance (each unit of imbalance gives one zero mode with NN hopping).
- **dI/dV map and spectra**: as in tab 1 (zoom, "auto" broadening, exact map energy); click the
  levels or the DOS to choose the energy, click atoms for point spectra. End states show up
  here. "map: clicked level" shows |psi|^2 of the level clicked (orange).
- **sp3 / vacancies**: set the click mode to "remove (sp3)"; in the finite chain only the
  clicked atom is removed. Undo, Restore all, Save as .xyz or .cdxml.
- **Export (CSV)**: DOS and LDOS of every atom, plus a file with the levels.

- **Symmetric ends.** "Symmetric ends: next option" tries every cut between atom columns and
  every extra length up to one cell, and keeps the pieces whose two ends are equivalent (mirror
  or C2); each click shows the next one and fills in the cut shift and the extra length. For
  the 7-AGNR with 4 units the options are 52 atoms (two-ring columns at the ends) and 56 atoms
  (three-ring columns). "extra length" can also be typed by hand. When the chain is made of
  fused-ring units linked by single bonds (olympicene chain), only pieces made of whole units
  are offered, and the status line gives the number of units: with a dimer cell this is the
  way to get odd chains (1 cell: 2 or 3 units, 38 or 57 atoms; 2 cells: 4 or 5 units).
- **Mean-field Hubbard of the finite system** (no sites to choose, as in tab 5): the states
  found are listed with their energies; choose one in "show" to see its spin-up and spin-down
  levels, spin DOS, spin-density map and spin-resolved dI/dV spectra at the clicked atoms
  (up solid, down dashed). The map can also show the dI/dV of the mean-field state at the map
  energy: up + down (what a normal tip measures), up, down, up - down (a spin-polarised tip),
  or |psi|^2 of the clicked level (the left column of the levels is spin up, the right spin
  down). Clicking the levels while the spin density is shown switches to the dI/dV map. "tight binding (U = 0)" goes back to the plain levels. The output
  also gives the **singlet-triplet gap from the mean field** (see tab 5). The same calculation,
  with plots of all the states, is in tab 5 in "finite chain" mode.

## Tab 4: Spin model (interacting effective model)

Physics (details in the docstring of `tbmany.py`):

- **Projected interaction.** The effective sites are Wannier-like orbitals w_i (Loewdin
  projection of the fitted TB bands). The atomic Hubbard interaction U n_up n_dn on every
  carbon, projected on them, gives U_i = U sum_x |w_i(x)|^4 on each site (a more spread
  orbital feels a smaller U) and small inter-site terms (density, direct exchange, pair
  hopping), all U sum_x w_a w_b w_c w_d. "on-site U only" keeps U_i, "full projected
  interaction" keeps every term above the threshold. U_atom ~ 3 eV is the usual value for
  graphene pi electrons; it is an upper bound (screening by the other bands is not included).
- **Link with tab 2.** Box 1 lists the fitted parameters of tab 2 and where each one enters
  the ED (on-site energies eps_i measured from E_F, hoppings t_ij). Tab 4 always uses the
  current tab 2 model: if the editor changed, it is refitted and the list is updated. The
  interaction is Hubbard (U_i on every site), written as U(n_up - 1/2)(n_dn - 1/2)
  (particle-hole symmetric) so that the electrostatics already in the TB fit is not counted
  twice.
- **Hamiltonians.** Each box shows its Hamiltonian, and the text output prints it with the
  numbers used: the ED Hamiltonian (eps_i, t_ij with the parameter names of tab 2, U_i, and
  the largest inter-site terms in the full mode), the fitted spin Hamiltonian written bond
  by bond (J1 (S_Tu0.S_Td0 + ...)), and the mean-field Hamiltonian (atomistic, t1 t2 t3 of
  tab 1: it does not use the tab 2 model and is an independent check).
- **Exact diagonalisation** of the effective model on a cluster of unit cells (open chain
  or ring), at the neutral filling by default (the electrons of the fitted bands: one per
  site only when half of the model bands are occupied, e.g. not with an odd number of
  electrons per cell): many-body spectrum with total spin S, charges, local moments
  <S_i^2> (0.75 = free spin) and spin correlations <S_i.S_j>. Up to about 12 sites
  (dimension ~ 10^6, a minute or two). Degenerate eigenvectors are rotated to states of
  definite S.
- **Odd chains (partial last cell).** "last cell keeps sites" (blank = all) drops the other
  sites of the last cell. With a dimer cell (sites A, B, e.g. the olympicene chain) 3 cells
  and "A" give A0 B0 A1 B1 A2: 5 units, 5 electrons, an S = 1/2 ground state. The hoppings and
  U_i are those of the full cluster (the dropped sites are simply removed), the neutral
  electron count is scaled to the sites kept, and the drawing shows only the units of the
  kept sites. Open chains only. The same entry is in tab 7.
- **Transitions (click a level in the spectrum).** For the transition from the ground state
  to the clicked level: Delta E, Delta S and the selection rule of inelastic (spin-flip)
  tunnelling, Delta S = 0, +-1 (not 0 -> 0); the spin-flip weight
  W_i = sum_{a=x,y,z} sum_{M_f} |<f M_f| S_i^a |0>|^2 (unpolarised tip, zero field; by the
  Wigner-Eckart theorem it does not depend on the member of the ground multiplet), per
  effective site and per atom (S_x projected on the effective orbitals,
  S_x = sum_ab w_a(x) w_b(x) S_ab, drawn as an orange simulated map of the inelastic step);
  the change of the spin correlations <S_i.S_j> (which bonds break); and the label spin
  excitation (same charges and local moments), spin* (local moments change) or charge
  excitation (electrons move). After every ED the text box lists all the transitions with
  their weights, and the spectrum shows them as orange dots (size ~ W). Click the ground
  state to go back to its correlations.
- **Spin model** H = sum J_ij S_i.S_j on the ticked spin sites, fitted to the lowest ED
  states (one state per spin configuration). Bonds related by a symmetry of the structure
  (translation, mirror, glide, C2; found automatically) share a J by default; rename
  couplings to merge or split them. The energies alone do not fix which bond carries which J
  (relabelling the spins leaves the spectrum unchanged, and an exact fit can put the singlets
  on the wrong bonds), so the ground-state correlations <S_i.S_j> are fitted too. The program
  warns when the lowest states are not an isolated spin manifold (the chosen sites are not
  the only moments), when some ED states have a spin the model cannot give, when some J are
  not determined, when the fit is poor, or when the Heisenberg correlations differ from the
  ED ones (couplings that depend on the position in a small cluster). Analytic estimates:
  superexchange 4t^2/U between spin sites and 8 tau^2/U between a spin site and another
  orbital.
- **Mean-field Hubbard** on the full atomistic model (benchmark): ferromagnetic and
  antiferromagnetic starts on the atoms of the spin sites (AFM by sublattice, or alternating
  the spin sites when they are all on one sublattice), spin-density map and spin resolved
  bands. With one spin site per cell FM and AFM are the same state (an AFM order between
  cells needs a doubled cell). Mean field breaks the spin symmetry and usually overestimates
  couplings.

Workflow: load a structure (tab 1), fit its effective model (tab 2), then in tab 4 press
"Load sites from the model (tab 2) and compute U_i", tick the spin sites, "Run ED",
click the levels of the spectrum, "Fit J to the ED spectrum", and "Run mean-field Hubbard"
to compare.

Cluster size and boundary conditions matter when the moments couple through a metallic
band: a small ring can have a degenerate Fermi level (e.g. 4 backbone sites) and give much
larger couplings than an open chain. Compare open chains and rings of 2 and 3 cells.

Example, real goblet chain (U = 3 eV): U_i = 0.42 eV on the radical heads (orbital on ~7
atoms) and 0.25 eV on the backbone (~17 atoms). ED (open chains of 2 and 3 cells): the heads
are almost free spins, <S^2> = 0.75, all head-head couplings below ~0.06 meV, spin manifold
separated by ~130-180 meV from the other excitations. 2-cell open chain, spin model fitted
with the correlations: J(Tu-Td, 27.6 A) = +0.035 meV (AFM, opposite sublattices),
J(Tu-Tu = Td-Td, 17.2 A, related by the glide) = -0.009 meV (FM, same sublattice), as
expected from Lieb's rule. The first triplet (0.026 meV) has spin-flip weight on all four
heads; the singlet at 0.078 meV and the quintet are dark. Mean-field Hubbard: 0.4 meV per cell
(AFM) on the effective model, 8.5 meV per cell on the atomistic model at U = 3 eV, where the
backbone also polarises (1.0 meV at U = 1 eV).

## Tab 5: Mean-field Hubbard

Atomistic mean-field Hubbard on the structure of tab 1, H_s = H_TB + U sum_x <n_x,-s> n_x,s on
every carbon (collinear, self-consistent, same conventions as the `hubbard` package). No
sites have to be chosen: for each total S_z per cell ("auto": the smallest allowed, the
sublattice imbalance |N_A - N_B|/2 of Lieb's theorem, and one more) two starts are run,
spins by sublattice and all spins up, and the lower solution is kept; a non-magnetic
solution is always added. The log lists the states with E - E_min per cell and the largest
moment; the plots show E versus S_z (click a state), its spin density, its spin-resolved bands
and DOS. Real goblet chain, U = 3 eV: S_z = 0 lowest, S_z = 1 at +4.7 meV per cell,
non-magnetic at +279 meV.

### Spatial projection of the mean-field states

The box "Spatial projection" gives, for the state shown, the same maps and spectra as tab 1 but
spin resolved. Each spin has its own Hamiltonian H_s = H_TB + U diag(<n_-s>), diagonalised on a
dense k grid (the bands are drawn from it, not from the k points of the self-consistency) or, in
finite mode, its levels. Map choices: spin density; dI/dV up + down, up, down or up - down at the
map energy; the clicked band (averaged over k or at the clicked k) or level. Click the bands, the
levels or the DOS to choose the energy and the band (the spin of the band nearest to the click;
in finite mode the left column is spin up, the right spin down); click atoms on the map for the
spin-resolved spectra (up solid, down dashed). Energy window, broadening ("auto" as in tab 1) and
zoom work as in tab 1. "Export" saves the DOS and LDOS of every atom for both spins.

### Finite chain mode and the singlet-triplet gap

"finite chain / molecule (tab 3)" runs the same scan on the finite piece built (or loaded) in
tab 3: no k points, S_z of the whole piece, spin-resolved levels instead of bands. The mean field
gives broken-symmetry determinants, not spin eigenstates, so the singlet-triplet gap is given
three ways (in general: from the lowest S_z to S_z + 1; a negative value means the higher spin
is lower):

- **raw**: E(S_z + 1) - E(S_z) of the mean-field states;
- **spin-projected (Yamaguchi)**: Delta = (E_HS - E_LS) [S_h(S_h+1) - S_l(S_l+1)] / (<S^2>_HS - <S^2>_LS),
  with <S^2> = S_z(S_z+1) + N_dn - Tr(P_up P_dn) of each determinant. For two separated radicals
  <S^2>_BS = 1 and Delta = 2 (E_T - E_BS); for a closed-shell singlet Delta = E_T - E_S. Use it with
  **two radicals**;
- **Heisenberg mapping**: the spin centres (radicals) are found in the spin density (units linked
  by single bonds, or the oppositely polarised ends of a fused ribbon); extra mean-field runs start
  from the lowest state with the moments of some centres flipped (every sign pattern with the two
  S_z, closest first). The energies of the collinear states are fitted with an Ising model
  E = E0 + sum J_ab s_a s_b of nearest centres, and H = sum J_ab S_a.S_b is diagonalised exactly.
  Use it with **more than two radicals**, where the two-state projection is not valid.

Checks (tests/validate_many.py): <S^2> = brute force in the Fock space; Hubbard dimer
(t = 50 meV, U = 0.566 eV): projected 17.13 meV, exact J 17.15 meV; as-indacene (12 C, t1 only,
U = 3 eV): projected 53.2 meV, exact Hubbard ED 50.0 meV. When the radicals sit on the same
sublattice and share atoms (m-xylylene) the triplet ground state comes out right but the size is
overestimated several times (-1.08 eV vs -0.27 eV exact): the output warns.

Olympicene chains (U = 3 eV), mean field vs ED of the reduced model with the U of tab 6:

| units | mean field: raw | projected / Heisenberg | ED, reduced model (U = 0.566 eV) |
|---|---|---|---|
| 2 | 18.2 meV | S-T 35.5 meV | S-T 17.2 meV |
| 3 | 36.1 meV | first (S = 1/2) 36.1, S = 3/2 at 54.1 meV | 17.0, 25.7 meV |
| 4 | 17.8 meV | S-T 23.7 meV | S-T 11.3 meV |
| 5 | 35.6 meV | first (S = 1/2) 25.9, S = 3/2 at 40.2 meV | 12.3, 19.2 meV |

The mean-field coupling is about twice the one of the reduced model (J about 36 vs 17 meV): the U
of tab 6 reproduces the mean-field bands, not the mean-field energies (the tab 6 log shows it,
E_FM - E_AFM = 35.8 meV atomistic vs 17.5 meV reduced). The atomistic mean field also contains the
spin polarisation of the other pi electrons, which the two-orbital model lacks; mean field is
also known to overestimate magnetic couplings. Which number is closer to the full Hubbard model
can only be settled by a larger active space.

## Tab 6: U fit to the mean field

The U of the reduced model is fitted instead of projected:

1. **t from tab 2.** The reduced model and its fitted hoppings / energies are loaded and kept
   fixed. The table sets the U groups (sites with the same name share one U; default: sites
   with the same on-site parameter).
2. **Reference states** from tab 5 (the magnetic ones are ticked by default). For each spin
   the bands of the tab 2 window are kept, and the occupations are projected on the effective
   orbitals (moments m_i).
3. **Fit of U_i with t fixed.** The reduced model gets its own self-consistent mean field,
   H_s(k) = H_t(k) + sum_i U_i (<n_i,-s> - 1/2) n_i,s, with the same S_z per cell as the
   atomistic state (the extra spin of the cell sits in the model bands). The U_i are fitted
   so that its spin-up and spin-down bands match the atomistic ones in all ticked states
   (one rigid offset per state, the Hartree shift, about U_atom/2), optionally with the site
   moments. Several starts are used because the mean field can switch between solutions.
   The report gives the fitted U_i next to the projected U sum|w|^4, the sensitivity of the
   bands to each U (a site that does not polarise does not determine its U), the moments,
   the rms per state and the energies of the states, atomistic and reduced. The panels show
   the bands of the first two states, the moments and the two sets of U.
4. "Use these U_i in tab 7" (ED, spectra, maps) or "in tab 4" (spin model).

Real goblet chain (U_atom = 3 eV, states S_z = 0 and 1 of tab 5): U = 0.57 eV on the heads and
0.39 eV on the backbone (projected: 0.42 and 0.25); band rms 6-7 meV (the level of the t fit).

## Tab 7: ED, spectra and maps

Exact diagonalisation of H = sum eps_i n_is + sum t_ij c+_is c_js + sum U_i (n_i,up - 1/2)
(n_i,dn - 1/2) on a cluster of cells (t from tab 2; U_i fitted in tab 6, projected
U_atom sum|w|^4, or typed by hand), and what a tip would measure. "last cell keeps sites"
makes odd chains (see tab 4), e.g. 4 cells keeping A = 7 olympicene units. Every step, with
the equations, checks against brute-force diagonalisation and worked examples (olympicene chains
of 4 and 5 units, goblet chain): `notebooks/panel7_ED_explained.ipynb`.

- **Many-body spectrum** (energies, total S, orange dots = spin-flip weight). Click a level:
  the map shows its spin-flip weight on the atoms (the inelastic dI/dV step map), per site,
  and the change of the correlations <S_i.S_j>.
- **Tip position.** Click an atom of the map (or choose an effective site): the two lower
  panels are the spectra there. An atom x is seen through the effective orbitals,
  c_x = sum_a w_a(x) c_a.
- **Inelastic (spin-flip) spectrum** at the tip: steps at +-Delta E_f of height
  W_f(x) = sum_a sum_Mf |<f|S^a_x|0>|^2 (IETS; width "inel." in meV, auto = 3 % of the
  first excitation).
- **Charge excitations** ("Compute charge excitations"): lowest states with N - 1 and N + 1
  electrons and their Dyson amplitudes d_f(a) = <f|c+_a|0> or <f|c_a|0>. The tip spectrum is
  A_x(E) = sum_f |sum_a w_a(x) d_f(a)|^2 delta(E - E_f), with addition energies
  E(N+1) - E0(N) above E_F and removal energies E0(N) - E(N-1) below (Hubbard peaks, charge
  gap). Click a peak: the map is |Dyson orbital|^2 on the atoms, the constant-height STM map
  at that bias. Dark states (e.g. spin 3/2 states reached from a singlet) are skipped. Only
  the lowest states of each sector are computed ("states / sector"), so the spectrum is
  complete near the gap; the log gives the range.

Checks: the spectral weights obey the sum rules (sum over the N - 1 states = <n_a>, over the
N + 1 states = 2 - <n_a>), reduce to the single-particle levels for U = 0, and agree with
OpenFermion in the full Fock space (1e-13).

### Validation

`python tests/validate_many.py` checks the many-body code against exact results (Hubbard
dimer, Heisenberg chains and rings, Lieb theorem), against OpenFermion (exact fermionic
Hamiltonians, including the full projected interaction; agreement 1e-13 eV) and against the
mean-field Hubbard package `hubbard` of DIPC (energies and spin densities, finite and
periodic; agreement 1e-9). The spin-flip weights (per site and per atom) are checked against
OpenFermion in the full Fock space (1e-14) and with the sum rule sum_f W_i = <S_i^2>; the
spin-model fit is checked to recover known couplings and the ED correlation pattern. OpenFermion and hubbard are optional, only for these checks:
    pip install openfermion
    pip install git+https://github.com/dipc-cc/hubbard.git
