"""f23: per-sample helix start points and the generated trajectory (HITEMU 3.8).
The numbers of (b), (c) are the output of hitemu_selftest/check_origins.py
(run_selftest.sh, 60k toy events per sample, 11 Oct 2026)."""
import os
import numpy as np, matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, FancyArrowPatch
plt.rcParams['mathtext.default'] = 'regular'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10.5})
HERE = os.path.dirname(os.path.abspath(__file__))
C_PV, C_VX, C_TRK, C_GEN = '#1a7f37', '#2a5a9a', '#5b5f66', '#b5651d'

fig = plt.figure(figsize=(13.5, 4.9))
gs = fig.add_gridspec(1, 3, width_ratios=[1.25, 1, 1], wspace=0.38)

# (a) schematic, transverse plane, not to scale
A = fig.add_subplot(gs[0])
A.add_patch(Ellipse((0, 0), 1.2, 0.8, fc='#e8f3ea', ec=C_PV, lw=1.2, ls='--'))
A.plot(0, 0, 'x', color=C_PV, ms=10, mew=2.5)
A.text(0.0, -0.62, 'PV,  3σ of $L_{xy}$', color=C_PV, ha='center', fontsize=9.5)
# prompt-like candidate: vertex inside the 3 sigma region
pv_ = np.array([0.22, 0.12])
A.plot(*pv_, 'o', color=C_VX, ms=7)
for ang in (0.55, 1.35):
    d = np.array([np.cos(ang), np.sin(ang)])
    A.annotate('', xy=pv_ + 2.2 * d, xytext=pv_, arrowprops=dict(arrowstyle='->', color=C_TRK, lw=1.4))
A.text(-0.9, 2.75, '$lxy_{sig}$ < 3: helix starts at the PV\n(pv$_x$, pv$_y$, pv$_z$ + dz)', color=C_PV,
       fontsize=9.5, ha='left')
# displaced candidate
dv = np.array([2.6, -0.9])
A.plot([0, dv[0]], [0, dv[1]], color=C_VX, lw=1, ls=':')
A.plot(*dv, 'o', color=C_VX, ms=7)
for ang in (-0.05, 0.45):
    d = np.array([np.cos(ang), np.sin(ang)])
    A.annotate('', xy=dv + 2.0 * d, xytext=dv, arrowprops=dict(arrowstyle='->', color=C_TRK, lw=1.4))
A.text(1.55, -1.75, '$lxy_{sig}$ ≥ 3: helix starts at the\ndimuon vertex (vx, vy, vz)', color=C_VX, fontsize=9.5)
A.text(1.0, -0.3, '$L_{xy}$', color=C_VX, fontsize=9.5)
A.set_xlim(-1.0, 5.2); A.set_ylim(-2.2, 3.6); A.set_aspect('equal'); A.axis('off')
A.set_title('(a) data with --helix-origin-data pv_or_vertex\n(not to scale: the PV region is tens of µm,\nL1 is at 2.9 cm)',
            fontsize=10.5)

# (b), (c): L1 probes whose cell differs from the cell of the TRUE trajectory
rows = {
    'data': dict(n_disp=26.4, o=['dimuon_vertex', 'pv', 'track', 'pv_or_vertex'],
                 lo=[1.617, 1.587, 1.619, 1.587], hi=[1.563, 10.414, 1.613, 1.563]),
    'MC': dict(n_disp=84.1, o=['dimuon_vertex', 'pv', 'track', 'pv_or_vertex', 'gen_kin', 'gen'],
               lo=[1.586, 1.833, 1.586, 1.833, 0.914, 0.0], hi=[1.557, 10.703, 1.560, 1.557, 0.954, 0.0]),
}
for k, (name, r) in enumerate(rows.items()):
    B = fig.add_subplot(gs[k + 1])
    x = np.arange(len(r['o']))
    B.bar(x - 0.2, r['lo'], 0.4, color=C_PV, label='$lxy_{sig}$ < 3 (prompt-like)')
    B.bar(x + 0.2, r['hi'], 0.4, color=C_VX, label='$lxy_{sig}$ ≥ 3 (displaced)')
    for xi, lo, hi in zip(x, r['lo'], r['hi']):
        B.text(xi - 0.2, lo + 0.15, '%.1f' % lo, ha='center', fontsize=7.5)
        B.text(xi + 0.2, hi + 0.15, '%.1f' % hi, ha='center', fontsize=7.5)
    B.set_xticks(x); B.set_xticklabels(r['o'], rotation=35, ha='right', fontsize=9)
    B.set_ylim(0, 12.5); B.set_ylabel('L1 probes in another cell than the true one [%]')
    B.set_title('(%s) toy %s (%.0f%% displaced)' % ('bc'[k], name, r['n_disp']), fontsize=10.5)
    B.axhline(1.6, color='#8a8f98', ls=':', lw=1)
    if k == 0:
        B.legend(fontsize=8.5, loc='upper right', frameon=False)
        B.text(3.45, 2.3, 'floor: reco\nmomentum\nresolution', fontsize=8, color='#5b5f66', ha='right')
fig.savefig(os.path.join(HERE, 'f23_origins_per_sample.png'), dpi=150, bbox_inches='tight')
