"""f21: where a muon's helix crosses a pixel surface (HITEMU 3.6). Uses the
repository's own crossing functions (pixel_eff_maps.cross_barrel / cross_disk)."""
import os, sys
import numpy as np, matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['mathtext.default'] = 'regular'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10.5})
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', '..'))
import pixel_eff_maps as P
G = P.GEOM; B = G['B_FIELD']
BL, OR, GR, GY, RD = '#2a5a9a', '#b5651d', '#1a7f37', '#8a8f98', '#b03030'
a = lambda v: np.array([v], float)

def path(pt, eta, phi0, q, x0, y0, z0, amax, n=400):
    R, cx, cy = P.helix_frame(a(pt), a(phi0), a(q), a(x0), a(y0), B)
    al = np.linspace(0, amax, n)
    x, y = P.helix_xy(al, q, x0, y0, cx[0], cy[0])
    return x, y, z0 + R[0] * al * np.sinh(eta)

fig, ax = plt.subplots(1, 2, figsize=(12, 5.4), gridspec_kw=dict(width_ratios=[1, 1.35], wspace=0.25))

# (a) transverse plane, exaggerated curvature
A = ax[0]
for r in G['BPIX_R'].values():
    t = np.linspace(0, 2 * np.pi, 400); A.plot(r * np.cos(t), r * np.sin(t), color=GY, lw=1)
r1 = G['BPIX_R'][1]
for k in range(48):                       # the 48 phi cells of L1
    f = -np.pi + k * 2 * np.pi / 48
    A.plot([0.93 * r1 * np.cos(f), 1.07 * r1 * np.cos(f)], [0.93 * r1 * np.sin(f), 1.07 * r1 * np.sin(f)], color=GY, lw=0.6)
pt, phi0, q, x0, y0 = 0.25, 0.75, +1, 0.6, -0.4          # exaggerated: low pT, offset PV
x, y, _ = path(pt, 0.5, phi0, q, x0, y0, 0.0, 1.35)
A.plot(x, y, color=BL, lw=2, label='helix, q > 0 (turns clockwise)')
A.plot([x0, x0 + 13 * np.cos(phi0)], [y0, y0 + 13 * np.sin(phi0)], color=BL, ls=':', lw=1.2, label='direction φ$_0$ at the start')
A.plot(x0, y0, 'k*', ms=11, label='start: PV (x$_0$, y$_0$)')
zc, fc = P.cross_barrel(r1, a(pt), a(0.5), a(phi0), a(q), a(x0), a(y0), a(0.0), B)
A.plot(r1 * np.cos(fc), r1 * np.sin(fc), 'o', color=RD, ms=8, label='crossing with L1: φ$_L$')
A.plot([0, r1 * np.cos(fc[0])], [0, r1 * np.sin(fc[0])], color=RD, lw=1)
A.set_aspect('equal'); A.set_xlim(-6, 17.5); A.set_ylim(-6, 17.5)
A.set_xlabel('x [cm]'); A.set_ylabel('y [cm]')
A.set_title('(a) transverse plane: BPix L1–L4\n(curvature exaggerated: p$_T$ = 0.25 GeV, PV offset 7 mm)', fontsize=10.5)
A.legend(fontsize=8.5, loc='upper left', frameon=False)

# (b) r-z view with the real numbers of the two worked examples
A = ax[1]
for k, r in G['BPIX_R'].items():
    A.plot([-G['BPIX_HALF_Z'], G['BPIX_HALF_Z']], [r, r], color=GY, lw=2)
    A.text(-G['BPIX_HALF_Z'] + 0.3, r + 0.3, 'L%d' % k, ha='left', va='bottom', fontsize=9, color=GY)
for k, z in G['FPIX_Z'].items():
    for sgn in (-1, 1):
        A.plot([sgn * z] * 2, list(G['FPIX_R']), color=GY, lw=2)
    A.text(z, G['FPIX_R'][1] + 0.6, 'D%d+' % k, ha='center', fontsize=9, color=GY)
    if k == 1: A.text(-z, G['FPIX_R'][1] + 0.6, 'D1−', ha='center', fontsize=9, color=GY)
for zz in np.arange(-40, 41) * G['ROC_PITCH_Z']:       # L1 cells in z: one ROC pitch
    if abs(zz) < G['BPIX_HALF_Z']:
        A.plot([zz, zz], [r1 - 0.35, r1 + 0.35], color=GY, lw=0.5)
EX = [dict(pt=5.0, eta=0.5, phi0=1.0, q=+1, z0=2.0, c=BL, lab='example 1: p$_T$ 5 GeV, η 0.5, z$_0$ 2 cm'),
      dict(pt=4.0, eta=2.0, phi0=-2.0, q=-1, z0=1.0, c=OR, lab='example 2: p$_T$ 4 GeV, η 2.0, z$_0$ 1 cm')]
for e in EX:
    x, y, z = path(e['pt'], e['eta'], e['phi0'], e['q'], 0, 0, e['z0'], 0.06)
    r = np.hypot(x, y); m = (r < 17.5) & (np.abs(z) < 55)
    A.plot(z[m], r[m], color=e['c'], lw=1.8, label=e['lab'])
    for k, rl in G['BPIX_R'].items():
        zl, _ = P.cross_barrel(rl, a(e['pt']), a(e['eta']), a(e['phi0']), a(e['q']), a(0), a(0), a(e['z0']), B)
        if np.isfinite(zl[0]) and abs(zl[0]) < G['BPIX_HALF_Z']:
            A.plot(zl, [rl], 'o', color=e['c'], ms=6)
    for k, zd in G['FPIX_Z'].items():
        rd, _ = P.cross_disk(zd, a(e['pt']), a(e['eta']), a(e['phi0']), a(e['q']), a(0), a(0), a(e['z0']), B)
        if np.isfinite(rd[0]) and G['FPIX_R'][0] < rd[0] < G['FPIX_R'][1]:
            A.plot([zd], rd, 's', color=e['c'], ms=6)
A.plot([2.0, 1.0], [0, 0], 'k*', ms=10)
A.set_xlabel('z [cm]'); A.set_ylabel('r [cm]'); A.set_xlim(-30, 56); A.set_ylim(0, 17.5)
A.set_title('(b) longitudinal view: z = z$_0$ + s$_T$ sinh η; dots = crossings in acceptance\n(L1 ticks: the cells in z, one ROC = 0.83 cm)', fontsize=10.5)
A.legend(fontsize=8.5, loc='lower right', frameon=True)
fig.savefig(os.path.join(HERE, 'f21_helix_crossing.png'), dpi=150, bbox_inches='tight')
