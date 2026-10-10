"""f22: why covflow works with canonical partial correlations (HITEMU 7.5)."""
import os
import numpy as np, matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['mathtext.default'] = 'regular'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10.5})
HERE = os.path.dirname(os.path.abspath(__file__))
fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.9), gridspec_kw=dict(wspace=0.3))

A = ax[0]
g = np.linspace(-1, 1, 801); X, Y = np.meshgrid(g, g)
cols = ['#2a5a9a', '#b5651d', '#1a7f37']
for r32, c in zip((0.0, 0.6, 0.9), cols):
    det = 1 - X**2 - Y**2 - r32**2 + 2 * X * Y * r32           # det of the 3x3 correlation matrix
    A.contour(X, Y, det, levels=[0], colors=[c], linewidths=2)
    A.plot([], [], color=c, lw=2, label='ρ$_{32}$ = %.1f' % r32)
A.plot([-1, 1, 1, -1, -1], [-1, -1, 1, 1, -1], color='k', lw=1)
A.set_aspect('equal'); A.set_xlim(-1.08, 1.08); A.set_ylim(-1.08, 1.08)
A.set_xlabel('ρ$_{21}$'); A.set_ylabel('ρ$_{31}$')
A.set_title('(a) 3×3: valid (ρ$_{21}$, ρ$_{31}$) lie inside each curve, not in the\nwhole square. Independent draws in (−1, 1) are valid\nin 62% of the cases for 3×3, in 2.2% for 5×5', fontsize=10.5)
A.legend(fontsize=9, loc='upper center', bbox_to_anchor=(0.5, -0.13), ncol=3, frameon=False)

A = ax[1]
t = np.linspace(-4, 4, 400)
A.plot(t, np.tanh(t), color='#2a5a9a', lw=2)
A.axhline(1, color='#8a8f98', ls=':'); A.axhline(-1, color='#8a8f98', ls=':')
for v in (-2.0, 0.55, 1.6):
    A.plot([v, v], [-1.1, np.tanh(v)], color='#b5651d', lw=1, ls='--')
    A.plot([-4, v], [np.tanh(v)] * 2, color='#b5651d', lw=1, ls='--')
    A.plot(v, np.tanh(v), 'o', color='#b5651d')
A.set_xlim(-4, 4); A.set_ylim(-1.1, 1.1)
A.set_xlabel('flow feature y = atanh z  (any real number)')
A.set_ylabel('canonical partial correlation z')
A.set_title('(b) each feature is free on the whole real line;\nz = tanh y is always in (−1, 1)', fontsize=10.5)
fig.savefig(os.path.join(HERE, 'f22_cholesky_cpc.png'), dpi=150, bbox_inches='tight')
