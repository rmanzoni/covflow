#!/usr/bin/env python3
"""Self-test of the helix origins (HITEMU 3.6-3.8) on the toy samples of
gen_fake_masks.py, where the true trajectory is known.

For every origin, the L1 kill-map cell (48 phi x 0.8325 cm ROC bins) of each
muon is compared with the cell of the TRUE trajectory, separately for
prompt-like (lxy_sig < 3) and displaced candidates. Then the meta of the kill
maps given on the command line is printed and checked.

Usage: python3 hitemu_selftest/check_origins.py FAKE_DIR KM_STEM [KM_STEM ...]
"""
import argparse
import json
import os
import sys

import numpy as np
import uproot

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import hitemu as H                    # noqa: E402
import pixel_eff_maps as P            # noqa: E402

fake, stems = sys.argv[1], sys.argv[2:]
xe, ye = H.surface_edges('L1', 48, 20)
CUT = 3.0


def cell(pt, eta, phi, q, x0, y0, z0):
    x, ph = H.cross('L1', pt, eta, phi, q, x0, y0, z0)
    ix, iy, ok = H.cell_index(xe, ye, x, ph)
    return np.where(ok & H.in_acceptance('L1', x), ix * 1000 + iy, -1)


def args(data, mc):
    return argparse.Namespace(branch=[], helix_origin=data, helix_origin_data=data,
                              helix_origin_mc=mc, origin_sig_cut=CUT)


fail = []
for sample, fname, origins, truth in (
        ('data', 'data2026B.root', ('dimuon_vertex', 'pv', 'track', 'pv_or_vertex'), 'truth'),
        ('mc', 'hb_Summer24.root', ('dimuon_vertex', 'pv', 'track', 'pv_or_vertex',
                                    'gen_kin', 'gen'), 'gen')):
    with uproot.open(os.path.join(fake, fname)) as fh:
        arr = fh['tree'].arrays(library='np')
    sel = np.ones(len(arr['run']), bool)
    disp = arr['lxy_sig'] >= CUT
    print('\n%s: %d candidates, %.1f%% with lxy_sig >= %g' % (sample, sel.sum(), 100 * disp.mean(), CUT))
    print('  %-14s %26s %26s %10s' % ('origin', 'other cell, lxy_sig < 3', 'other cell, >= 3', 'from PV'))
    for o in origins:
        a = args('dimuon_vertex' if o in P.GEN_ORIGINS else o, o)
        tm, _, _ = P.branch_templates(a, ['L1'], sample)
        res = []
        frompv = []
        for mu in ('mu1', 'mu2'):
            v = P.muon_view(arr, mu, tm, sel, sig_cut=CUT)
            g = lambda k: np.asarray(arr['%s_%s_%s' % (mu, truth, k)], np.float64)
            qt = -np.sign(g('pdgid')) if truth == 'gen' else v['q']
            ct = cell(g('pt'), g('eta'), g('phi'), qt, g('vx'), g('vy'), g('vz'))
            c = cell(v['hpt'], v['heta'], v['hphi'], v['hq'], v['hx0'], v['hy0'], v['hz0'])
            m = (ct >= 0) | (c >= 0)
            res.append((m, c != ct))
            frompv.append(v['from_pv'])
        m = np.concatenate([r[0] for r in res]); diff = np.concatenate([r[1] for r in res])
        dd = np.concatenate([disp, disp]); fp = np.concatenate(frompv)
        f = lambda k: 100 * diff[m & k].mean()
        print('  %-14s %25.3f%% %25.3f%% %9.1f%%' % (o, f(~dd), f(dd), 100 * fp.mean()))
        if o == 'gen' and diff[m].any():
            fail.append('gen origin: the MC cell differs from the true one')
        if o == 'pv_or_vertex' and not np.array_equal(fp, np.concatenate([~disp, ~disp])):
            fail.append('pv_or_vertex: the PV start does not follow lxy_sig < %g' % CUT)
        if o in ('dimuon_vertex', 'pv_or_vertex', 'track') and f(~dd) > 2.0:
            fail.append('%s %s: %.2f%% of the prompt-like probes in another cell'
                        % (sample, o, f(~dd)))

print('\nkill maps:')
for st in stems:
    meta = json.load(open(st + '.json'))
    print('  %-50s data %-14s MC %-14s cut %-5s PV start data %.1f%% MC %.1f%%  selection %s'
          % (os.path.relpath(st), H.meta_origin(meta, 'data'), H.meta_origin(meta, 'mc'),
             meta.get('origin_sig_cut'), 100 * meta['start_from_pv']['data'],
             100 * meta['start_from_pv']['mc'], meta.get('selection')))
if fail:
    print('\nSELFTEST FAILED:\n  ' + '\n  '.join(fail))
    sys.exit(1)
print('\ncheck_origins: ok')
