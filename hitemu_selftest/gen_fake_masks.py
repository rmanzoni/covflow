"""Fake dimuon ntuples WITH the per-layer hit masks of the Bmmm patch
(pix_valid_mask, pix_miss_mask, pix_inact_mask, pix_hit_count).

Hit model as gen_fake.py (C^-1 = prior + sum over valid hits H^T V^-1 H), plus
dead / inefficient regions on L2, L3 and D2 -- the 2026 situation the L1/D1-only
emulation could not reproduce -- and module overlaps (two hits on a layer)."""
import sys, numpy as np, uproot
sys.path.insert(0, __import__('os').path.dirname(__import__('os').path.dirname(__import__('os').path.abspath(__file__))))
import hitemu as H, features as F

BIT = H.MASK_BIT

def vtrue(name, eta, scale):
    a = np.abs(eta)
    if name[0] == 'L':
        su, sv = 10e-4 * (1 + 0.2 * a), 20e-4 * (1 + 0.5 * a)
    else:
        su, sv = 12e-4 * np.ones_like(a), 20e-4 * np.ones_like(a)
    return (su * scale) ** 2, (sv * scale) ** 2

def box(x, ph, x0, x1, p0, p1):
    return (x > x0) & (x < x1) & (ph > p0) & (ph < p1)

def efficiency(name, x, ph, run, t, is_data):
    """(eps, known_dead): known_dead = the conditions flag the module inactive."""
    n = len(x)
    eff = np.full(n, 0.99)
    known = np.zeros(n, bool)
    if is_data:
        if name == 'L1':
            eff = 0.97 - 0.10 * t
            d1 = box(x, ph, 0, 3.33, 0.5, 1.0)
            d2 = (run > 380100) & box(x, ph, -6.66, -3.33, -1.5, -1.0)
            eff = np.where(d1 | d2, 0.0, eff); known = d1
        elif name == 'L2':
            d = (run > 380050) & box(x, ph, 2, 6, -0.5, 0.0)
            d |= box(x, ph, -12, -6, 2.6, 3.1)
            eff = np.where(d, 0.0, 0.985 - 0.04 * t); known = d
        elif name == 'L3':
            d = box(x, ph, -10, -5, 2.0, 2.6)                 # not in the conditions
            eff = np.where(d, 0.05, 0.985)
        elif name == 'D1+':
            eff = np.where(run > 380150, 0.93, 0.97) * np.ones(n)
            eff = np.where(box(x, ph, 6, 9, -2, -1.5), 0.0, eff)
        elif name == 'D1-':
            eff = np.where(run > 380150, 0.93, 0.97) * np.ones(n)
        elif name == 'D2+':
            d = box(x, ph, 6, 9, 1.0, 1.6)
            eff = np.where(d, 0.0, 0.98); known = d
    else:
        if name == 'L1':
            d = box(x, ph, -13.3, -6.66, -2.9, -2.6)
            eff = np.where(d, 0.0, eff); known = d
        elif name == 'L2':
            d = box(x, ph, -5, -2, 1.0, 1.4)                  # MC-only dead module
            eff = np.where(d, 0.0, eff); known = d
    return eff, known

def sample(n, is_data, seed):
    rng = np.random.default_rng(seed)
    d = {}
    run = rng.integers(380000, 380200, n) if is_data else np.ones(n, np.int64)
    pv_z = rng.normal(0.0 if is_data else 0.4, 3.8 if is_data else 3.6, n)
    d['pv_x'] = np.full(n, 0.01); d['pv_y'] = np.full(n, -0.02); d['pv_z'] = pv_z
    # dimuon vertex: both muons start there. Data: 70% prompt (vertex at the PV,
    # within resolution), 30% non-prompt; MC: non-prompt only (as the Hb MC)
    prompt = (rng.random(n) < 0.7) if is_data else np.zeros(n, bool)
    lxy = np.where(prompt, np.abs(rng.normal(0, 0.003, n)), rng.exponential(0.05, n))
    fv = rng.uniform(-np.pi, np.pi, n)
    vx = d['pv_x'] + lxy * np.cos(fv); vy = d['pv_y'] + lxy * np.sin(fv)
    vz = pv_z + np.where(prompt, rng.normal(0, 0.003, n), rng.normal(0, 0.02, n))
    d['vx'], d['vy'], d['vz'] = vx, vy, vz
    d['lxy'], d['cos2d'], d['pt'] = lxy, np.ones(n), np.full(n, 3.0969 / 0.008 * 0.05)
    d['run'] = run
    d['mass'] = 3.0969 + 0.03 * rng.normal(size=n)
    t = (run - 380000) / 200.0
    for mu in ('mu1', 'mu2'):
        pt = 3.5 + rng.exponential(4, n); eta = rng.uniform(-2.4, 2.4, n)
        phi = rng.uniform(-np.pi, np.pi, n); q = rng.choice([-1., 1.], n)
        z0 = vz
        par = H.helix_to_curv(pt, eta, phi, q, z0)
        info = np.zeros((n, 5, 5))
        qop = par[:, 0]
        for i, s in enumerate([0.01 * np.abs(qop), 2e-3, 2e-3, 0.05, 0.1]):
            info[:, i, i] = 1.0 / np.broadcast_to(s, (n,)) ** 2
        vmask = np.zeros(n, np.int64); mmask = np.zeros(n, np.int64); imask = np.zeros(n, np.int64)
        count = np.zeros(n, np.int64)
        outlier = np.zeros(n, bool)
        for name in H.ALL_SURFACES:
            x, ph = H.cross(name, pt, eta, phi, q, vx, vy, z0)
            acc = H.in_acceptance(name, x)
            if name[0] == 'D':
                acc &= (eta >= 0) if name.endswith('+') else (eta < 0)
            eff, known = efficiency(name, x, ph, run, t, is_data)
            valid = acc & (rng.random(n) < eff)
            if name == 'L1' and is_data:          # rejected hits: track worse, not a dead module
                out = valid & (rng.random(n) < 0.03)
                valid &= ~out; outlier |= out
            ncnt = np.where(valid, 1 + (rng.random(n) < 0.08), 0)   # 8% module overlaps
            if valid.any():
                Hk = H.jacobian(name, par[valid])
                vu, vv = vtrue(name, eta[valid], 1.1 if is_data else 1.0)
                Vi = np.zeros((valid.sum(), 2, 2)); Vi[:, 0, 0] = 1 / vu; Vi[:, 1, 1] = 1 / vv
                info[valid] += (np.swapaxes(Hk, 1, 2) @ Vi @ Hk) * ncnt[valid][:, None, None]
            bit = BIT[name[:2]]
            lost = acc & ~valid
            vmask |= np.where(valid, 1 << bit, 0)
            imask |= np.where(lost & known, 1 << bit, 0)
            mmask |= np.where(lost & ~known, 1 << bit, 0)
            count |= ncnt << (2 * bit)
        C = np.linalg.inv(info)
        C[outlier] *= 1.96
        packed = F.matrix_to_packed(C)
        for j, nm in enumerate(F.PACK_NAMES):
            d['%s_cov_%s' % (mu, nm)] = packed[:, j]
        nb = sum((count >> (2 * BIT['L%d' % k])) & 3 for k in range(1, 5))
        ne = sum((count >> (2 * BIT['D%d' % k])) & 3 for k in range(1, 4))
        fb, fe, nlay, code = H.first_from_mask(vmask)
        sdxy = np.sqrt(C[:, 3, 3])
        d[mu + '_bs_dxy'] = rng.choice([-1, 1], n) * rng.exponential(0.01, n) + sdxy * rng.normal(size=n)
        d[mu + '_bs_dxy_e'] = np.sqrt(sdxy ** 2 + (5e-4) ** 2)
        d[mu + '_best_trk_pt'] = pt; d[mu + '_pt'] = pt
        d[mu + '_best_trk_eta'] = eta; d[mu + '_eta'] = eta
        d[mu + '_phi'] = phi; d[mu + '_charge'] = q
        # impact parameters w.r.t. the PV, CMSSW straight-line convention
        ddx, ddy = vx - d['pv_x'], vy - d['pv_y']
        d[mu + '_dxy'] = -ddx * np.sin(phi) + ddy * np.cos(phi)
        d[mu + '_dz'] = (z0 - pv_z) - (ddx * np.cos(phi) + ddy * np.sin(phi)) * np.sinh(eta)
        d[mu + '_n_pix_b_hit'] = nb; d[mu + '_n_pix_e_hit'] = ne; d[mu + '_n_pix_hit'] = nb + ne
        d[mu + '_pix_first_b_layer'] = fb; d[mu + '_pix_first_e_disk'] = fe
        d[mu + '_n_pix_layer'] = nlay; d[mu + '_pix_first_layer'] = code
        d[mu + '_pix_valid_mask'] = vmask; d[mu + '_pix_miss_mask'] = mmask
        d[mu + '_pix_inact_mask'] = imask; d[mu + '_pix_hit_count'] = count
        d[mu + '_id_medium'] = np.ones(n)
        if not is_data:
            d[mu + '_gen_pdgid'] = np.full(n, 13.)
    if not is_data:
        w = rng.uniform(0.5, 1.5, n); w[:20] = np.nan
        d['pu_weight_2026'] = w
    return d

def write(path, d):
    with uproot.recreate(path) as f:
        f.mktree('tree', {k: (np.int64 if k == 'run' else np.float32) for k in d})
        f['tree'].extend({k: (v.astype(np.int64) if k == 'run' else v.astype(np.float32)) for k, v in d.items()})

if __name__ == '__main__':
    out = sys.argv[1]; n = int(sys.argv[2])
    write(out + '/data2026B.root', sample(n, True, 11))
    write(out + '/data2026D.root', sample(n // 2, True, 12))
    write(out + '/hb_Summer24.root', sample(n, False, 13))
    print('ok')
