#!/usr/bin/env python3
"""
batch/submit_run3_epochs.py
===========================

One covflow GPU training per (data-taking epoch, muon), as defined in
configs/run3_epochs.py.

Run it on the UI (t3ui07), where /pnfs is mounted, with the covflow conda env
active (it needs numpy and uproot) and a grid proxy on a shared filesystem:

    voms-proxy-init --voms cms --valid 192:00 --out $HOME/.x509up_u$(id -u)
    export X509_USER_PROXY=$HOME/.x509up_u$(id -u)

    python batch/submit_run3_epochs.py --dry-run          # checks + configs only
    python batch/submit_run3_epochs.py                    # submit everything
    python batch/submit_run3_epochs.py --only 2024 2025 --muons mu2

What it does, in order, stopping at the first problem:

  1. coverage   every data*.root in INPUT_DIR belongs to exactly one epoch,
                every hb_*.root is used, every other file is declared
                VALIDATION_ONLY, and every listed file exists;
  2. branches   every file has every branch the jobs will read (covariance,
                context, selections, pileup weight) -- checked on metadata
                here rather than after a day in the gpu queue;
  3. weights    the pileup-weight branch of each MC file is not all-NaN for the
                year it serves, and the NaN fraction is reported (those rows
                are dropped by the loader, together with bad covariances);
  4. proxy      valid for at least SLURM['proxy_min_valid'];
  5. render     OUT_ROOT/TAG/<epoch>/<mu>/job.conf, a self-contained bash
                config for batch/train_gpu.sh (inputs staged by xrdcp);
  6. submit     batch/submit_gpu.sh <job.conf>, one Slurm array per job,
                recorded in OUT_ROOT/TAG/submitted.json.

--dry-run does 1-5 (a proxy problem is then only reported).
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import re
import runpy
import shlex
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
DEFAULT_CFG = os.path.join(REPO_ROOT, 'configs', 'run3_epochs.py')
SUBMIT = os.path.join(HERE, 'submit_gpu.sh')


def die(msg):
    sys.stderr.write('\nERROR: ' + msg.rstrip() + '\n')
    sys.exit(1)


def warn(msg):
    sys.stderr.write('[warn] ' + msg.rstrip() + '\n')


def progress(i, n, label, t0=None, width=30, pre=False):
    """One status line per step; redrawn in place on a terminal.

    pre=True marks the call made BEFORE a slow step (which file is being
    opened now); it is shown on a terminal only, so a log file gets one line
    per completed step.
    """
    if pre and not sys.stderr.isatty():
        return
    frac = i / n if n else 1.0
    done = int(width * frac)
    eta = ''
    if t0 is not None and 0 < i < n:
        el = time.time() - t0
        eta = '  eta %ds' % (el / i * (n - i))
    line = '[%s%s] %d/%d %-48s%s' % ('#' * done, '.' * (width - done), i, n,
                                      label[:48], eta)
    if sys.stderr.isatty():
        sys.stderr.write('\r' + line + ('\n' if i == n else ''))
    else:
        sys.stderr.write(line + '\n')
    sys.stderr.flush()


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--cfg', default=DEFAULT_CFG)
    p.add_argument('--only', nargs='+', default=None, metavar='EPOCH',
                   help='submit only these epochs (default: all)')
    p.add_argument('--muons', nargs='+', default=None,
                   help='default: MUONS from the config')
    p.add_argument('--tag', default=None, help='override TAG from the config')
    p.add_argument('--dry-run', action='store_true',
                   help='run the checks and write the configs, submit nothing')
    p.add_argument('--force', action='store_true',
                   help='submit again a job already submitted under this TAG '
                        '(its old task directories are NOT cleaned)')
    p.add_argument('--skip-branch-check', action='store_true',
                   help='skip steps 2 and 3 (they open every input file)')
    return p.parse_args()


# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------

def load_cfg(path):
    if not os.path.isfile(path):
        die('config not found: %s' % path)
    cfg = runpy.run_path(path)
    for key in ('INPUT_DIR', 'SE_HOST', 'OUT_ROOT', 'TAG', 'MUONS', 'EPOCHS',
                'VALIDATION_ONLY', 'TRAINING', 'SLURM', 'CONDA_ENV'):
        if key not in cfg:
            die('%s does not define %s' % (path, key))
    for name, ep in cfg['EPOCHS'].items():
        for key in ('mc', 'data', 'pu_year'):
            if not ep.get(key):
                die('epoch %s: %r is missing or empty' % (name, key))
    return cfg


def fill(template, **kw):
    """str.format on a template; fails loud on an unknown placeholder."""
    try:
        return template.format(**kw)
    except (KeyError, IndexError) as e:
        die('placeholder %s in %r is not one of %s' % (e, template, sorted(kw)))


def job_settings(cfg, epoch, mu):
    """Everything one job needs, placeholders resolved."""
    T = cfg['TRAINING']
    ep = cfg['EPOCHS'][epoch]
    kw = dict(mu=mu, pu_year=ep['pu_year'])
    s = dict(
        epoch=epoch, mu=mu, pu_year=ep['pu_year'],
        data=[os.path.join(cfg['INPUT_DIR'], f) for f in ep['data']],
        mc=[os.path.join(cfg['INPUT_DIR'], f) for f in ep['mc']],
        tree=T['tree'],
        cov_prefix=fill(T['cov_prefix'], **kw),
        context=[fill(c, **kw) for c in T['context']],
        log_pt=fill(T['log_pt'], **kw) if T.get('log_pt') else '',
        context_edges={fill(k, **kw): list(v)
                       for k, v in T.get('context_edges', {}).items()},
        data_weight=fill(T.get('data_weight', ''), **kw),
        mc_weight=fill(T.get('mc_weight', ''), **kw),
        selection=T.get('selection') or '',
        data_selection=T.get('data_selection') or '',
        mc_selection=T.get('mc_selection') or '',
    )
    for k in s['context_edges']:
        if k not in s['context']:
            die('context_edges names %s, which is not in the context %s'
                % (k, s['context']))
    return s


# ---------------------------------------------------------------------------
# 1. coverage
# ---------------------------------------------------------------------------

def check_coverage(cfg):
    d = cfg['INPUT_DIR']
    if not os.path.isdir(d):
        die('INPUT_DIR %s is not visible. Run this on the UI, where /pnfs is '
            'mounted.' % d)
    present = sorted(f for f in os.listdir(d) if f.endswith('.root'))

    owner = {}
    for name, ep in cfg['EPOCHS'].items():
        for f in ep['data']:
            owner.setdefault(f, []).append(name)
    twice = {f: e for f, e in owner.items() if len(e) > 1}
    if twice:
        die('data file(s) assigned to more than one epoch:\n' +
            '\n'.join('  %-45s %s' % (f, ', '.join(e)) for f, e in sorted(twice.items())))

    used_mc = {f for ep in cfg['EPOCHS'].values() for f in ep['mc']}
    listed = set(owner) | used_mc | set(cfg['VALIDATION_ONLY'])

    missing = sorted(listed - set(present))
    if missing:
        die('listed in the config but not in %s:\n  ' % d + '\n  '.join(missing))

    unassigned = [f for f in present if f not in listed]
    if unassigned:
        die('in %s but in no epoch and not VALIDATION_ONLY -- assign or declare '
            'them:\n  ' % d + '\n  '.join(unassigned))

    unused_hb = sorted(f for f in present if f.startswith('hb_') and f not in used_mc)
    if unused_hb:
        die('Hb MC file(s) not used by any epoch:\n  ' + '\n  '.join(unused_hb))

    gb = lambda fs: sum(os.path.getsize(os.path.join(d, f)) for f in fs) / 1e9
    print('\n[1] coverage: %d files in %s' % (len(present), d))
    print('    %-14s %-6s %5s %8s   %s' % ('epoch', 'PU', 'files', 'data GB', 'MC'))
    for name, ep in cfg['EPOCHS'].items():
        print('    %-14s %-6s %5d %8.2f   %s (%.2f GB)'
              % (name, ep['pu_year'], len(ep['data']), gb(ep['data']),
                 ', '.join(ep['mc']), gb(ep['mc'])))
    print('    validation only: %s' % ', '.join(cfg['VALIDATION_ONLY']))


# ---------------------------------------------------------------------------
# 2-3. branches and pileup weights
# ---------------------------------------------------------------------------

def check_branches(cfg, jobs):
    try:
        import numpy as np
        import uproot
    except ImportError as e:
        die('%s -- activate the covflow conda env, or use --skip-branch-check' % e)
    sys.path.insert(0, os.path.dirname(REPO_ROOT))
    if os.path.basename(REPO_ROOT) != 'covflow':
        die('the repository directory must be called "covflow" for the package '
            'import (it is %s)' % REPO_ROOT)
    from covflow import data as D
    from covflow import features as F

    # what each file must provide, summed over the jobs that read it
    need, pu_years = {}, {}
    for s in jobs:
        common = set(s['context']) | {s['cov_prefix'] + n for n in F.PACK_NAMES}
        if s['selection']:
            common |= set(D.selection_branches(s['selection']))
        for f in s['data']:
            b = set(common)
            if s['data_selection']:
                b |= set(D.selection_branches(s['data_selection']))
            if s['data_weight']:
                b.add(s['data_weight'])
            need.setdefault(f, set()).update(b)
        for f in s['mc']:
            b = set(common)
            if s['mc_selection']:
                b |= set(D.selection_branches(s['mc_selection']))
            if s['mc_weight']:
                b.add(s['mc_weight'])
                pu_years.setdefault(f, set()).add(s['mc_weight'])
            need.setdefault(f, set()).update(b)

    files = sorted(need)
    print('\n[2] branches: %d files' % len(files))
    problems, t0 = [], time.time()
    for i, f in enumerate(files, 1):
        progress(i - 1, len(files), os.path.basename(f), t0, pre=True)
        try:
            with uproot.open(f) as fh:
                if cfg['TRAINING']['tree'] not in fh:
                    problems.append('%s: no tree %r' % (f, cfg['TRAINING']['tree']))
                    progress(i, len(files), os.path.basename(f), t0)
                    continue
                have = set(fh[cfg['TRAINING']['tree']].keys())
        except Exception as e:
            problems.append('%s: cannot open (%s)' % (f, e))
            progress(i, len(files), os.path.basename(f), t0)
            continue
        miss = sorted(need[f] - have)
        if miss:
            problems.append('%s: %d missing, e.g. %s'
                            % (os.path.basename(f), len(miss), ', '.join(miss[:6])))
        progress(i, len(files), os.path.basename(f), t0)
    if problems:
        die('branch check failed:\n  ' + '\n  '.join(problems))
    print('    all branches present')

    print('\n[3] pileup weights (rows with a non-finite weight are dropped by '
          'the loader)')
    t0 = time.time()
    mc_files = sorted(pu_years)
    for i, f in enumerate(mc_files, 1):
        progress(i - 1, len(mc_files), os.path.basename(f), t0, pre=True)
        with uproot.open(f) as fh:
            arr = fh[cfg['TRAINING']['tree']].arrays(sorted(pu_years[f]), library='np')
        progress(i, len(mc_files), os.path.basename(f))
        for br in sorted(pu_years[f]):
            w = arr[br]
            fin = np.isfinite(w)
            frac_bad = 1.0 - fin.mean() if len(w) else 1.0
            print('    %-24s %-18s %10d rows  non-finite %6.3f%%  mean %.4f  max %.2f'
                  % (os.path.basename(f), br, len(w), 100 * frac_bad,
                     w[fin].mean() if fin.any() else float('nan'),
                     w[fin].max() if fin.any() else float('nan')))
            if not fin.any():
                die('%s: %s is NaN everywhere -- wrong year for this campaign, '
                    'or MC produced without --pu' % (f, br))
            if frac_bad > 0.01:
                warn('%s: %.2f%% of %s is non-finite; those rows will be dropped'
                     % (os.path.basename(f), 100 * frac_bad, br))


# ---------------------------------------------------------------------------
# 4. proxy, repo
# ---------------------------------------------------------------------------

def check_proxy(cfg, fatal):
    report = die if fatal else warn
    proxy = os.environ.get('X509_USER_PROXY', '')
    if not proxy or not os.path.isfile(proxy):
        return report('X509_USER_PROXY is unset or missing. Create it with\n'
                      '  voms-proxy-init --voms cms --valid 192:00 --out $HOME/.x509up_u$(id -u)\n'
                      '  export X509_USER_PROXY=$HOME/.x509up_u$(id -u)')
    if proxy.startswith('/tmp/'):
        return report('X509_USER_PROXY=%s is node-local; the worker nodes cannot '
                      'read it' % proxy)
    valid = cfg['SLURM'].get('proxy_min_valid', '48:00')
    if shutil.which('voms-proxy-info') is None:
        return report('voms-proxy-info not found')
    if subprocess.call(['voms-proxy-info', '-exists', '-valid', valid],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) != 0:
        return report('grid proxy valid for less than %s: renew it '
                      '(voms-proxy-init --voms cms --valid 192:00 --out %s)'
                      % (valid, proxy))
    print('\n[4] proxy: %s, valid for at least %s' % (proxy, valid))


def check_repo(repo):
    tc = os.path.join(repo, 'train_covflow.py')
    if not os.path.isfile(tc):
        die('REPO %s has no train_covflow.py' % repo)
    if '--context-edges' not in open(tc).read():
        die('REPO %s predates --context-edges; the jobs would fail on it' % repo)
    head = dirty = ''
    try:
        head = subprocess.run(['git', '-C', repo, 'rev-parse', '--short', 'HEAD'],
                              capture_output=True, text=True).stdout.strip()
        dirty = subprocess.run(['git', '-C', repo, 'status', '--porcelain',
                                '--untracked-files=no'],
                               capture_output=True, text=True).stdout.strip()
    except OSError:
        pass
    if dirty:
        warn('REPO %s has uncommitted changes; the runs record only HEAD (%s)'
             % (repo, head or '?'))
    return head or 'unknown'


# ---------------------------------------------------------------------------
# 5. render
# ---------------------------------------------------------------------------

def q(v):
    return shlex.quote(str(v))


def qa(vals):
    return '( ' + ' '.join(q(v) for v in vals) + ' )'


def _arg_lines(args):
    """One line per option, with its values: '  --closure-bins 4,4,3,4,3'."""
    lines = []
    for x in args:
        if str(x).startswith('--') or not lines:
            lines.append('  ' + q(x))
        else:
            lines[-1] += ' ' + q(x)
    return lines


def render(cfg, s, repo, head, out_base, cfg_path):
    T, S = cfg['TRAINING'], cfg['SLURM']
    nbytes = sum(os.path.getsize(f) for f in s['data'] + s['mc'])
    extra = list(T.get('extra_args', []))
    if s['context_edges']:
        extra += ['--context-edges'] + ['%s=%s' % (k, ','.join('%g' % x for x in v))
                                        for k, v in s['context_edges'].items()]
    if s['selection']:
        extra += ['--selection', s['selection']]
    if s['data_selection']:
        extra += ['--data-selection', s['data_selection']]
    if s['mc_selection']:
        extra += ['--mc-selection', s['mc_selection']]
    # train_gpu.sh owns these; a duplicate in EXTRA_ARGS would silently win
    owned = {'--data', '--mc', '--tree', '--cov-prefix', '--context', '--log-pt',
             '--epochs', '--batch-size', '--lr', '--transforms', '--hidden',
             '--bins', '--seed', '--device', '--out'}
    clash = sorted(owned & set(extra))
    if clash:
        die('extra_args repeats option(s) train_gpu.sh already sets: %s' % clash)

    lines = [
        '# ============================================================',
        '# GENERATED by batch/submit_run3_epochs.py -- edit %s instead.' % cfg_path,
        '# epoch %s, %s, MC pileup year %s' % (s['epoch'], s['mu'], s['pu_year']),
        '# covflow HEAD at rendering: %s' % head,
        '# rendered %s' % datetime.datetime.now().isoformat(timespec='seconds'),
        '# ============================================================',
        '',
        'COVFLOW_REPO=%s' % q(repo),
        'COVFLOW_CONDA_ENV=%s' % q(cfg['CONDA_ENV']),
        '',
        '# inputs, copied to local scratch by train_gpu.sh (no /pnfs on the nodes)',
        'COVFLOW_DATA=%s' % qa(s['data']),
        'COVFLOW_MC=%s' % qa(s['mc']),
        'COVFLOW_STAGE=1',
        'COVFLOW_SE_HOST=%s' % q(cfg['SE_HOST']),
        'COVFLOW_STAGE_BYTES=%d' % nbytes,
        '',
        'COVFLOW_TREE=%s' % q(s['tree']),
        'COVFLOW_COV_PREFIX=%s' % q(s['cov_prefix']),
        'COVFLOW_CONTEXT=%s' % q(' '.join(s['context'])),
        'COVFLOW_LOG_PT=%s' % q(s['log_pt']),
        'COVFLOW_DATA_WEIGHT=%s' % q(s['data_weight']),
        'COVFLOW_MC_WEIGHT=%s' % q(s['mc_weight']),
        '',
        'COVFLOW_EPOCHS=%s' % q(T['epochs']),
        'COVFLOW_BATCH_SIZE=%s' % q(T['batch_size']),
        'COVFLOW_LR=%s' % q(T['lr']),
        'COVFLOW_TRANSFORMS=%s' % q(T['transforms']),
        'COVFLOW_HIDDEN=%s' % q(T['hidden']),
        'COVFLOW_BINS=%s' % q(T['bins']),
        'COVFLOW_SEED=%s' % q(T['seeds'][0]),
        'COVFLOW_SEEDS=%s' % qa(T['seeds']),
        '',
        'COVFLOW_OUT_BASE=%s' % q(out_base),
        '',
        'EXTRA_ARGS=(',
    ] + _arg_lines(extra) + [
        ')',
        '',
        'COVFLOW_TIME=%s' % q(S['time']),
        'COVFLOW_MEM=%s' % q(S['mem']),
        'COVFLOW_MIN_MEM=%s' % q(S.get('min_mem', 16384)),
        'COVFLOW_PROXY_MIN_VALID=%s' % q(S.get('proxy_min_valid', '48:00')),
        'COVFLOW_SBATCH_ARGS=%s' % q('--job-name=cf_%s_%s' % (s['epoch'], s['mu'])),
        '',
    ]
    return '\n'.join(lines), nbytes


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    a = parse_args()
    cfg_path = os.path.abspath(a.cfg)
    cfg = load_cfg(cfg_path)
    tag = a.tag or cfg['TAG']
    muons = a.muons or list(cfg['MUONS'])
    epochs = list(cfg['EPOCHS'])
    if a.only:
        bad = [e for e in a.only if e not in cfg['EPOCHS']]
        if bad:
            die('unknown epoch(s) %s; known: %s' % (bad, epochs))
        epochs = [e for e in epochs if e in a.only]
    for mu in muons:
        if not re.fullmatch(r'mu\d', mu):
            die('muon %r is not of the form muN' % mu)

    repo = os.path.abspath(cfg.get('REPO') or REPO_ROOT)
    head = check_repo(repo)
    run_root = os.path.join(cfg['OUT_ROOT'], tag)

    print('covflow Run 3 per-epoch trainings')
    print('  config : %s' % cfg_path)
    print('  repo   : %s (HEAD %s)' % (repo, head))
    print('  output : %s/<epoch>/<mu>' % run_root)
    print('  jobs   : %d epochs x %d muons = %d'
          % (len(epochs), len(muons), len(epochs) * len(muons)))

    check_coverage(cfg)
    jobs = [job_settings(cfg, e, mu) for e in epochs for mu in muons]
    if a.skip_branch_check:
        warn('branch and pileup-weight checks skipped')
    else:
        check_branches(cfg, jobs)
    check_proxy(cfg, fatal=not a.dry_run)

    # Refuse to touch a job that was already SUBMITTED, before writing
    # anything. A job.conf that was only rendered (--dry-run) is simply
    # rendered again: it comes from the same config file.
    record_path = os.path.join(run_root, 'submitted.json')
    record = json.load(open(record_path)) if os.path.isfile(record_path) else []
    done = {(r['epoch'], r['mu']) for r in record}
    confs = {}
    for s in jobs:
        out_base = os.path.join(run_root, s['epoch'], s['mu'])
        conf = os.path.join(out_base, 'job.conf')
        submitted = ((s['epoch'], s['mu']) in done or
                     (os.path.isdir(out_base) and
                      any(d.startswith('task_') for d in os.listdir(out_base))))
        if submitted and not a.force:
            die('%s %s was already submitted under TAG %s (%s). Use a new '
                '--tag, or --force to submit it again into the same '
                'directory.' % (s['epoch'], s['mu'], tag, out_base))
        confs[(s['epoch'], s['mu'])] = (out_base, conf)

    print('\n[5] rendering %d configs' % len(jobs))
    os.makedirs(run_root, exist_ok=True)
    shutil.copy2(cfg_path, os.path.join(run_root, os.path.basename(cfg_path)))
    for s in jobs:
        out_base, conf = confs[(s['epoch'], s['mu'])]
        os.makedirs(out_base, exist_ok=True)
        text, nbytes = render(cfg, s, repo, head, out_base, cfg_path)
        with open(conf, 'w') as fh:
            fh.write(text)
        print('    %-14s %-4s %2d data + %d MC files, %5.1f GB staged  %s'
              % (s['epoch'], s['mu'], len(s['data']), len(s['mc']), nbytes / 1e9, conf))

    if a.dry_run:
        print('\n--dry-run: nothing submitted. Inspect one config, e.g.\n  less %s'
              % next(iter(confs.values()))[1])
        return

    print('\n[6] submitting')
    t0 = time.time()
    for i, s in enumerate(jobs, 1):
        out_base, conf = confs[(s['epoch'], s['mu'])]
        progress(i - 1, len(jobs), '%s %s' % (s['epoch'], s['mu']), t0, pre=True)
        r = subprocess.run([SUBMIT, conf], capture_output=True, text=True)
        log = os.path.join(out_base, 'submit.log')
        with open(log, 'w') as fh:
            fh.write(r.stdout + r.stderr)
        m = re.search(r'Submitted Slurm array: (\S+)', r.stdout)
        if r.returncode != 0 or not m:
            die('submission of %s %s failed (exit %d); see %s\n%s'
                % (s['epoch'], s['mu'], r.returncode, log, (r.stdout + r.stderr)[-2000:]))
        record.append(dict(epoch=s['epoch'], mu=s['mu'], job=m.group(1), conf=conf,
                           head=head, submitted=datetime.datetime.now()
                           .isoformat(timespec='seconds')))
        with open(record_path, 'w') as fh:
            json.dump(record, fh, indent=2)
        progress(i, len(jobs), '%s %s -> %s' % (s['epoch'], s['mu'], m.group(1)), t0)

    print('\n    %-14s %-4s %s' % ('epoch', 'muon', 'Slurm array'))
    for rec in record[-len(jobs):]:
        print('    %-14s %-4s %s' % (rec['epoch'], rec['mu'], rec['job']))
    ids = ','.join(rec['job'] for rec in record[-len(jobs):])
    print('\nMonitor:\n  squeue -j %s' % ids)
    print('Accounting:\n  sacct -j %s --format=JobName%%24,State,Elapsed,MaxRSS,NodeList' % ids)
    print('Record: %s' % record_path)


if __name__ == '__main__':
    main()
