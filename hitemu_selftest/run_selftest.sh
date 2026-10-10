#!/bin/bash
# Synthetic end-to-end test of the per-layer hit-mask emulation (HITEMU.md 9.9).
# Usage (from the covflow repository): bash hitemu_selftest/run_selftest.sh [outdir] [nevents]
set -e
OUT=${1:-hitemu_selftest_out}; N=${2:-150000}
mkdir -p $OUT/fake/configs
cat > $OUT/fake/configs/run3_epochs.py <<CFG
INPUT_DIR = '$(cd $OUT && pwd)/fake'
SE_HOST = 'dummy:1094'
EPOCHS = {'2026': dict(mc=['hb_Summer24.root'], pu_year='2026',
                       data=['data2026B.root', 'data2026D.root'])}
TRAINING = dict(tree='tree', selection='(mu1_pt > 3.5) & (abs(mass-3.0969)<0.1)',
                mc_selection='(abs(mu1_gen_pdgid)==13) & (abs(mu2_gen_pdgid)==13)',
                mc_weight='pu_weight_{pu_year}')
CFG
python3 hitemu_selftest/gen_fake_masks.py $OUT/fake $N
CFGF=$OUT/fake/configs/run3_epochs.py
OLD="--branch mask= --branch count="
python3 build_kill_maps.py  --epoch 2026 --config $CFGF --out $OUT/km_masks
python3 emulate_hit_loss.py --epoch 2026 --config $CFGF --killmaps $OUT/km_masks/killmaps_2026 --closure-only --out $OUT/emu_masks
python3 build_kill_maps.py  --epoch 2026 --config $CFGF $OLD --out $OUT/km_old
python3 emulate_hit_loss.py --epoch 2026 --config $CFGF $OLD --killmaps $OUT/km_old/killmaps_2026 --closure-only --out $OUT/emu_old
echo "compare $OUT/emu_masks/emu_2026.pdf (per-layer masks) with $OUT/emu_old/emu_2026.pdf (L1/D1 only)"
