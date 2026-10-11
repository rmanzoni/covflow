#!/bin/bash
# Synthetic end-to-end test of the per-layer hit-mask emulation (HITEMU.md 9.9)
# and of the helix start points per sample (HITEMU.md 3.8).
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
# --- helix origins per sample (HITEMU 3.6-3.8): a second config with the
# decay-length cut of the covflow selection, dropped by --drop-cut lxy
sed 's/selection=.(mu1_pt > 3.5)/selection='"'"'(lxy * cos2d \/ pt * 3.0969 > 0.008) \& (mu1_pt > 3.5)/' \
    $CFGF > $OUT/fake/configs/run3_epochs_dl.py
CFG2=$OUT/fake/configs/run3_epochs_dl.py
ALLP="--helix-origin-data pv_or_vertex --origin-sig-cut 3 --helix-origin-mc dimuon_vertex"
python3 build_kill_maps.py  --epoch 2026 --config $CFG2 $ALLP --drop-cut lxy --out $OUT/km_allprobes
python3 emulate_hit_loss.py --epoch 2026 --config $CFG2 $ALLP --killmaps $OUT/km_allprobes/killmaps_2026 --closure-only --out $OUT/emu_allprobes
GEN="--helix-origin-data dimuon_vertex --helix-origin-mc gen"
python3 build_kill_maps.py  --epoch 2026 --config $CFG2 $GEN --drop-cut lxy --out $OUT/km_gen
# guards: an emulation with other origins than the maps must stop
for bad in "$ALLP --origin-sig-cut 5" "--helix-origin dimuon_vertex" "--helix-origin gen"; do
    if python3 emulate_hit_loss.py --epoch 2026 --config $CFG2 $bad --killmaps $OUT/km_allprobes/killmaps_2026 \
           --closure-only --max-events 1000 --out $OUT/emu_guard > $OUT/guard.log 2>&1; then
        echo "SELFTEST FAILED: emulate_hit_loss accepted '$bad' with the allprobes maps"; exit 1
    fi
    echo "guard ok ($bad): $(grep -m1 -A1 ERROR $OUT/guard.log | tail -1)"
done
python3 hitemu_selftest/check_origins.py $OUT/fake $OUT/km_masks/killmaps_2026 $OUT/km_allprobes/killmaps_2026 $OUT/km_gen/killmaps_2026
echo "compare $OUT/emu_masks/emu_2026.pdf (per-layer masks) with $OUT/emu_old/emu_2026.pdf (L1/D1 only)"
