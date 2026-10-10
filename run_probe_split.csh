#!/bin/tcsh
# ---------------------------------------------------------------------------
# run_probe_split.csh -- do prompt and non-prompt J/psi see the same pixel
# hit efficiency? (HITEMU.md 3.7)
#
# For each epoch, two kill-map builds of the DATA, same run ranges:
#   non-prompt : the covflow selection of configs/run3_epochs.py, as it is
#                (decay length L_xy cos(alpha) m / pT > 80 um)
#   prompt     : the same selection without that cut, and the decay length
#                BELOW the cut instead (--drop-cut lxy --data-extra ...)
# then compare_killmaps.py, cell by cell. The MC is the same in both (non-prompt
# Hb MC; only the data maps are compared).
#
# Usage (tcsh, UI):  ./run_probe_split.csh [epoch ...]      default: 2024 2026
# Outputs: probe_split/<epoch>/{nonprompt,prompt,compare}/, logs in probe_split/logs
# ---------------------------------------------------------------------------

set REPO   = /work/manzoni/correct_track_covariance/covflow
set CONDA  = /work/manzoni/miniconda3
set ENV    = covflow
set EPOCHS = (2024 2026)
set OPTS   = (--helix-origin dimuon_vertex)
# the decay-length variable of the covflow selection, and its cut
set DL     = '(lxy * cos2d / pt * 3.0969)'
set CUT    = 0.008

if ( $#argv > 0 ) set EPOCHS = ($argv)
cd $REPO
foreach f (build_kill_maps.py compare_killmaps.py hitemu.py pixel_eff_maps.py configs/run3_epochs.py)
    if ( ! -f $f ) then
        echo "ERROR: $REPO/$f not found"
        exit 1
    endif
end
if ( ! $?CONDA_DEFAULT_ENV ) setenv CONDA_DEFAULT_ENV none
if ( "$CONDA_DEFAULT_ENV" != "$ENV" ) then
    source $CONDA/etc/profile.d/conda.csh
    conda activate $ENV
endif
mkdir -p probe_split/logs

foreach ep ($EPOCHS)
    set O = probe_split/$ep
    echo "=== ${ep}: non-prompt build"
    python build_kill_maps.py --epoch $ep $OPTS --out $O/nonprompt |& tee probe_split/logs/${ep}_nonprompt.log
    if ( ! -f $O/nonprompt/killmaps_$ep.json ) then
        echo "ERROR: $ep non-prompt build failed, see probe_split/logs/${ep}_nonprompt.log"
        continue
    endif
    echo "=== ${ep}: prompt build (same run ranges)"
    python build_kill_maps.py --epoch $ep $OPTS --drop-cut lxy --data-extra "$DL < $CUT" \
        --ranges-from $O/nonprompt/killmaps_$ep --out $O/prompt |& tee probe_split/logs/${ep}_prompt.log
    if ( ! -f $O/prompt/killmaps_$ep.json ) then
        echo "ERROR: $ep prompt build failed, see probe_split/logs/${ep}_prompt.log"
        continue
    endif
    echo "=== ${ep}: comparison"
    python compare_killmaps.py $O/nonprompt/killmaps_$ep $O/prompt/killmaps_$ep \
        --labels non-prompt prompt --out $O/compare |& tee probe_split/logs/${ep}_compare.log
end
echo "done: probe_split/<epoch>/compare/compare_<epoch>.pdf"
