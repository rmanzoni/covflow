#!/bin/tcsh
#
# run_hitemu.csh -- the hit-loss emulation chain for every covflow epoch, one
# epoch after the other. Meant to run inside screen on the UI (t3ui07).
#
#   screen -S hitemu
#   tcsh run_hitemu.csh                        # all 7 epochs
#   tcsh run_hitemu.csh 2025 2026              # only these
#   (detach: Ctrl-a d   reattach: screen -r hitemu)
#
# Per epoch, in <REPO>/hitemu/<epoch>/ :
#   1. build_kill_maps.py                -> killmaps/      run ranges, kill maps
#   2. noL1_data_study.py (data)         -> noL1_data/     mixture test, data hit errors
#   3. noL1_data_study.py --sample mc    -> noL1_mc/       MC hit errors
#   4. emulate_hit_loss.py --route B [A] -> emu/           emulation, closure, routes
#      route A (and route B final) only if the epoch's trained flows exist under
#      $RUNS/<epoch>/<mu>/task_0 ; with WRITE_TREE = 1 also the emulated MC tree.
# A step that fails stops that epoch (later steps need its output); the loop
# goes on with the next epoch. A step counts as OK only if it wrote its json.
# If step 4 fails with route B (typically: no hit-error fit for a surface), it
# is rerun without route B (emulate: OK_noB / OK_A_noB).
# Logs: hitemu/logs/<epoch>_<step>.log. Summary table at the end, also in
# hitemu/logs/summary_<timestamp>.txt.

# ---------------------------------------------------------------- settings
set REPO   = /work/manzoni/correct_track_covariance/covflow
set CONDA  = /work/manzoni/miniconda3
set ENV    = covflow
set EPOCHS = (2022_preEE 2022_postEE 2023_preBPix 2023_postBPix 2024 2025 2026)

# trained per-epoch flows: $RUNS/<epoch>/<mu>/task_0/covflow.json
# CHECK the TAG of the production you want to use
# set RUNS   = /work/manzoni/correct_track_covariance/covflow-runs/run3_epochs_05oct26
set RUNS   = /work/manzoni/correct_track_covariance/covflow-runs/run3_epochs_05oct26_trgmatch_e1200

set WRITE_TREE = 0          # 1: also write emu/emu_<epoch>_route<A|B>.root (large)

# route B hit errors: the epoch's own MC fit, then its data fit, then these
# (per surface, the first file with a fit is used). Useful for the early epochs,
# where few dead cells may leave no fit, e.g.
#   set EXTRA_HITERR = (hitemu/2024/noL1_mc/hiterrors_mc_2024.json hitemu/2024/noL1_data/hiterrors_data_2024.json)
set EXTRA_HITERR = ()

# options passed to every script. Add e.g. --max-events 300000 for a quick pass.
# The helix that finds the crossed pixel cells starts at the dimuon (J/psi)
# vertex (vx, vy, vz), the script default since Oct 2026: the probes come from
# displaced J/psi. Kill maps built before that used the PV; the emulation stops
# if the maps and the run disagree. Old behaviour: --helix-origin pv
set OPTS = (--helix-origin dimuon_vertex)

# ------------------------------------------------------------------ checks
if ( $#argv > 0 ) set EPOCHS = ($argv)
if ( ! -d $REPO ) then
    echo "ERROR: repository $REPO not found"
    exit 1
endif
cd $REPO
foreach f (hitemu.py build_kill_maps.py noL1_data_study.py emulate_hit_loss.py pixel_eff_maps.py features.py configs/run3_epochs.py)
    if ( ! -f $f ) then
        echo "ERROR: $REPO/$f not found"
        exit 1
    endif
end
grep -q "set_thetalim" pixel_eff_maps.py
if ( $status != 0 ) then
    echo "ERROR: pixel_eff_maps.py predates the polar-plot fix; copy the latest version"
    exit 1
endif
foreach ep ($EPOCHS)
    grep -q "'$ep'" configs/run3_epochs.py
    if ( $status != 0 ) then
        echo "ERROR: epoch '$ep' is not in configs/run3_epochs.py"
        exit 1
    endif
end

if ( ! $?CONDA_DEFAULT_ENV ) setenv CONDA_DEFAULT_ENV none
if ( "$CONDA_DEFAULT_ENV" != "$ENV" ) then
    if ( ! -f $CONDA/etc/profile.d/conda.csh ) then
        echo "ERROR: $CONDA/etc/profile.d/conda.csh not found; activate $ENV yourself"
        exit 1
    endif
    source $CONDA/etc/profile.d/conda.csh
    conda activate $ENV
endif
python -c "import numpy, uproot, matplotlib" >& /dev/null
if ( $status != 0 ) then
    echo "ERROR: python of env '$ENV' lacks numpy/uproot/matplotlib"
    exit 1
endif

mkdir -p hitemu/logs
set STAMP   = `date +%Y%m%d_%H%M%S`
set SUMMARY = hitemu/logs/summary_$STAMP.txt
set T_ALL   = `date +%s`
printf "%-15s %-10s %-10s %-10s %-14s %8s\n" epoch killmaps noL1_data noL1_mc emulate minutes > $SUMMARY

echo "============================================================"
echo " hit-loss emulation over: $EPOCHS"
echo " repo    : $REPO"
echo " flows   : $RUNS/<epoch>/<mu>/task_0"
echo " python  : `which python`"
echo " options : $OPTS"
echo " started : `date`"
echo "============================================================"

foreach ep ($EPOCHS)
    set T0  = `date +%s`
    set OUT = hitemu/$ep
    set KM  = $OUT/killmaps/killmaps_$ep
    set HD  = $OUT/noL1_data/hiterrors_data_$ep.json
    set HM  = $OUT/noL1_mc/hiterrors_mc_$ep.json
    set R1 = -
    set R2 = -
    set R3 = -
    set R4 = -
    set GO = 1
    mkdir -p $OUT

    echo ""
    echo "=== $ep  [1/4] kill maps   `date '+%H:%M:%S'`"
    rm -f $KM.json
    python build_kill_maps.py --epoch $ep $OPTS --out $OUT/killmaps |& tee hitemu/logs/${ep}_1_killmaps.log
    if ( -f $KM.json ) then
        set R1 = OK
    else
        set R1 = FAILED
        set GO = 0
    endif

    if ( $GO == 1 ) then
        echo "=== $ep  [2/4] no-L1 study, data   `date '+%H:%M:%S'`"
        rm -f $HD
        python noL1_data_study.py --epoch $ep $OPTS --killmaps $KM --split-test --out $OUT/noL1_data |& tee hitemu/logs/${ep}_2_noL1_data.log
        if ( -f $HD ) then
            set R2 = OK
        else
            set R2 = FAILED
            set GO = 0
        endif
    endif

    if ( $GO == 1 ) then
        echo "=== $ep  [3/4] no-L1 study, MC   `date '+%H:%M:%S'`"
        rm -f $HM
        python noL1_data_study.py --epoch $ep $OPTS --killmaps $KM --sample mc --split-test --out $OUT/noL1_mc |& tee hitemu/logs/${ep}_3_noL1_mc.log
        if ( -f $HM ) then
            set R3 = OK
        else
            set R3 = FAILED
            set GO = 0
        endif
    endif

    if ( $GO == 1 ) then
        echo "=== $ep  [4/4] emulation   `date '+%H:%M:%S'`"
        set FLOWS = ()
        set ROUTES = (B)
        set R4 = OK_B
        if ( -f $RUNS/$ep/mu1/task_0/covflow.json && -f $RUNS/$ep/mu2/task_0/covflow.json ) then
            set FLOWS = (--flow-dir "$RUNS/@epoch@/@mu@/task_0")
            set ROUTES = (A B)
            set R4 = OK_AB
        else
            echo "    no trained flows under $RUNS/$ep : route A skipped"
        endif
        set TREE = ()
        if ( $WRITE_TREE == 1 ) set TREE = (--write-tree)
        rm -f $OUT/emu/emu_$ep.json
        python emulate_hit_loss.py --epoch $ep $OPTS:q --killmaps $KM --route $ROUTES \
               --hit-errors $HM $HD $EXTRA_HITERR $FLOWS:q $TREE --out $OUT/emu |& tee hitemu/logs/${ep}_4_emulate.log
        if ( ! -f $OUT/emu/emu_$ep.json ) then
            # most likely no hit-error fit for a surface: keep the emulation and route A
            echo "    emulation with route B failed; retrying without route B"
            set R4 = FAILED
            set ROUTES = ()
            if ( $#FLOWS > 0 ) set ROUTES = (--route A)
            python emulate_hit_loss.py --epoch $ep $OPTS:q --killmaps $KM $ROUTES $FLOWS:q \
                   --out $OUT/emu |& tee hitemu/logs/${ep}_4b_emulate_noB.log
            if ( -f $OUT/emu/emu_$ep.json ) then
                set R4 = OK_noB
                if ( $#FLOWS > 0 ) set R4 = OK_A_noB
            endif
        endif
    endif

    @ MIN = ( `date +%s` - $T0 ) / 60
    printf "%-15s %-10s %-10s %-10s %-14s %8d\n" $ep $R1 $R2 $R3 $R4 $MIN >> $SUMMARY
    echo "=== $ep  done after $MIN min: killmaps $R1, noL1_data $R2, noL1_mc $R3, emulate $R4"
end

@ MIN_ALL = ( `date +%s` - $T_ALL ) / 60
echo ""
echo "============================================================"
echo " finished `date`, $MIN_ALL min in total"
echo "============================================================"
cat $SUMMARY
echo ""
echo "summary: $REPO/$SUMMARY"
grep -q FAILED $SUMMARY
if ( $status == 0 ) then
    echo "Some steps FAILED: see the end of their logs in hitemu/logs/."
    exit 2
endif
exit 0
