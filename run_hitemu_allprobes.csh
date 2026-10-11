#!/bin/tcsh
#
# run_hitemu_allprobes.csh -- the hit-loss emulation chain (as run_hitemu.csh),
# with the kill maps measured on ALL J/psi probes, prompt and non-prompt
# (HITEMU.md 3.6-3.8). Output in <REPO>/hitemu_allprobes/, so the default
# results in hitemu/ are not overwritten.
#
#   screen -S hitemu_ap
#   tcsh run_hitemu_allprobes.csh                 # all 7 epochs
#   tcsh run_hitemu_allprobes.csh 2025 2026       # only these
#
# What differs from run_hitemu.csh:
#   step 1 (kill maps) drops the decay-length cut of the covflow selection
#     (--drop-cut lxy), in data AND MC: the probes are all selected J/psi.
#     Steps 2-4 keep the covflow selection (they act on the tracks the flow
#     corrects).
#   the helix that finds the crossed pixel cells starts, in every step:
#     data: at the PV (pv_x, pv_y, pv_z + {mu}_dz) when the dimuon is not
#           displaced, lxy_sig < SIG_CUT; at the dimuon vertex (vx, vy, vz)
#           otherwise (--helix-origin-data pv_or_vertex)
#     MC  : at the dimuon vertex (the Hb MC is non-prompt;
#           --helix-origin-mc dimuon_vertex), or see MC_ORIGIN below
#   The maps store these origins; steps 2-4 stop if they are run with others.
#
# Per epoch, in <REPO>/hitemu_allprobes/<epoch>/ : killmaps/ noL1_data/ noL1_mc/ emu/
# Logs and summary in hitemu_allprobes/logs/. Everything else as run_hitemu.csh.

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
#   set EXTRA_HITERR = (hitemu_allprobes/2024/noL1_mc/hiterrors_mc_2024.json hitemu_allprobes/2024/noL1_data/hiterrors_data_2024.json)
set EXTRA_HITERR = ()

# output directory (relative to REPO)
set TOP = hitemu_allprobes

# data: PV below this decay-length significance lxy_sig = L_xy / sigma(L_xy),
# dimuon vertex above. 3: the dimuon vertex is compatible with the PV within
# 3 sigma. Below the cut the two points are within ~3 sigma (tens of um) of
# each other, far less than a pixel cell (0.83 cm x 0.13 rad): the cut value
# matters very little (HITEMU 3.8).
set SIG_CUT = 3

# MC helix origin:
#   dimuon_vertex  reco momentum from the reco dimuon vertex (default)
#   gen_kin        GENERATED momentum (mu{1,2}_gen_pt/eta/phi, charge from
#                  mu{1,2}_gen_pdgid) from the reco dimuon vertex
#   gen            generated momentum from the GENERATED vertex: needs
#                  mu{1,2}_gen_vx/vy/vz, PLACEHOLDER names, not in the ntuples
#                  yet (rename with --branch gen_vx=... in OPTS when they are)
# gen / gen_kin give the MC the true crossing, while the data keep the reco
# one: a diagnostic, see HITEMU 3.8, not the default.
set MC_ORIGIN = dimuon_vertex

# options passed to every script. Add e.g. --max-events 300000 for a quick pass.
set OPTS = (--helix-origin-data pv_or_vertex --origin-sig-cut $SIG_CUT --helix-origin-mc $MC_ORIGIN)

# extra options for step 1 only (build_kill_maps.py): the probe selection.
# --drop-cut lxy removes the decay-length term (lxy * cos2d / pt * 3.0969 > 0.008)
# from the common, data and MC selections of configs/run3_epochs.py: prompt
# J/psi become probes. Worth checking once with run_probe_split.csh that
# prompt and non-prompt probes see the same efficiencies.
set KM_OPTS = (--drop-cut lxy)

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
grep -q "pv_or_vertex" pixel_eff_maps.py
if ( $status != 0 ) then
    echo "ERROR: pixel_eff_maps.py predates the per-sample helix origins; update the repository"
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

mkdir -p $TOP/logs
set STAMP   = `date +%Y%m%d_%H%M%S`
set SUMMARY = $TOP/logs/summary_$STAMP.txt
set T_ALL   = `date +%s`
printf "%-15s %-10s %-10s %-10s %-14s %8s\n" epoch killmaps noL1_data noL1_mc emulate minutes > $SUMMARY

echo "============================================================"
echo " hit-loss emulation, ALL J/psi probes, over: $EPOCHS"
echo " output  : $REPO/$TOP"
echo " repo    : $REPO"
echo " flows   : $RUNS/<epoch>/<mu>/task_0"
echo " python  : `which python`"
echo " options : $OPTS"
echo " step 1  : $KM_OPTS"
echo " started : `date`"
echo "============================================================"

foreach ep ($EPOCHS)
    set T0  = `date +%s`
    set OUT = $TOP/$ep
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
    python build_kill_maps.py --epoch $ep $OPTS $KM_OPTS --out $OUT/killmaps |& tee $TOP/logs/${ep}_1_killmaps.log
    if ( -f $KM.json ) then
        set R1 = OK
    else
        set R1 = FAILED
        set GO = 0
    endif

    if ( $GO == 1 ) then
        echo "=== $ep  [2/4] no-L1 study, data   `date '+%H:%M:%S'`"
        rm -f $HD
        python noL1_data_study.py --epoch $ep $OPTS --killmaps $KM --split-test --out $OUT/noL1_data |& tee $TOP/logs/${ep}_2_noL1_data.log
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
        python noL1_data_study.py --epoch $ep $OPTS --killmaps $KM --sample mc --split-test --out $OUT/noL1_mc |& tee $TOP/logs/${ep}_3_noL1_mc.log
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
               --hit-errors $HM $HD $EXTRA_HITERR $FLOWS:q $TREE --out $OUT/emu |& tee $TOP/logs/${ep}_4_emulate.log
        if ( ! -f $OUT/emu/emu_$ep.json ) then
            # most likely no hit-error fit for a surface: keep the emulation and route A
            echo "    emulation with route B failed; retrying without route B"
            set R4 = FAILED
            set ROUTES = ()
            if ( $#FLOWS > 0 ) set ROUTES = (--route A)
            python emulate_hit_loss.py --epoch $ep $OPTS:q --killmaps $KM $ROUTES $FLOWS:q \
                   --out $OUT/emu |& tee $TOP/logs/${ep}_4b_emulate_noB.log
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
    echo "Some steps FAILED: see the end of their logs in $TOP/logs/."
    exit 2
endif
exit 0
